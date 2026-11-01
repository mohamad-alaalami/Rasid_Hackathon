"""Isolated execution runtime for generated code (Linux namespaces + seccomp).

Generated code never runs in the agent, trainer, or evaluator process. It runs
in a worker (ml/sandbox_worker.py) launched in fresh Linux user, mount,
network, PID, IPC, and UTS namespaces with util-linux tools, the same kernel
primitives bubblewrap uses:

* identity: when the host process is root, the sandbox is started as uid/gid
  65534 (nobody) before the user namespace is created, so the sandbox's
  namespace-root never maps to host root.
* filesystem: a new tmpfs root (``pivot_root``; the host tree is detached)
  containing only read-only shared libraries, the Python environment, the
  dynamic-linker cache, /dev/null-style devices, a small private /tmp, a
  read-only /proc for the sandbox's own PID namespace, and a read-only /work
  with the worker and component. No /bin, /etc (beyond ld.so.cache), home
  directory, repository, dataset, or experiment output.
* network: an empty network namespace (no metadata service, no internet), and
  seccomp refuses socket creation outright.
* privileges: minimal environment, ``no_new_privs``, all capabilities dropped
  (bounding, inheritable, and ambient sets).
* processes: a seccomp filter installed by the trusted worker before torch or
  any generated code loads blocks fork, vfork, non-thread clone, exec, sockets,
  io_uring, x32-ABI syscalls, and namespace, mount, ptrace, bpf, keyring, and
  module syscalls (threads remain allowed). RLIMIT_NPROC is also set but is not
  enforced for a namespace-root process on every kernel, so seccomp is the
  control that matters.
* resources: RLIMIT_AS, RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_NOFILE, plus a
  host-side wall-clock timeout on every request; the whole process tree is
  killed on breach.

Execution is enabled only after an active probe confirms, from inside the
sandbox, that forbidden files, the network, sockets, a shell, fork, exec,
chroot, new namespaces, and capabilities are unavailable, that /proc/sys is not
writable, that the uid does not map to host root, and that the memory limit is
enforced. Otherwise every request is refused with IsolationUnavailable; there
is no unsandboxed fallback.

The host never unpickles sandbox output: replies are JSON headers plus raw
float32 bytes whose sizes are checked against the request.
"""

from __future__ import annotations

import json
import os
import platform
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("sandbox_worker.py")
MAX_HEADER = 1 << 20
HELPERS = ("unshare", "mount", "umount", "pivot_root", "setpriv", "prlimit")
UNPRIVILEGED_ID = 65534  # nobody/nogroup: used when the host process is root

LAUNCHER = r"""
set -eu
ROOT="$1"; PLAN="$2"; MOUNT="$3"; UMOUNT="$4"; shift 4
TAB="$(printf '\t')"
mount --make-rprivate /
mount -t tmpfs -o size=16m,mode=0755 gotrain-root "$ROOT"
while IFS="$TAB" read -r kind src dst opts; do
  case "$kind" in
    rdir) mkdir -p "$ROOT$dst"; mount --rbind "$src" "$ROOT$dst" ;;
    ro)   mount -o "remount,bind,$opts" "$ROOT$dst" ;;
    file) mkdir -p "$ROOT$(dirname "$dst")"; : > "$ROOT$dst"; mount --bind "$src" "$ROOT$dst"
          mount -o "remount,bind,$opts" "$ROOT$dst" ;;
    dev)  mkdir -p "$ROOT$(dirname "$dst")"; : > "$ROOT$dst"; mount --bind "$src" "$ROOT$dst" ;;
    link) mkdir -p "$ROOT$(dirname "$dst")"; ln -s "$src" "$ROOT$dst" ;;
    tmp)  mkdir -p "$ROOT$dst"; mount -t tmpfs -o "size=$src,mode=1777,nosuid,nodev" gotrain-tmp "$ROOT$dst" ;;
    proc) mkdir -p "$ROOT$dst"; mount -t proc -o ro,nosuid,nodev,noexec proc "$ROOT$dst" ;;
  esac
done < "$PLAN"
cd "$ROOT"
mkdir .old
pivot_root . .old
"/.old$MOUNT" -o remount,bind,ro,nosuid,nodev /
"/.old$UMOUNT" -l /.old
exec "$@"
"""


class IsolationUnavailable(RuntimeError):
    """No verified isolation runtime: generated code must not run."""


class SandboxError(RuntimeError):
    def __init__(self, error_type, message, detail=None):
        super().__init__(message)
        self.error_type = error_type
        self.detail = detail or {}

    def to_dict(self):
        return {"error_type": self.error_type, "message": str(self), **self.detail}


def _env(name, default, kind=int):
    return lambda: kind(os.environ.get(name, default))


@dataclass(frozen=True)
class Limits:
    memory_mb: int = field(default_factory=_env("GOTRAIN_SANDBOX_MEMORY_MB", "4096"))
    cpu_seconds: int = field(default_factory=_env("GOTRAIN_SANDBOX_CPU_SECONDS", "7200"))
    request_timeout: float = field(default_factory=_env("GOTRAIN_SANDBOX_TIMEOUT", "60", float))
    startup_timeout: float = 120.0
    threads: int = field(default_factory=_env("GOTRAIN_SANDBOX_THREADS", "2"))
    max_tensor_bytes: int = 1 << 30


@dataclass
class RuntimeReport:
    available: bool
    reason: str | None = None
    backend: str = "linux-namespaces+seccomp (util-linux unshare/pivot_root/setpriv/prlimit)"
    probe: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"available": self.available, "reason": self.reason, "backend": self.backend,
                "probe": self.probe, "limits": self.limits}


# ---------------------------------------------------------------------------
# Sandbox construction
# ---------------------------------------------------------------------------

def _kernel_counts_nproc_per_userns():
    try:
        major, minor = (int(part) for part in platform.release().split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (5, 14)


def _protected_paths():
    paths = {REPO_ROOT, Path.cwd().resolve(), Path.home().resolve(), Path("/")}
    for name in ("GEO_BENCH_DIR", "GOTRAIN_STORE", "GOTRAIN_SOLUTIONS_DIR",
                 "GOTRAIN_COMPONENTS_DIR"):
        if os.environ.get(name) and not os.environ[name].startswith("s3://"):
            paths.add(Path(os.environ[name]).expanduser().resolve())
    paths.add((Path.home() / ".aws").resolve())
    return paths


LOCKED_FLAGS = ("nosuid", "nodev", "noexec", "noatime", "nodiratime", "relatime", "strictatime")


def _mounts():
    """(mount point, per-mount options) from /proc/self/mountinfo."""
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        point = fields[4].encode().decode("unicode_escape")
        mounts.append((point, fields[5].split(",")))
    return mounts


def _readonly_options(path, mounts):
    """Read-only remount options that keep the mount's locked flags (userns rule)."""
    owner = max((m for m in mounts if path == m[0] or path.startswith(m[0].rstrip("/") + "/")),
                key=lambda m: len(m[0]), default=("/", []))
    flags = [flag for flag in owner[1] if flag in LOCKED_FLAGS]
    return ",".join(["ro", *sorted(set(flags) | {"nosuid"})])


def _expand_binds(rows):
    """Recursive read-only binds: every submount is remounted read-only too.

    Plan rows have four tab-separated fields; "-" marks an unused field because
    the shell's ``read`` collapses consecutive tabs.
    """
    mounts = _mounts()
    expanded = []
    for kind, source, destination in rows:
        if kind == "dir":
            expanded.append(("rdir", source, destination, "-"))
            expanded.append(("ro", "-", destination, _readonly_options(source, mounts)))
            for point, options in mounts:
                if point.startswith(source.rstrip("/") + "/"):
                    flags = sorted({f for f in options if f in LOCKED_FLAGS} | {"nosuid"})
                    expanded.append(("ro", "-", destination + point[len(source.rstrip("/")):],
                                     ",".join(["ro", *flags])))
        elif kind == "file":
            expanded.append(("file", source, destination, _readonly_options(source, mounts)))
        else:
            expanded.append((kind, source, destination, "-"))
    return expanded


def _bind_plan(work: Path, helpers: dict[str, str]):
    """(kind, source, destination, options) rows; refuses binds that expose protected paths."""
    rows = []
    for top in ("/lib", "/lib64", "/lib32", "/usr/lib", "/usr/lib64", "/usr/lib32"):
        path = Path(top)
        if path.is_symlink():
            rows.append(("link", os.readlink(path), top))
        elif path.is_dir():
            rows.append(("dir", top, top))
    bound = [Path(row[2]) for row in rows if row[0] == "dir"]

    def covered(path):
        return any(path == base or base in path.parents for base in bound)

    import importlib.util

    candidates = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    for module in ("torch", "numpy"):
        spec = importlib.util.find_spec(module)
        if spec and spec.origin:
            candidates.append(Path(spec.origin).resolve().parents[1])
    for prefix in candidates:
        # System prefixes hold /usr/bin and friends; expose only their libraries.
        targets = ([prefix / "lib", prefix / "lib64"] if str(prefix) in ("/usr", "/usr/local", "/")
                   else [prefix])
        for target in targets:
            if target.is_dir() and not covered(target):
                rows.append(("dir", str(target), str(target)))
                bound.append(target)
    # Recreate the interpreter's symlink chain (venv bin/python -> /usr/bin/...).
    executable = Path(os.path.abspath(sys.executable))
    for _ in range(16):
        if not executable.is_symlink():
            break
        target = os.readlink(executable)
        if not covered(executable):
            rows.append(("link", target, str(executable)))
        executable = Path(os.path.normpath(executable.parent / target))
    if not covered(executable):
        rows.append(("file", str(executable), str(executable)))
    if Path("/etc/ld.so.cache").is_file():
        rows.append(("file", "/etc/ld.so.cache", "/etc/ld.so.cache"))
    for name in ("null", "zero", "urandom", "random"):
        rows.append(("dev", f"/dev/{name}", f"/dev/{name}"))
    rows += [("tmp", "64m", "/tmp"), ("proc", "proc", "/proc"),
             ("dir", str(work), "/work")]
    for name, path in helpers.items():
        if name in ("setpriv", "prlimit"):
            rows.append(("file", path, f"/.gotrain/{name}"))

    protected = _protected_paths()
    for kind, source, _ in rows:
        if kind not in ("dir", "file") or source == str(work):
            continue
        resolved = Path(source).resolve()
        for guarded in protected:
            if resolved == guarded or resolved in guarded.parents:
                raise IsolationUnavailable(
                    f"refusing to expose {resolved}: it contains protected path {guarded}")
    return _expand_binds(rows)


def _launch(work: Path, limits: Limits):
    if sys.platform != "linux":
        raise IsolationUnavailable(f"generated code needs Linux namespaces; this host is {sys.platform}")
    if os.environ.get("GOTRAIN_SANDBOX_DISABLE"):
        raise IsolationUnavailable("disabled by GOTRAIN_SANDBOX_DISABLE")
    helpers = {name: shutil.which(name, path="/usr/bin:/bin:/usr/sbin:/sbin") for name in HELPERS}
    missing = [name for name, path in helpers.items() if not path]
    if missing:
        raise IsolationUnavailable(f"missing util-linux tools: {missing}")
    plan = _bind_plan(work, helpers)
    root = work.parent / "root"
    root.mkdir()
    plan_path = work.parent / "plan.tsv"
    plan_path.write_text("".join("\t".join(row) + "\n" for row in plan), encoding="utf-8")
    plan_path.chmod(0o644)
    rlimits = [f"--as={limits.memory_mb << 20}", f"--cpu={limits.cpu_seconds}",
               "--core=0", f"--fsize={64 << 20}", "--nofile=64"]
    if _kernel_counts_nproc_per_userns():
        rlimits.append(f"--nproc={limits.threads + 16}")
    setpriv_help = subprocess.run([helpers["setpriv"], "--help"], capture_output=True, text=True).stdout
    caps = ["--inh-caps=-all", "--bounding-set=-all"]
    if "--ambient-caps" in setpriv_help:
        caps.append("--ambient-caps=-all")
    command = []
    if os.geteuid() == 0:
        # Never let the sandbox's namespace-root map to host root.
        command += [helpers["setpriv"], f"--reuid={UNPRIVILEGED_ID}",
                    f"--regid={UNPRIVILEGED_ID}", "--clear-groups", "--"]
    command += [
        helpers["unshare"], "--user", "--map-root-user", "--mount", "--net", "--pid",
        "--ipc", "--uts", "--fork",
    ]
    if "--kill-child" in subprocess.run([helpers["unshare"], "--help"], capture_output=True,
                                        text=True).stdout:
        command.append("--kill-child")
    command += [
        "/bin/sh", "-c", LAUNCHER, "gotrain-sandbox", str(root), str(plan_path),
        helpers["mount"], helpers["umount"],
        "/.gotrain/setpriv", "--no-new-privs", *caps, "--",
        "/.gotrain/prlimit", *rlimits, "--",
        sys.executable, "-I", "-B", "/work/sandbox_worker.py",
    ]
    environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/tmp", "LANG": "C.UTF-8",
                   "OMP_NUM_THREADS": str(limits.threads), "MKL_NUM_THREADS": str(limits.threads),
                   "GOTRAIN_SANDBOX_THREADS": str(limits.threads), "PYTHONHASHSEED": "0"}
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=environment, start_new_session=True,
                            cwd="/")


# ---------------------------------------------------------------------------
# Session: one sandboxed worker
# ---------------------------------------------------------------------------

class SandboxSession:
    """A running sandboxed worker. Use as a context manager; always closed."""

    def __init__(self, files: dict[str, str] | None = None, limits: Limits | None = None):
        self.limits = limits or Limits()
        self.process = None
        self._stderr = b""
        self._temp = tempfile.mkdtemp(prefix="gotrain-sandbox-")
        # Traversable (not listable) by the unprivileged sandbox user; nothing secret here.
        os.chmod(self._temp, 0o711)
        work = Path(self._temp) / "work"
        work.mkdir()
        work.chmod(0o755)
        shutil.copyfile(WORKER, work / "sandbox_worker.py")
        for name, text in (files or {}).items():
            (work / name).write_text(text, encoding="utf-8")
        for path in work.iterdir():
            path.chmod(0o444)
        try:
            self.process = _launch(work, self.limits)
        except BaseException:
            self.close()
            raise
        # Drain stderr continuously so a chatty worker cannot block on a full pipe.
        self._stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True)
        self._stderr_thread.start()
        # Every session must pass the worker's own isolation self-check before use.
        try:
            ready = self._read_reply(time.monotonic() + self.limits.startup_timeout, None)
        except BaseException:
            self.close()
            raise
        if ready.get("status") != "ready":
            self.close()
            raise SandboxError(ready.get("error_type", "SANDBOX_SELF_CHECK"),
                               f"sandbox failed its self-check: {ready.get('message')}")
        self.self_check = ready

    def _collect_stderr(self):
        stream = self.process.stderr
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
            if not chunk:
                return
            self._stderr = (self._stderr + chunk)[-20000:]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- framing -------------------------------------------------------------------

    def _read_exact(self, count, deadline):
        stream = self.process.stdout
        data = b""
        while len(data) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SandboxError("SANDBOX_TIMEOUT",
                                   f"no response within the {self.limits.request_timeout:.0f}s limit")
            ready, _, _ = select.select([stream], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(stream.fileno(), count - len(data))
            if not chunk:
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                raise SandboxError("SANDBOX_CRASHED", "the sandboxed worker exited",
                                   {"returncode": self.process.poll(),
                                    "stderr_tail": self.stderr_tail()})
            data += chunk
        return data

    def _read_reply(self, deadline, expect):
        """Read one header-only frame (the startup handshake)."""
        try:
            (length,) = struct.unpack(">I", self._read_exact(4, deadline))
            if length > MAX_HEADER:
                raise SandboxError("SANDBOX_PROTOCOL", "handshake header too large")
            reply = json.loads(self._read_exact(length, deadline))
            if not isinstance(reply, dict) or reply.get("buffers"):
                raise SandboxError("SANDBOX_PROTOCOL", "malformed handshake")
            return reply
        except SandboxError:
            self.kill()
            raise
        except (ValueError, TypeError, struct.error) as exc:
            self.kill()
            raise SandboxError("SANDBOX_PROTOCOL", f"malformed handshake: {exc}") from exc

    def request(self, header, buffers=(), timeout=None, expect=None):
        """Send one request; validate reply sizes against ``expect`` (list of byte counts)."""
        if self.process is None or self.process.poll() is not None:
            raise SandboxError("SANDBOX_CRASHED", "the sandboxed worker is not running",
                               {"stderr_tail": self.stderr_tail()})
        encoded = json.dumps(dict(header, buffers=[{"bytes": len(b)} for b in buffers])).encode()
        frame = [struct.pack(">I", len(encoded)), encoded]
        for buffer in buffers:
            frame += [struct.pack(">Q", len(buffer)), buffer]
        try:
            self.process.stdin.write(b"".join(frame))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.kill()
            raise SandboxError("SANDBOX_CRASHED", f"cannot write to the worker: {exc}",
                               {"stderr_tail": self.stderr_tail()}) from exc
        deadline = time.monotonic() + (timeout or self.limits.request_timeout)
        try:
            (length,) = struct.unpack(">I", self._read_exact(4, deadline))
            if length > MAX_HEADER:
                raise SandboxError("SANDBOX_PROTOCOL", "reply header too large")
            reply = json.loads(self._read_exact(length, deadline))
            if not isinstance(reply, dict):
                raise SandboxError("SANDBOX_PROTOCOL", "reply header is not an object")
            sizes = [int(item.get("bytes", -1)) for item in reply.get("buffers", [])]
            if reply.get("status") == "ok" and expect is not None and sizes != list(expect):
                raise SandboxError("SANDBOX_PROTOCOL",
                                   f"reply buffers {sizes} do not match the expected {list(expect)}")
            if any(size < 0 or size > self.limits.max_tensor_bytes for size in sizes):
                raise SandboxError("SANDBOX_PROTOCOL", "reply buffer size out of bounds")
            out = []
            for _ in sizes:
                (size,) = struct.unpack(">Q", self._read_exact(8, deadline))
                if size > self.limits.max_tensor_bytes:
                    raise SandboxError("SANDBOX_PROTOCOL", "reply buffer too large")
                out.append(self._read_exact(size, deadline))
            return reply, out
        except SandboxError:
            self.kill()
            raise
        except (ValueError, TypeError, AttributeError, struct.error) as exc:
            self.kill()
            raise SandboxError("SANDBOX_PROTOCOL", f"malformed reply: {exc}") from exc

    # -- lifecycle -----------------------------------------------------------------

    def stderr_tail(self):
        time.sleep(0.05)  # let the drain thread catch up with a just-exited worker
        return self._stderr.decode("utf-8", "replace")[-4000:]

    def kill(self):
        """Kill the sandbox's whole process tree (namespace init and descendants)."""
        if self.process is None:
            return
        pid = self.process.pid
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            children = []
        for child in children:  # PID-namespace init: its death kills the namespace
            try:
                os.kill(int(child), signal.SIGKILL)
            except OSError:
                pass
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.request({"op": "shutdown"}, timeout=5)
            except SandboxError:
                pass
            self.kill()
        shutil.rmtree(self._temp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Verified runtime
# ---------------------------------------------------------------------------

REQUIRED_BLOCKS = ("write_root", "write_work", "write_proc_sys", "read_etc_passwd",
                   "network_metadata_service", "network_public", "dns", "unix_socket",
                   "spawn_shell", "chroot", "allocate_over_limit", "fork", "exec_python",
                   "exec_in_place", "new_user_namespace")
_CACHE: dict[tuple, RuntimeReport] = {}


def _probe_failures(reply) -> list[str]:
    results = reply.get("results", {})
    failures = [name for name, outcome in results.items()
                if (name in REQUIRED_BLOCKS or name.startswith(("read_forbidden_",
                                                                "list_forbidden_")))
                and not outcome["blocked"]]
    failures += [f"missing_probe:{name}" for name in REQUIRED_BLOCKS if name not in results]
    if reply.get("effective_capabilities") != 0:
        failures.append(f"capabilities={reply.get('effective_capabilities')!r}")
    if not reply.get("seccomp"):
        failures.append("seccomp_not_installed")
    uid_map = reply.get("uid_map")
    if not uid_map or len(uid_map) < 2 or uid_map[1] == "0":
        failures.append(f"uid_maps_to_host_root:{uid_map!r}")
    failures += [k for k in reply.get("environment", []) if k.startswith("AWS")]
    if not reply.get("torch_ok"):
        failures.append("torch_unusable")
    return failures


def check_runtime(limits: Limits | None = None, refresh=False) -> RuntimeReport:
    """Launch a sandbox, run the trusted probe, and enable execution only if every
    escape attempt was blocked. Cached per process and limit set."""
    limits = limits or Limits()
    key = (limits.memory_mb, limits.cpu_seconds, os.environ.get("GOTRAIN_SANDBOX_DISABLE"))
    if key in _CACHE and not refresh:
        return _CACHE[key]
    report = RuntimeReport(available=False, limits=limits.__dict__.copy())
    if sys.platform != "linux" or os.environ.get("GOTRAIN_SANDBOX_DISABLE"):
        # Refuse before constructing anything; nothing is launched.
        report.reason = ("disabled by GOTRAIN_SANDBOX_DISABLE" if os.environ.get(
            "GOTRAIN_SANDBOX_DISABLE") else f"generated code needs Linux namespaces; this host is "
            f"{sys.platform}")
        _CACHE[key] = report
        return report
    canary_dir = Path(tempfile.mkdtemp(prefix="gotrain-canary-"))
    canary = canary_dir / "secret.txt"
    canary.write_text("gotrain-canary", encoding="utf-8")
    forbidden = [str(canary), str(REPO_ROOT / "ml" / "test_evaluation.py"),
                 str(REPO_ROOT / "artifacts"), str(Path.home() / ".aws" / "credentials")]
    if os.environ.get("GEO_BENCH_DIR"):
        forbidden.append(os.environ["GEO_BENCH_DIR"])
    try:
        with SandboxSession(limits=limits) as session:
            reply, _ = session.request(
                {"op": "probe", "forbidden_paths": forbidden, "memory_limit_mb": limits.memory_mb},
                timeout=limits.startup_timeout, expect=[])
        report.probe = reply
        if reply.get("status") != "ok":
            report.reason = f"probe failed: {reply.get('message')}"
        else:
            failures = _probe_failures(reply)
            if failures:
                report.reason = f"isolation probe found leaks: {failures}"
            else:
                report.available = True
    except IsolationUnavailable as exc:
        report.reason = str(exc)
    except SandboxError as exc:
        report.reason = f"sandbox could not start: {exc} {exc.detail.get('stderr_tail', '')[-800:]}"
    finally:
        shutil.rmtree(canary_dir, ignore_errors=True)
    _CACHE[key] = report
    return report


def require_runtime(limits: Limits | None = None) -> RuntimeReport:
    report = check_runtime(limits)
    if not report.available:
        raise IsolationUnavailable(report.reason or "isolation runtime unavailable")
    return report


if __name__ == "__main__":  # python -m ml.sandbox: print the verified runtime report
    result = check_runtime().to_dict()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["available"] else 1)
