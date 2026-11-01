"""Worker that runs INSIDE the isolation sandbox (see ml/sandbox.py).

This file is copied into the sandbox's read-only /work directory and run as
``python -I -B /work/sandbox_worker.py``. It must depend only on the standard
library and torch: the sandbox cannot see the repository.

It is deliberately dumb. The host generates every input and checks every
output; this process only executes the generated loss on host-provided
tensors and returns the scalar value and the gradient with respect to the
logits. Nothing it reports is trusted without host-side checks.

Protocol (both directions): 4-byte big-endian header length, JSON header,
then for each entry in header["buffers"] an 8-byte length and raw bytes.
"""

import json
import os
import struct
import sys
import traceback

MAX_HEADER = 1 << 20

# Keep the protocol on private descriptors so prints from generated code
# (fd 1) cannot corrupt it; they go to stderr instead.
_PROTO_IN = os.fdopen(os.dup(0), "rb", buffering=0)
_PROTO_OUT = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
_devnull = os.open(os.devnull, os.O_RDONLY) if os.path.exists(os.devnull) else None
if _devnull is not None:
    os.dup2(_devnull, 0)


def _read_exact(count):
    data = b""
    while len(data) < count:
        chunk = _PROTO_IN.read(count - len(data))
        if not chunk:
            raise EOFError("host closed the channel")
        data += chunk
    return data


def receive():
    (length,) = struct.unpack(">I", _read_exact(4))
    if length > MAX_HEADER:
        raise ValueError("header too large")
    header = json.loads(_read_exact(length))
    buffers = []
    for _ in header.get("buffers", []):
        (size,) = struct.unpack(">Q", _read_exact(8))
        buffers.append(_read_exact(size))
    return header, buffers


def send(header, buffers=()):
    header = dict(header, buffers=[{"bytes": len(b)} for b in buffers])
    encoded = json.dumps(header).encode()
    parts = [struct.pack(">I", len(encoded)), encoded]
    for buffer in buffers:
        parts += [struct.pack(">Q", len(buffer)), buffer]
    _PROTO_OUT.write(b"".join(parts))
    _PROTO_OUT.flush()


def _error(kind, exc):
    return {"status": "error", "error_type": kind, "exception": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:]}


# ---------------------------------------------------------------------------
# seccomp: no new processes, no exec, no sockets, no namespace/mount/ptrace/bpf/
# io_uring syscalls. Installed first in main(), before torch is imported or any
# thread exists, so every later thread inherits it. Threads (clone with
# CLONE_THREAD) are allowed; clone3 returns ENOSYS so libc falls back to clone.
# ---------------------------------------------------------------------------

SYSCALLS = {
    "x86_64": {"arch": 0xC000003E, "clone": 56, "clone3": 435, "x32_bit": 0x40000000, "deny": {
        "fork": 57, "vfork": 58, "execve": 59, "execveat": 322, "unshare": 272, "setns": 308,
        "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161, "ptrace": 101,
        "process_vm_readv": 310, "process_vm_writev": 311,
        "keyctl": 250, "add_key": 248, "request_key": 249, "bpf": 321, "perf_event_open": 298,
        "userfaultfd": 323, "init_module": 175, "finit_module": 313, "kexec_load": 246,
        "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432,
        "mount_setattr": 442, "io_uring_setup": 425, "io_uring_enter": 426,
        "io_uring_register": 427, "socket": 41, "socketpair": 53}},
    "aarch64": {"arch": 0xC00000B7, "clone": 220, "clone3": 435, "x32_bit": None, "deny": {
        "execve": 221, "execveat": 281, "unshare": 97, "setns": 268, "mount": 40, "umount2": 39,
        "pivot_root": 41, "chroot": 51, "ptrace": 117, "process_vm_readv": 270,
        "process_vm_writev": 271, "keyctl": 219, "add_key": 217,
        "request_key": 218, "bpf": 280, "perf_event_open": 241, "userfaultfd": 282,
        "init_module": 105, "finit_module": 273, "kexec_load": 104, "open_tree": 428,
        "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432,
        "mount_setattr": 442, "io_uring_setup": 425, "io_uring_enter": 426,
        "io_uring_register": 427, "socket": 198, "socketpair": 199}},
}


def install_seccomp():
    import ctypes
    import platform

    table = SYSCALLS.get(platform.machine())
    if table is None:
        raise RuntimeError(f"no seccomp policy for architecture {platform.machine()!r}")
    load, jeq, jge, jset, ret = 0x20, 0x15, 0x35, 0x45, 0x06
    allow, kill = 0x7FFF0000, 0x80000000
    eperm, enosys = 0x00050000 | 1, 0x00050000 | 38
    # seccomp_data: nr at offset 0, arch at 4, args[0] (low word) at 16.
    program = [(load, 0, 0, 4), (jeq, 1, 0, table["arch"]), (ret, 0, 0, kill), (load, 0, 0, 0)]
    if table["x32_bit"]:  # x32 ABI syscalls share the x86_64 arch value: refuse them all
        program += [(jge, 0, 1, table["x32_bit"]), (ret, 0, 0, eperm)]
    program += [(jeq, 0, 1, table["clone3"]), (ret, 0, 0, enosys)]
    for number in table["deny"].values():
        program += [(jeq, 0, 1, number), (ret, 0, 0, eperm)]
    program += [(jeq, 0, 4, table["clone"]), (load, 0, 0, 16), (jset, 0, 1, 0x00010000),
                (ret, 0, 0, allow), (ret, 0, 0, eperm), (ret, 0, 0, allow)]

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte),
                    ("k", ctypes.c_uint)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    filters = (Filter * len(program))(*[Filter(*item) for item in program])
    fprog = Program(len(program), filters)  # must stay alive until prctl returns
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_NO_NEW_PRIVS (38), then PR_SET_SECCOMP (22) with SECCOMP_MODE_FILTER (2).
    for args in ((38, 1, 0, 0, 0), (22, 2, ctypes.addressof(fprog), 0, 0)):
        if libc.prctl(*[ctypes.c_ulong(value) for value in args]) != 0:
            raise OSError(ctypes.get_errno(), "prctl failed while installing seccomp")
    return {"architecture": platform.machine(), "denied": sorted(table["deny"]),
            "clone": "threads only", "clone3": "ENOSYS",
            "x32_abi": "denied" if table["x32_bit"] else "n/a"}


# ---------------------------------------------------------------------------
# Isolation probe: trusted checks run before any generated code is loaded.
# ---------------------------------------------------------------------------

def probe(request):
    results = {}

    def attempt(name, action):
        try:
            value = action()
            results[name] = {"blocked": False, "detail": repr(value)[:200]}
        except BaseException as exc:  # noqa: BLE001 - every failure means blocked
            results[name] = {"blocked": True, "detail": f"{type(exc).__name__}: {exc}"[:200]}

    for index, path in enumerate(request.get("forbidden_paths", [])):
        attempt(f"read_forbidden_{index}", lambda p=path: open(p, "rb").read(16))
        attempt(f"list_forbidden_{index}", lambda p=path: os.listdir(os.path.dirname(p) or "/"))
    attempt("write_root", lambda: open("/gotrain-escape", "wb").write(b"x"))
    attempt("write_work", lambda: open("/work/gotrain-escape", "wb").write(b"x"))
    attempt("write_proc_sys", lambda: open("/proc/sys/kernel/core_pattern", "w").write("x"))
    attempt("read_etc_passwd", lambda: open("/etc/passwd").read(16))

    import socket

    def connect(host, port):
        with socket.create_connection((host, port), timeout=2):
            return "connected"
    attempt("network_metadata_service", lambda: connect("169.254.169.254", 80))
    attempt("network_public", lambda: connect("1.1.1.1", 443))
    attempt("dns", lambda: socket.getaddrinfo("amazonaws.com", 443))
    attempt("unix_socket", lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))

    import subprocess

    attempt("spawn_shell", lambda: subprocess.run(["/bin/sh", "-c", "true"], check=True))
    attempt("chroot", lambda: os.chroot("/tmp"))

    def fork():
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        return pid
    attempt("fork", fork)
    attempt("exec_python", lambda: subprocess.run([sys.executable, "-c", "pass"], check=True))
    attempt("exec_in_place", lambda: os.execv(sys.executable, [sys.executable, "-c", "pass"]))

    def new_user_namespace():
        if hasattr(os, "unshare"):
            return os.unshare(os.CLONE_NEWUSER)
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.unshare(0x10000000) != 0:
            raise OSError(ctypes.get_errno(), "unshare failed")
        return "unshared"
    attempt("new_user_namespace", new_user_namespace)

    import threading

    worker_thread = threading.Thread(target=lambda: None)
    worker_thread.start()
    worker_thread.join()  # threads must still work for torch

    def capabilities():
        with open("/proc/self/status") as handle:
            line = next(l for l in handle if l.startswith("CapEff:"))
        return int(line.split()[1], 16)
    try:
        effective_capabilities = capabilities()
    except Exception:  # /proc unavailable: report unknown, the host treats it as a failure
        effective_capabilities = None

    limit = int(request.get("memory_limit_mb", 0))
    if limit:
        attempt("allocate_over_limit", lambda: len(bytearray((limit + 256) << 20)))
    environment = sorted(os.environ)
    root_entries = sorted(os.listdir("/"))
    import torch

    tensor_ok = float(torch.ones(16, 1024, 1024).sum()) == 16 * 1024 * 1024
    try:
        with open("/proc/self/uid_map") as handle:
            uid_map = handle.read().split()  # [inside, outside, count]
    except OSError:
        uid_map = None
    try:
        import resource

        limits = {name: resource.getrlimit(getattr(resource, f"RLIMIT_{name}"))[0]
                  for name in ("AS", "CPU", "NPROC", "NOFILE", "FSIZE", "CORE")}
    except Exception:  # resource is POSIX-only
        limits = None
    return {"status": "ok", "results": results, "environment": environment, "rlimits": limits,
            "seccomp": _SECCOMP, "effective_capabilities": effective_capabilities,
            "root_entries": root_entries, "torch": torch.__version__, "torch_ok": tensor_ok,
            "uid_inside": os.getuid(), "uid_map": uid_map}


# ---------------------------------------------------------------------------
# Generated component execution
# ---------------------------------------------------------------------------

_COMPONENT = {}


def load_component():
    with open("/work/component.py", encoding="utf-8") as handle:
        source = handle.read()
    with open("/work/params.json", encoding="utf-8") as handle:
        params = json.load(handle)
    namespace = {"__name__": "gotrain_custom_component"}
    exec(compile(source, "component.py", "exec"), namespace)  # inside the sandbox only
    function = namespace.get("compute_loss")
    if not callable(function):
        raise TypeError("component must define a callable compute_loss(...)")
    _COMPONENT.update(function=function, params=params)


def loss(request, buffers):
    import torch

    def tensor(index, dtype):
        spec = request["inputs"][index]
        array = torch.frombuffer(bytearray(buffers[index]), dtype=dtype)
        return array.reshape(spec["shape"])

    logits = tensor(0, torch.float32).clone().requires_grad_(bool(request["need_grad"]))
    targets = tensor(1, torch.int64).clone()
    weights = tensor(2, torch.float32).clone() if len(buffers) > 2 else None
    torch.manual_seed(int(request.get("seed", 0)))
    value = _COMPONENT["function"](
        logits, targets, task=request["task"], class_weights=weights,
        ignore_index=request.get("ignore_index"), **_COMPONENT["params"],
    )
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise TypeError(f"compute_loss must return a 0-dim tensor, got "
                        f"{type(value).__name__}{tuple(getattr(value, 'shape', ()))}")
    grad = b""
    if request["need_grad"]:
        value.backward()
        if logits.grad is None:
            raise ValueError("the loss does not depend on the logits (no gradient)")
        grad = logits.grad.detach().to(torch.float32).contiguous().numpy().tobytes()
    return {"status": "ok", "loss": float(value.detach())}, ([grad] if grad else [])


_SECCOMP = None


def main():
    global _SECCOMP
    try:
        _SECCOMP = install_seccomp()
    except Exception as exc:  # fail closed: never serve requests without the filter
        send(_error("SANDBOX_SECCOMP", exc))
        return
    threads = int(os.environ.get("GOTRAIN_SANDBOX_THREADS", "1"))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception as exc:  # torch unusable inside the sandbox
        send(_error("SANDBOX_RUNTIME", exc))
        return
    loaded = None
    while True:
        try:
            request, buffers = receive()
        except EOFError:
            return
        op = request.get("op")
        if op == "shutdown":
            send({"status": "ok"})
            return
        try:
            if op == "probe":
                send(probe(request))
            elif op == "load":
                if loaded:
                    raise RuntimeError("a component is already loaded in this sandbox")
                load_component()
                loaded = True
                send({"status": "ok"})
            elif op == "loss":
                if not loaded:
                    raise RuntimeError("component not loaded")
                header, out = loss(request, buffers)
                send(header, out)
            else:
                send({"status": "error", "error_type": "PROTOCOL", "message": f"unknown op {op!r}"})
        except BaseException as exc:  # noqa: BLE001 - report every component failure
            kind = "COMPONENT_LOAD" if op == "load" else "COMPONENT_EXECUTION"
            if isinstance(exc, MemoryError):
                kind = "SANDBOX_MEMORY_LIMIT"
            send(_error(kind, exc))


if __name__ == "__main__":
    main()
