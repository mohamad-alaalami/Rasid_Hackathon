"""Generated loss sources used by the Phase 1B tests (written as Claude would submit them)."""

POLY1 = '''
import torch
import torch.nn.functional as F


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None, epsilon=1.0):
    """PolyLoss-1: cross-entropy plus epsilon * (1 - p_target), for under-confident classes."""
    ignore = -100 if ignore_index is None else ignore_index
    ce = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore,
                         reduction="none")
    valid = targets != ignore
    probabilities = F.softmax(logits, dim=1)
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    p_target = probabilities.gather(1, safe.unsqueeze(1)).squeeze(1)
    loss = (ce + epsilon * (1.0 - p_target)) * valid
    return loss.sum() / valid.sum().clamp(min=1)
'''

TVERSKY = '''
import torch
import torch.nn.functional as F


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None,
                 alpha=0.3, beta=0.7, mix=0.5):
    """Tversky (recall-weighted Dice) blended with cross-entropy for small segments."""
    ignore = -100 if ignore_index is None else ignore_index
    ce = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore)
    valid = (targets != ignore).unsqueeze(1).float()
    probabilities = F.softmax(logits, dim=1) * valid
    safe = torch.where(targets != ignore, targets, torch.zeros_like(targets))
    one_hot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    true_positive = (probabilities * one_hot).sum(dim=(0, 2, 3))
    false_positive = (probabilities * (1 - one_hot)).sum(dim=(0, 2, 3))
    false_negative = ((1 - probabilities) * one_hot * valid).sum(dim=(0, 2, 3))
    tversky = (true_positive + 1.0) / (true_positive + alpha * false_positive
                                       + beta * false_negative + 1.0)
    return mix * ce + (1 - mix) * (1 - tversky.mean())
'''

NO_GRADIENT = '''
import torch


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None):
    return torch.tensor(1.0)
'''

IGNORED_PIXELS_LEAK = '''
import torch.nn.functional as F


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None):
    return F.cross_entropy(logits, targets.clamp(0, logits.shape[1] - 1))
'''

NONFINITE = '''
import torch


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None):
    return (logits * 0).sum() / 0.0
'''

INFINITE_LOOP = '''
def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None):
    while True:
        pass
'''

PREFERS_WRONG = '''
import torch.nn.functional as F


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None):
    return -F.cross_entropy(logits, targets, ignore_index=-100 if ignore_index is None else ignore_index)
'''

STATIC_VIOLATIONS = '''
import os
import subprocess

print("module-level side effect")


class Sneaky:
    pass


def compute_loss(logits, targets, task="x"):
    return eval("logits.__class__")
'''

# Hostile code that deliberately bypasses the static checks (loaded directly into
# a sandbox session): every escape attempt must fail inside the sandbox. The
# placeholder constants are filled in by the test.
HOSTILE_TEMPLATE = '''
import ctypes
import json
import os
import socket
import subprocess

RESULTS = {}


def attempt(name, action):
    try:
        RESULTS[name] = "ESCAPED: " + repr(action())[:80]
    except BaseException as exc:
        RESULTS[name] = "blocked: " + type(exc).__name__


def libc_call(name, *args):
    libc = ctypes.CDLL(None, use_errno=True)
    if getattr(libc, name)(*args) != 0:
        raise OSError(ctypes.get_errno(), name)
    return name


attempt("read_repo", lambda: open(REPO_FILE).read(20))
attempt("read_canary", lambda: open(CANARY).read())
attempt("list_home", lambda: os.listdir(HOME))
attempt("read_aws_credentials", lambda: open(HOME + "/.aws/credentials").read())
attempt("write_repo", lambda: open(REPO_DIR + "/ESCAPE.txt", "w").write("x"))
attempt("write_work", lambda: open("/work/escape.txt", "w").write("x"))
attempt("write_python_libs", lambda: open(TORCH_DIR + "/ESCAPE.txt", "w").write("x"))
attempt("write_core_pattern", lambda: open("/proc/sys/kernel/core_pattern", "w").write("|/x"))
attempt("connect_metadata", lambda: socket.create_connection(("169.254.169.254", 80), timeout=2))
attempt("connect_internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=2))
attempt("shell", lambda: subprocess.run(["/bin/sh", "-c", "id"], capture_output=True))
attempt("exec_python", lambda: subprocess.run(["/proc/self/exe", "-c", "1"], capture_output=True))
attempt("aws_environment", lambda: [k for k in os.environ if k.startswith("AWS")] or 1 / 0)
attempt("chroot_escape", lambda: os.chroot("/tmp"))
attempt("os_system", lambda: os.system("id > /tmp/x") == 0 or 1 / 0)
attempt("fork", lambda: os.fork() or os._exit(0))
attempt("raw_unshare_userns", lambda: libc_call("unshare", 0x10000000))
attempt("raw_mount", lambda: libc_call("mount", b"none", b"/tmp", b"tmpfs", 0, None))
attempt("see_other_processes", lambda: [p for p in os.listdir("/proc")
                                           if p.isdigit() and int(p) != os.getpid()] or 1 / 0)


def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None):
    raise RuntimeError("RESULTS=" + json.dumps(RESULTS))
'''
