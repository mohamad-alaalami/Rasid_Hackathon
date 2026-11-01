"""Host-independent tests of the sandbox's seccomp program and self-check logic.

The exact BPF program the worker installs is executed by a small classic-BPF
interpreter over synthetic ``seccomp_data``, so these regressions are caught
on any OS, including hosts where the sandbox itself cannot run.
"""

import importlib
import os
import struct
import unittest

worker = importlib.import_module("ml.sandbox_worker")

X32_BIT = 0x40000000
CLONE_THREAD = 0x00010000
AUDIT_ARCH_I386 = 0x40000003


def run_bpf(program, nr, arch, arg0=0):
    """Execute a classic-BPF seccomp filter; returns the SECCOMP_RET_* value."""
    data = struct.pack("<iIQ6Q", nr, arch, 0, arg0, 0, 0, 0, 0, 0)
    accumulator, pc, steps = 0, 0, 0
    while True:
        steps += 1
        if steps > 10_000 or not 0 <= pc < len(program):
            raise AssertionError(f"filter ran off the program at pc={pc}")
        code, jt, jf, k = program[pc]
        if code == 0x20:  # BPF_LD | BPF_W | BPF_ABS
            accumulator = struct.unpack_from("<I", data, k)[0]
            pc += 1
        elif code in (0x15, 0x35, 0x45):  # JEQ, JGE (unsigned), JSET
            taken = {0x15: accumulator == k, 0x35: accumulator >= k,
                     0x45: bool(accumulator & k)}[code]
            pc += 1 + (jt if taken else jf)
        elif code == 0x06:  # BPF_RET | BPF_K
            return k
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")


class SeccompProgramTests(unittest.TestCase):
    def setUp(self):
        self.programs = {machine: worker.build_seccomp_program(machine)
                         for machine in worker.SYSCALLS}

    def test_importing_the_worker_has_no_side_effects(self):
        self.assertIsNone(worker._PROTO_IN)
        self.assertIsNone(worker._PROTO_OUT)
        os.fstat(1)  # stdout is still ours

    def test_x86_64_rejects_every_x32_syscall(self):
        program, table = self.programs["x86_64"]
        arch = table["arch"]
        # Ordinary allowed syscalls (read=0, getpid=39) pass; their x32 forms do not.
        self.assertEqual(run_bpf(program, 0, arch), worker.SECCOMP_ALLOW)
        self.assertEqual(run_bpf(program, 39, arch), worker.SECCOMP_ALLOW)
        for number in range(0, 1024):
            self.assertEqual(run_bpf(program, X32_BIT | number, arch), worker.SECCOMP_EPERM,
                             f"x32 syscall {number} was not rejected")
        for name, number in table["deny"].items():
            self.assertEqual(run_bpf(program, X32_BIT | number, arch), worker.SECCOMP_EPERM, name)
        x32_thread_clone = run_bpf(program, X32_BIT | table["clone"], arch, CLONE_THREAD)
        self.assertEqual(x32_thread_clone, worker.SECCOMP_EPERM)
        self.assertEqual(run_bpf(program, -1, arch), worker.SECCOMP_EPERM)

    def test_denied_syscalls_clone_and_architecture(self):
        for machine, (program, table) in self.programs.items():
            arch = table["arch"]
            with self.subTest(machine):
                for name, number in table["deny"].items():
                    self.assertEqual(run_bpf(program, number, arch), worker.SECCOMP_EPERM, name)
                self.assertEqual(run_bpf(program, table["clone3"], arch), worker.SECCOMP_ENOSYS)
                self.assertEqual(run_bpf(program, table["clone"], arch, CLONE_THREAD | 0x3D0F00),
                                 worker.SECCOMP_ALLOW)  # pthread_create flags
                self.assertEqual(run_bpf(program, table["clone"], arch, 0x11),
                                 worker.SECCOMP_EPERM)  # fork-style clone (SIGCHLD)
                self.assertEqual(run_bpf(program, 0, AUDIT_ARCH_I386), worker.SECCOMP_KILL)
        self.assertIn("socket", self.programs["x86_64"][1]["deny"])
        self.assertIn("execve", self.programs["aarch64"][1]["deny"])

    def test_unknown_architecture_fails_closed(self):
        with self.assertRaises(RuntimeError):
            worker.build_seccomp_program("riscv64")


GOOD_STATUS = "Seccomp:\t2\nNoNewPrivs:\t1\nCapEff:\t0000000000000000\nCapPrm:\t0000000000000000\nCapAmb:\t0000000000000000\n"
GOOD_MOUNTS = "\n".join([
    "1 0 0:1 / / ro,nosuid,nodev - tmpfs gotrain-root rw",
    "2 1 8:1 /usr/lib /usr/lib ro,nosuid,relatime - ext4 /dev/sda rw",
    "3 1 0:2 / /tmp rw,nosuid,nodev - tmpfs gotrain-tmp rw",
    "4 1 0:5 /null /dev/null rw,nosuid - devtmpfs udev rw",
    "5 1 0:3 / /proc ro,nosuid,nodev,noexec - proc proc rw",
])
GOOD_ENV = {"PATH": "/usr/bin", "HOME": "/tmp", "LANG": "C.UTF-8", "PWD": "/"}


class SelfCheckTests(unittest.TestCase):
    def test_isolated_session_passes(self):
        self.assertEqual(worker.self_check_failures(GOOD_STATUS, GOOD_MOUNTS, "0 1000 1",
                                                    GOOD_ENV), [])

    def test_each_missing_property_is_reported(self):
        cases = {
            "seccomp mode": (GOOD_STATUS.replace("Seccomp:\t2", "Seccomp:\t0"), GOOD_MOUNTS,
                             "0 1000 1", GOOD_ENV),
            "no_new_privs": (GOOD_STATUS.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0"),
                             GOOD_MOUNTS, "0 1000 1", GOOD_ENV),
            "CapEff": (GOOD_STATUS.replace("CapEff:\t0000000000000000", "CapEff:\t00000000a80425fb"),
                       GOOD_MOUNTS, "0 1000 1", GOOD_ENV),
            "host root": (GOOD_STATUS, GOOD_MOUNTS, "0 0 1", GOOD_ENV),
            "/usr/lib is writable": (GOOD_STATUS, GOOD_MOUNTS.replace(
                "/usr/lib ro,nosuid", "/usr/lib rw,nosuid"), "0 1000 1", GOOD_ENV),
            "AWS_ACCESS_KEY_ID": (GOOD_STATUS, GOOD_MOUNTS, "0 1000 1",
                                  {**GOOD_ENV, "AWS_ACCESS_KEY_ID": "x"}),
        }
        for expected, arguments in cases.items():
            with self.subTest(expected):
                failures = " | ".join(worker.self_check_failures(*arguments))
                self.assertIn(expected, failures)

    def test_missing_proc_data_fails_closed(self):
        self.assertTrue(worker.self_check_failures("", "", "", {}))


if __name__ == "__main__":
    unittest.main()
