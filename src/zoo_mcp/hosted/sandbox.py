"""Linux Landlock filesystem confinement for credential-bearing CAD workers."""

import ctypes
import os
import sys
from pathlib import Path


def confine(workspace: Path, *, unsafe_local_dev: bool = False) -> None:
    if sys.platform != "linux":
        if unsafe_local_dev:
            return
        raise RuntimeError("Hosted modeling requires Linux with Landlock support")
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux assigns the same Landlock syscall numbers on x86_64 and aarch64.
    create, add, restrict = 444, 445, 446
    abi = libc.syscall(create, 0, 0, 1)
    if abi < 3:
        if unsafe_local_dev:
            return
        raise RuntimeError("Hosted modeling requires Landlock ABI 3 or later")
    read = (1 << 0) | (1 << 2) | (1 << 3)
    handled = (1 << 15) - 1

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = Ruleset(handled)
    ruleset = libc.syscall(create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        readonly = [
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path(__file__).resolve().parents[2],
            Path("/usr/lib"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc/ssl"),
            Path("/etc/resolv.conf"),
            Path("/etc/hosts"),
            Path("/etc/nsswitch.conf"),
            Path("/etc/localtime"),
            Path("/dev/urandom"),
            Path("/dev/null"),
        ]
        for path, access in [(p, read) for p in readonly] + [(workspace, handled)]:
            if not path.exists():
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = access if path.is_dir() else access & ~(1 << 3)
                rule = PathBeneath(allowed, fd)
                if libc.syscall(add, ruleset, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(restrict, ruleset, 0):
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset)
