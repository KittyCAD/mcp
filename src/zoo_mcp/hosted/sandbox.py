"""Fail-closed Linux Landlock policy, installed before importing CAD libraries.

Runtime directories must contain only application code, never mounted secrets or
workspaces. No environment-controlled read allowances are accepted.
"""

import ctypes
import os
import platform
import sys
import sysconfig
from pathlib import Path


def confine(workspace: Path) -> None:
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Hosted workers require Linux x86_64/aarch64 and Landlock")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError("Hosted workers require Landlock ABI 3 or later")

    class Ruleset(ctypes.Structure):
        _fields_ = [
            ("handled_access_fs", ctypes.c_uint64),
            ("handled_access_net", ctypes.c_uint64),
            ("scoped", ctypes.c_uint64),
        ]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    # Handle all supported filesystem operations. Never grant device creation,
    # execution, or device ioctls in the writable workspace.
    handled = (1 << (17 if abi >= 9 else 16 if abi >= 5 else 15)) - 1
    read = (1 << 0) | (1 << 2) | (1 << 3)
    writable = handled & ~((1 << 0) | (1 << 6) | (1 << 11) | (1 << 15))
    attr = Ruleset(handled, 0, 3 if abi >= 6 else 0)
    size = ctypes.sizeof(attr) if abi >= 6 else 8
    ruleset = libc.syscall(444, ctypes.byref(attr), size, 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    readonly = {
        Path(sysconfig.get_path("stdlib")),
        Path(sysconfig.get_path("platstdlib")),
        Path(sysconfig.get_path("purelib")),
        Path(sysconfig.get_path("platlib")),
        Path(__file__).resolve().parents[1],
        Path(sys.executable).resolve(),
        Path("/usr/lib"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ssl/certs"),
        Path("/etc/resolv.conf"),
        Path("/etc/hosts"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/localtime"),
        Path("/dev/urandom"),
        Path("/dev/null"),
    }
    try:
        workspace = workspace.resolve(strict=True)
        for path in readonly:
            if workspace.is_relative_to(path.resolve()):
                raise RuntimeError("Workspaces must be outside runtime read roots")
        for path, access in [(p, read) for p in readonly] + [(workspace, writable)]:
            if not path.exists():
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneath(access if path.is_dir() else access & ~8, fd)
                if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, ruleset, 0):
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset)
