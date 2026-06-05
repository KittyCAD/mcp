"""Launch an MCP server command, feed EOF on stdin, and assert it starts up.

Shared by the release binary smoke check and the install-smoke matrix. The
server is a blocking stdio loop with no ``--help``, so we close stdin (EOF) and
bound it with a timeout, then assert the startup banner reached stderr.

Usage:
    python scripts/smoke_launch.py ./dist/zoo-mcp-linux-x86_64
    python scripts/smoke_launch.py python -m zoo_mcp
"""

import os
import subprocess
import sys

cmd = sys.argv[1:]
if not cmd:
    sys.exit("smoke_launch: expected a command to run")

# Run "python ..." with the interpreter executing this script (e.g. the venv's
# Python) rather than whatever "python" resolves to first on PATH.
if cmd[0] == "python":
    cmd[0] = sys.executable

# A token is required to construct the KittyCAD client at import time; use a
# dummy so the smoke check needs no secret (and runs on fork PRs).
env = {
    **os.environ,
    "ZOO_API_TOKEN": os.environ.get("ZOO_API_TOKEN", "dummy-smoke-token"),
}

try:
    proc = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=60, env=env
    )
    err = proc.stderr.decode(errors="replace")
except subprocess.TimeoutExpired as exc:
    err = (exc.stderr or b"").decode(errors="replace")

sys.stderr.write(err)
sys.exit(0 if "Starting MCP server" in err else 1)
