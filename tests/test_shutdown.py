"""Shutdown handling for open modeling sessions.

A leaked session pins an engine instance until the backend reaps it, so the
server has to close them on the way out no matter how it is stopped.
"""

import signal
import subprocess
import sys
import textwrap

import pytest

from zoo_mcp import server

# Registers a session whose close writes a marker, then blocks until signalled.
# Run as a subprocess so a real SIGTERM exercises the real handler.
_SERVER_UNDER_SIGNAL = """
import sys, time
from pathlib import Path
from unittest.mock import MagicMock

from zoo_mcp import server, zoo_tools

marker = Path(sys.argv[1])

session = zoo_tools._ModelingSession(
    context=MagicMock(),
    websocket=MagicMock(),
    lock=zoo_tools.Lock(),
    last_used=zoo_tools.monotonic(),
)
session.context.__exit__ = lambda *args: marker.write_text("closed")
zoo_tools._modeling_sessions["session-id"] = session

server.install_shutdown_handlers()

print("READY", flush=True)
time.sleep(30)
"""


def _run_until_signalled(tmp_path, signal_number: int) -> bool:
    """Start the stub server, signal it, and report whether it closed up."""
    marker = tmp_path / "closed.txt"
    script = tmp_path / "stub_server.py"
    script.write_text(textwrap.dedent(_SERVER_UNDER_SIGNAL))

    process = subprocess.Popen(
        [sys.executable, str(script), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal_number)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:  # pragma: no cover - only on a hang
            process.kill()
            process.wait(timeout=10)

    return marker.exists()


@pytest.mark.parametrize(
    "signal_number",
    [signal.SIGTERM, signal.SIGINT],
    ids=["sigterm", "sigint"],
)
def test_modeling_sessions_are_closed_on_termination(tmp_path, signal_number: int):
    """SIGTERM used to kill the process outright, leaking every open session."""
    assert _run_until_signalled(tmp_path, signal_number)


def test_install_shutdown_handlers_routes_sigterm_through_keyboard_interrupt():
    original = signal.getsignal(signal.SIGTERM)
    try:
        server.install_shutdown_handlers()
        assert signal.getsignal(signal.SIGTERM) is server._shutdown_on_signal
    finally:
        signal.signal(signal.SIGTERM, original)

    with pytest.raises(KeyboardInterrupt):
        server._shutdown_on_signal(signal.SIGTERM, None)


def test_install_shutdown_handlers_tolerates_a_non_main_thread(monkeypatch):
    """Embedders own signal disposition; we should not crash on their behalf."""

    def refuse(*args, **kwargs):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", refuse)

    server.install_shutdown_handlers()  # must not raise
