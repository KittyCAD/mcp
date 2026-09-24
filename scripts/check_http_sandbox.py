"""Run in the native Linux service image: assert the worker filesystem boundary."""

from pathlib import Path

from zoo_mcp.utils.sandbox import confine

owned = Path("/tmp/worker-one")
other = Path("/tmp/worker-two")
owned.mkdir()
other.mkdir()
(other / "source.kcl").write_text("private")
confine(owned)
(owned / "output.kcl").write_text("allowed")
assert (owned / "output.kcl").read_text() == "allowed"
for path in (other / "source.kcl", Path("/etc/passwd")):
    try:
        path.read_text()
    except PermissionError:
        continue
    raise AssertionError("Worker read outside its filesystem boundary")
print("Worker filesystem boundary enforced")
