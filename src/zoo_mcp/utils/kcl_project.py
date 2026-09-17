"""Capture the files referenced by a KCL entrypoint, preserving import paths."""

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

from zoo_mcp import ZooMCPException

# KCL imports have literal paths. Tokenize comments and strings as a unit so
# examples in comments or string values cannot introduce filesystem reads.
_TOKEN = re.compile(
    r"//[^\r\n]*|/\*.*?(?:\*/|\Z)|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\w+|\S",
    re.DOTALL,
)


def _imports(code: str) -> Iterator[tuple[int, int, str]]:
    tokens = [
        token
        for token in _TOKEN.finditer(code)
        if not token.group().startswith(("//", "/*"))
    ]
    for index, token in enumerate(tokens):
        if token.group() != "import":
            continue
        index += 1
        # import "file"; import name, other as alias from "file";
        # import * from "file". The KCL compiler remains the syntax validator.
        while index < len(tokens):
            value = tokens[index].group()
            if value.startswith(('"', "'")):
                path = value[1:-1]  # KCL preserves escapes in string literals.
                if not path.startswith("std::"):
                    yield tokens[index].start(), tokens[index].end(), path
                break
            if not (value.isidentifier() or value in (",", "*")):
                break
            index += 1


def _absolute(path: Path) -> Path:
    # Normalize dot components without resolving symlinks: imports are relative
    # to the caller's logical module location, not a linked file's real parent.
    return Path(os.path.abspath(path))


def _dependency(parent: Path, path: str) -> Path:
    # KCL accepts both Windows and Unix separators on every platform.
    return _absolute(parent / path.replace("\\", "/"))


def load_kcl_project(path: Path | str) -> tuple[str, list[dict[str, str | list[int]]]]:
    """Read the entrypoint, transitive imports, glTF buffers, and project config.

    Reading linked files through their logical paths materializes their bytes
    under the same module names in the captured project. No directory scan is
    needed, including for a directory argument (whose entrypoint is main.kcl).
    """
    path = _absolute(Path(path))
    entry = path / "main.kcl" if path.is_dir() else path
    contents: dict[Path, bytes] = {}
    # Avoid reading the same file twice through different symlink aliases.
    source_bytes: dict[Path, bytes] = {}
    active: set[Path] = set()

    def read(file: Path) -> None:
        physical = file.resolve()
        if physical in active:
            raise ZooMCPException(f"Circular KCL import involving '{file}'")
        if file in contents:
            return
        if physical not in source_bytes:
            source_bytes[physical] = file.read_bytes()
        data = source_bytes[physical]
        contents[file] = data
        active.add(physical)
        try:
            if file.suffix == ".kcl":
                code = data.decode("utf-8")
                replacements = []
                for start, end, imported in _imports(code):
                    import_path = Path(imported.replace("\\", "/"))
                    if imported.endswith(".kcl") and (
                        imported.startswith("..") or import_path.is_absolute()
                    ):
                        continue  # Invalid KCL imports belong to the compiler.
                    dependency = _dependency(file.parent, imported)
                    # Leave missing inputs to the compiler for a source-located
                    # diagnostic, rather than replacing it with a filesystem error.
                    if dependency.is_file():
                        read(dependency)
                    if import_path.is_absolute():
                        relative = Path(os.path.relpath(dependency, file.parent))
                        replacements.append((start, end, relative.as_posix()))
                # Absolute CAD imports must also resolve inside the capture.
                # Relative paths and all other source text remain byte-for-byte.
                for start, end, relative in reversed(replacements):
                    quote = code[start]
                    code = code[:start] + quote + relative + quote + code[end:]
                contents[file] = code.encode("utf-8")
            elif file.suffix.lower() in (".gltf", ".glb") and not data.startswith(
                b"glTF"
            ):
                try:
                    gltf = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    return  # Let KCL report malformed glTF.
                if isinstance(gltf, dict):
                    rewritten = False
                    for buffer in gltf.get("buffers", []):
                        uri = buffer.get("uri") if isinstance(buffer, dict) else None
                        if isinstance(uri, str) and not uri.startswith("data:"):
                            dependency = _dependency(file.parent, uri)
                            if dependency.is_file():
                                read(dependency)
                            if Path(uri.replace("\\", "/")).is_absolute():
                                buffer["uri"] = Path(
                                    os.path.relpath(dependency, file.parent)
                                ).as_posix()
                                rewritten = True
                    if rewritten:
                        contents[file] = json.dumps(gltf).encode("utf-8")
        finally:
            active.remove(physical)

    read(entry)
    config = entry.parent / "project.toml"
    if config.is_file():
        read(config)
    # Relative CAD imports and glTF buffers may live above the entrypoint's
    # directory. Keep their layout without walking or copying that ancestor.
    root = Path(os.path.commonpath([str(file.parent) for file in contents]))
    return entry.relative_to(root).as_posix(), [
        {"path": file.relative_to(root).as_posix(), "contents": list(data)}
        for file, data in sorted(contents.items())
    ]
