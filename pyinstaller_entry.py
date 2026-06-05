"""PyInstaller entry point for the standalone ``zoo-mcp`` binary.

Mirrors the ``zoo-mcp`` console script (``zoo_mcp.server:main``) so the
frozen executable behaves identically to ``uvx zoo-mcp`` / ``python -m zoo_mcp``.
"""

from zoo_mcp.server import main

if __name__ == "__main__":
    main()
