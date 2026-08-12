import sys

from zoo_mcp import logger
from zoo_mcp.server import main

if __name__ == "__main__":
    try:
        # Delegated rather than calling mcp.run directly so this entry point
        # gets the same shutdown handling as the zoo-mcp console script.
        main()

    except KeyboardInterrupt:
        logger.info("Shutting down MCP server...")
    except Exception as e:
        logger.exception("Server encountered an error: %s", e)
        sys.exit(1)
