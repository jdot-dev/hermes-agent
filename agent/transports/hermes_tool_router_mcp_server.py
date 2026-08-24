"""Small MCP facade over Hermes' large built-in + deferred MCP tool catalog.

Exposes only three tools to clients: tool_search, tool_describe, and tool_call.
The backing Hermes registry still owns MCP discovery, schemas, dispatch, lifecycle,
redaction, and server timeouts.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _build_server() -> Any:
    from mcp.server import MCPServer

    from model_tools import get_tool_definitions, handle_function_call
    from tools.mcp_tool import discover_mcp_tools

    # Register configured MCP servers. Lazy servers hydrate schemas from cache
    # and connect only when a routed call reaches them.
    discover_mcp_tools()

    server = MCPServer(
        "hermes-tool-router",
        instructions=(
            "Intelligent access to Hermes Agent's tool catalog. Search first, "
            "describe the selected tool to load its schema, then call it. "
            "The catalog includes configured deferred MCP servers and safe "
            "stateless Hermes tools without dumping every schema into context."
        ),
    )

    def _call(name: str, arguments: dict[str, Any]) -> str:
        try:
            # Recompute the public definitions so the bridge handlers receive
            # the current raw catalog through their normal Hermes dispatch path.
            get_tool_definitions(quiet_mode=True)
            return handle_function_call(name, arguments)
        except Exception as exc:
            logger.exception("router operation %s failed", name)
            return json.dumps({"error": str(exc), "operation": name})

    @server.tool(
        name="tool_search",
        description=(
            "Search Hermes' deferred tool catalog. Returns matching tool names "
            "and short descriptions. Use before tool_describe/tool_call."
        ),
    )
    def tool_search(*, query: str, limit: int = 5) -> str:
        return _call("tool_search", {"query": query, "limit": limit})

    @server.tool(
        name="tool_describe",
        description=(
            "Load the full JSON schema for one exact tool name returned by "
            "tool_search. Required before tool_call when arguments are unknown."
        ),
    )
    def tool_describe(*, name: str) -> str:
        return _call("tool_describe", {"name": name})

    @server.tool(
        name="tool_call",
        description=(
            "Invoke one deferred Hermes/MCP tool by exact name. Pass arguments "
            "matching the schema returned by tool_describe. Write-capable calls "
            "execute with the target tool's normal Hermes policy and credentials."
        ),
    )
    def tool_call(*, name: str, arguments: dict[str, Any]) -> str:
        return _call("tool_call", {"name": name, "arguments": arguments})

    return server


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    logging.basicConfig(
        level=logging.INFO if "--verbose" in argv or "-v" in argv else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")
    try:
        _build_server().run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("Hermes tool router crashed")
        sys.stderr.write(f"hermes-tool-router error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
