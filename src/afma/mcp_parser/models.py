from mcp_scan.models import ScanPathResult
from mcp_scan.MCPScanner import MCPScanner


async def scan_mcp_file(
    path: str,
    base_url: str = "https://mcp.invariantlabs.ai/",
    storage_file: str = "~/.mcp-scan",
    suppress_mcpserver_io: bool = True,
    include_built_in: bool = True,
    server_timeout: int = 10,
) -> ScanPathResult:
    scanner = MCPScanner(
        files=[path],
        base_url=base_url,
        storage_file=storage_file,
        suppress_mcpserver_io=suppress_mcpserver_io,
        include_built_in=include_built_in,
        server_timeout=server_timeout,
    )
    await scanner.scan()