from mcp_scan.models import ServerSignature, ScanPathResult, ServerScanResult, StdioServer
from mcp_scan.verify_api import analyze_scan_path
import tqdm.asyncio
import asyncio


data: list[ServerSignature] = []

with open("data/common_servers.jsonl", "r") as f:
    for line in f:
        data.append(ServerSignature.model_validate_json(line))

scan_paths: list[ScanPathResult] = [
    ScanPathResult(
        path="data/common_servers.jsonl",
        servers=[
            ServerScanResult(
                name=server.metadata.serverInfo.name,
                server=StdioServer(command="unknown"),
                signature=server,
                error=None,
            )
        ],
        issues=[],
        error=None,
    )
    for server in data
]
tasks = [analyze_scan_path(scan_path, base_url="https://mcp.invariantlabs.ai/") for scan_path in scan_paths]
async def run_tasks():
    results = await tqdm.asyncio.tqdm.gather(*tasks)
    return results
results = asyncio.run(run_tasks())

with open("data/common_servers_with_scan.jsonl", "w") as f:
    for result in results:
        f.write(result.model_dump_json() + "\n")
