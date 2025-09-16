from mcp_scan.models import ServerSignature, ScanPathResult, ServerScanResult, StdioServer
from mcp_scan.verify_api import analyze_scan_path
import tqdm.asyncio
from itertools import combinations
import asyncio
from pydantic import BaseModel, Field

class ScalarToolLabels(BaseModel):
    is_public_sink: int | float
    destructive: int | float
    untrusted_content: int | float
    private_data: int | float

class ScanPathResultWithLabels(ScanPathResult):
    labels: list[list[ScalarToolLabels]] = Field(default_factory=list)

def load_data(source_file: str = "data/common_servers.jsonl", target_file: str = "data/common_servers_with_scan.jsonl"):
    data: list[ServerSignature] = []

    with open(source_file, "r") as f:
        for line in f:
            data.append(ServerSignature.model_validate_json(line))

    scan_paths: list[ScanPathResult] = [
        ScanPathResult(
            path=source_file,
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
    async def run_tasks() -> list[ScanPathResult]:
        results = await tqdm.asyncio.tqdm.gather(*tasks)
        return results
    results = asyncio.run(run_tasks())

    with open(target_file, "w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def get_scans(
    source_file: str = "data/common_servers_with_scan.jsonl",
    n_servers_per_scan: int = 1
) -> list[ScanPathResult]:
    sprs: list[ScanPathResult] = []
    with open(source_file, "r") as f:
        for line in f:
            sprs.append(ScanPathResult.model_validate_json(line))

    for spr in sprs:
        assert len(spr.servers) == 1, "Expected exactly one server per scan"

    scans: list[ScanPathResult] = []
    for spr_tuple in combinations(sprs, n_servers_per_scan):
        scans.append(ScanPathResult(
            path=source_file,
            servers=[spr.servers[0] for spr in spr_tuple],
            issues=sum([spr.issues for spr in spr_tuple], []),
            labels=sum([spr.labels for spr in spr_tuple], []),
            error=None
        ))
    return scans


if __name__ == "__main__":
    load_data()
