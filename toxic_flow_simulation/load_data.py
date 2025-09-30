import sys
from mcp_scan.models import ServerSignature, ScanPathResult, ServerScanResult, StdioServer
from mcp_scan.verify_api import analyze_scan_path
import tqdm.asyncio
from itertools import combinations
import asyncio
from pydantic import BaseModel, Field
import aiohttp
import json

class ScalarToolLabels(BaseModel):
    is_public_sink: int | float
    destructive: int | float
    untrusted_content: int | float
    private_data: int | float

class ScanPathResultWithLabels(ServerSignature):
    labels: list[list[ScalarToolLabels]] = Field(default_factory=list)


async def call_labels(data: list[ScanPathResult]):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:9099/hidden/mcp-scan/labels",
            json=[scan_path.model_dump() for scan_path in data]) as response:
            return await response.json()


def load_data(source_file: str = "data/common_servers.jsonl", target_file: str = "data/common_servers_with_scan.jsonl"):
    data: list[ServerSignature] = []

    with open(source_file, "r") as f:
        for line in f:
            data.append(ServerSignature.model_validate_json(line))
    labels = asyncio.run(call_labels(data))
    assert len(labels) == len(data)
    scan_paths: list[ScanPathResultWithReasons] = [
        ScanPathResultWithReasons(
            path=source_file,
            servers=[
                ServerScanResult(
                    name=server.metadata.serverInfo.name,
                    server=StdioServer(command="unknown"),
                    signature=server,
                    error=None,
                )
            ],
            error=None,
            labels=[[ScalarToolLabelsWithReason.from_scalar_tool_thought_process_labels(ll) for ll in label]]
        )
        for server, label in zip(data, labels)
    ]
    with open(target_file, "w") as f:
        for result in scan_paths:
            f.write(result.model_dump_json() + "\n")

class ScalarToolLabelsWithReason(ScalarToolLabels):
    is_public_sink_reason: str
    destructive_reason: str
    untrusted_content_reason: str
    private_data_reason: str

    @classmethod
    def from_scalar_tool_thought_process_labels(cls, label: dict):
        return ScalarToolLabelsWithReason(
            is_public_sink=label["is_public_sink"]["value"],
            destructive=label["destructive"]["value"],
            untrusted_content=label["untrusted_content"]["value"],
            private_data=label["private_data"]["value"],
            is_public_sink_reason=label["is_public_sink"]["thought_process"],
            destructive_reason=label["destructive"]["thought_process"],
            untrusted_content_reason=label["untrusted_content"]["thought_process"],
            private_data_reason=label["private_data"]["thought_process"],
        )


class ScanPathResultWithReasons(ScanPathResult):
    labels: list[list[ScalarToolLabelsWithReason]] = Field(default_factory=list)

def get_scans(
    source_file: str = "data/common_servers_with_scan.jsonl",
    n_servers_per_scan: int = 1
) -> list[ScanPathResult]:
    sprs: list[ScanPathResultWithReasons] = []
    with open(source_file, "r") as f:
        for line in f:
            sprs.append(ScanPathResultWithReasons.model_validate_json(line))

    for spr in sprs:
        assert len(spr.servers) == 1, "Expected exactly one server per scan"

    scans: list[ScanPathResultWithReasons] = []
    for spr_tuple in combinations(sprs, n_servers_per_scan):
        scans.append(ScanPathResultWithReasons(
            path=source_file,
            servers=[spr.servers[0] for spr in spr_tuple],
            issues=sum([spr.issues for spr in spr_tuple], []),
            labels=sum([spr.labels for spr in spr_tuple], []),
            error=None
        ))
    return scans


if __name__ == "__main__":
    if len(sys.argv) == 3:
        source_file = sys.argv[1]
        target_file = sys.argv[2]
        load_data(source_file, target_file)
    elif len(sys.argv) == 1:
        # No arguments provided, use defaults
        load_data()
    else:
        print("Usage: python load_data.py [source_file target_file]")
        print("  source_file: Input JSONL file to read from")
        print("  target_file: Output JSONL file to write to")
        print("  If no arguments provided, defaults will be used")
        sys.exit(1)