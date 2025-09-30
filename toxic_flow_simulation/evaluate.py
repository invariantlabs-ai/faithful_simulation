import httpx 
import json
import os
import asyncio
from toxic_flow_simulation.load_data import get_scans
from pydantic import validate_email
from typing import Any
from afma.simulation.environment import McpEnvironment
scan_result = get_scans(n_servers_per_scan=1)[2]
mcp_environment = McpEnvironment(scan_result=scan_result)
async def call_together(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    logprobs: bool,
    echo: bool,
    url: str = "https://api.together.xyz/v1/chat/completions",
    extra_headers: dict[str, str] | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    print(tools)
    data = json.dumps({
        'messages': messages,
        'model': model,
        'max_tokens': max_tokens,
        'stream': False,
        'logprobs': logprobs,
        'echo': echo,
        'tools': tools,
    })
    if api_key is None:
        api_key = os.getenv("TOGETHER_API_KEY", None)
        if api_key is None:
            raise ValueError("TOGETHER_API_KEY is not set")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    if extra_headers is not None:
        headers.update(extra_headers)
    response = await httpx.AsyncClient().post(url, headers=headers, data=data.encode())
    if response.status_code != 200:
        raise ValueError(f"Error calling Together API: {response.text}")
    return response.json()


async def evaluate(
    message_history: list[dict[str, str]],
    output_message: dict[str, str],
    environment: McpEnvironment,
    model: str = "mistralai/Mistral-Small-24B-Instruct-2501",
    only_function_name: bool = False,
) -> float:
    response = await call_together(
        model=model,
        messages=message_history + [output_message],
        logprobs=True,  # This enables log probability return
        echo=True,
        max_tokens=1,
        tools=await environment.collect_resources(),
    )
    response2 = await call_together(
        model=model,
        messages=message_history,
        logprobs=True,  # This enables log probability return
        echo=True,
        max_tokens=200,
        tools=await environment.collect_resources(),
    )
    print("response2")
    print(response2["choices"][0])
    start_id = "<|start_header_id|>"
    start_id_indexes = [i for i, token in enumerate(response["prompt"][0]["logprobs"]["tokens"]) if token == start_id]
    ans_log_probs = response["prompt"][0]["logprobs"]["token_logprobs"][start_id_indexes[-2]: start_id_indexes[-1]]
    tokens = response["prompt"][0]["logprobs"]["tokens"][start_id_indexes[-2]: start_id_indexes[-1]]
    if only_function_name:
        parameter_delimeter = tokens.index("parameters")
        tokens = tokens[:parameter_delimeter]
        ans_log_probs = ans_log_probs[:parameter_delimeter]
    #print("".join(tokens))
    #print(tokens)
    #print(ans_log_probs)
    return sum(ans_log_probs)
    

models = [
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
]
trace = json.load(open("traces/trace_20250922_153350.json"))
message_history = trace[:4]
output_message = trace[4]

valid_models = []
for model in models:
    try:
        result = asyncio.run(evaluate(message_history, output_message, mcp_environment, model, only_function_name=True))
        print("OK", "Model:", model, "Result:", result)
        valid_models.append(model)
    except Exception as e:
        print("ERROR", "Model:", model, "Error:", e)
print("Valid models:", valid_models)