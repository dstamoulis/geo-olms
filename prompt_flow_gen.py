import json
import argparse
import sys
import os
import time
from collections import OrderedDict

from llm_clients.base_client import BaseClient
from utils import load_json_file, strip_json_code_block, re_args_component
from prompt_flows.gen_prompt import INIT_WORKFLOW_PROMPT, INIT_WORKFLOW_TEMPLATE, AGENT_LIST

import re

def extract_json_object(raw: str) -> str:
    """
    Extracts the first JSON object from `raw` text, ignoring any <think>...</think> sections.
    1) Strip out any <think>…</think> blocks entirely.
    2) If there's a ```json...``` code block, return its contents.
    3) Otherwise, find the first “{” and grab the balanced {...} substring.
    Returns the JSON string or raises ValueError if none found.
    """
    # 0) Remove any <think>...</think> sections
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Trim whitespace
    raw = raw.strip()

    # 1) Try to pull from a ```json …``` block
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return m.group(1)

    # 2) Otherwise look for the first “{” and match braces
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("Unbalanced braces in response")


def generate_workflow(query, args):

    model_client = BaseClient.from_cfg({
        "client": args.client,      # Options: "openai", "ollama", "vllm"
        "model": args.model,    # Model name, e.g., "gpt-4o-mini" or "llama3.3:70b" for ollama
        "temperature": args.temp,        # Default temperature setting
    })

    messages=[
                {
                    "role": "user",
                    "content": 
                    f'''
                    {INIT_WORKFLOW_PROMPT}\n

                    Available Agents:
                    {AGENT_LIST}

                    **INSTRUCTIONS:
                    ==> Provide the output in the same format as the example above. 
                    ==> Make sure you choose only the agents provided in the **Available Agents** to complete the task.

                    **IMPORTANT**

                    1. You must output the JSON object following the template above.
                    2. Do **NOT** include any markdown backticks, explanations, or extra text!! ONLY the raw JSON.
                    3. Your response must begin with \"{{\" and end with \"}}"" and parse as valid JSON.

                    Here is the task to be executed:

                    '{query}'
                    '''
            
                }
            ]

    response = model_client.get_response(messages)
    return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    parser.add_argument('--exp_id', default=0, help='run ID to choose')
    parser.add_argument('--client', default='openai', help='client to use')
    parser.add_argument('--model', default= "gpt-4o-mini", help='model LLM to use')
    parser.add_argument('--temp', default= 0.1, help='model LLM to use')
    parser.add_argument('--flow_ver', default=None, help='agent to use')
    args = parser.parse_args()

    # The query to generate flow json for!
    query = load_json_file(f'./prompt_flows/flows/flow_gt/{args.exp_id}.json')['query']
    # The output file (workflow json)
    flow_ver = f"{re_args_component(args.model)}" if args.flow_ver is None else re_args_component(args.flow_ver)

    base_dir = './prompt_flows/flows/'
    results_path = os.path.join(base_dir, flow_ver)
    geo_flow_dest = os.path.join(results_path, f"{args.exp_id}.json")
    os.makedirs(results_path, exist_ok=True)

    max_retries = 3
    attempts = 0

    while attempts < max_retries:
        attempts += 1
        try:
            start_time = time.time()
            response = generate_workflow(query, args)
            elapsed_time = round(time.time() - start_time, 4)

            # Save the generated JSON
            print(response.content)
            cleaned_json_str = extract_json_object(response.content)
            parsed_json = json.loads(cleaned_json_str)

            generated_workflow = OrderedDict()
            generated_workflow["query"] = query  # Replace with actual query
            generated_workflow["tasks"] = parsed_json.get("tasks", {})
            generated_workflow["model_client"] = {        
                "client_class": str(args.client),
                "model": str(args.model),
                "temperature": str(args.temp),
                "prompt_tokens": response.prompt_tokens,
                "cached_tokens": response.cached_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "time_elapsed": elapsed_time,
            }
            with open(geo_flow_dest, "w") as f:
                json.dump(generated_workflow, f, indent=2)
            # if we got here, it succeeded
            print(f"[Attempt {attempts}/{max_retries}] worked for exp_id {args.exp_id}")
            break

        except Exception as e:
            print(f"[Attempt {attempts}/{max_retries}] Failed to produce flow JSON for exp_id {args.exp_id}: {e}")
            if attempts < max_retries:
                print(f"Trying again!")