from openai import OpenAI
client = OpenAI()
import json
import argparse
import sys
import os
import time
from collections import OrderedDict

from utils import load_json_file, strip_json_code_block
from prompt_flows.gen_prompt import INIT_WORKFLOW_PROMPT, INIT_WORKFLOW_TEMPLATE, AGENT_LIST


def generate_workflow(query):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 
                f'''
                {INIT_WORKFLOW_PROMPT}\n

                For example:
                {INIT_WORKFLOW_TEMPLATE}\n

                Available Agents:
                {AGENT_LIST}

                Provide the output in the same format as the example above. Make sure you choose only the agents provided in the \"Available Agents\" to complete the task.

                Here is the task to be executed:

                '{query}'
                '''
        
            }
        ]
    )
    return completion


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    parser.add_argument('--exp_id', default=0, help='run ID to choose')
    parser.add_argument('--client', default='openai', help='client to use')
    parser.add_argument('--model', default= "gpt-4o-mini", help='model LLM to use')
    parser.add_argument('--temp', default= 0.1, help='model LLM to use')
    parser.add_argument('--agent', default= 'geoflow', help='agent to use')
    # parser.add_argument('--flow_ver', default= 'flow_gt', help='agent to use')
    args = parser.parse_args()

    # The query to generate flow json for!
    query = load_json_file(f'./prompt_flows/flows/flow_gt/{args.exp_id}.json')['query']
    # The output file (workflow json)
    flow_ver = f"flow_{args.model.replace(':', '_').replace('-', '_')}"

    base_dir = './prompt_flows/flows/'
    results_path = os.path.join(base_dir, flow_ver)
    geo_flow_dest = os.path.join(results_path, f"{args.exp_id}.json")
    os.makedirs(results_path, exist_ok=True)

    start_time = time.time()
    response = generate_workflow(query)
    elapsed_time = round(time.time() - start_time, 4)

    # Save the generated GeoFlow JSON
    response_msg = response.choices[0].message.content
    cleaned_json_str = strip_json_code_block(response_msg)
    parsed_json = json.loads(cleaned_json_str)

    generated_workflow = OrderedDict()
    generated_workflow["query"] = query  # Replace with actual query
    generated_workflow["tasks"] = parsed_json.get("tasks", {})
    generated_workflow["model_client"] = {        
        "client_class": str(args.client),
        "model": str(args.model),
        "temperature": 0.1,
        "prompt_tokens": response.usage.prompt_tokens,
        "cached_tokens": response.usage.prompt_tokens_details.cached_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "time_elapsed": elapsed_time,
    }
    with open(geo_flow_dest, "w") as f:
        json.dump(generated_workflow, f, indent=2)