from openai import OpenAI
client = OpenAI()
import prompt
import json
import argparse
import sys
import os
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import utils

def strip_json_code_block(raw_content):
    # Remove leading/trailing whitespace
    raw_content = raw_content.strip()
    
    # Remove triple backtick wrapper if present
    if raw_content.startswith("```json") or raw_content.startswith("```"):
        # Split by lines, remove the first and last lines
        lines = raw_content.splitlines()
        # Handle empty or malformed content defensively
        if len(lines) >= 3:
            return "\n".join(lines[1:-1])
    return raw_content

def generate_workflow(query):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        # Removed {prompt.TASK_EXECUTION_PROMPT}\n
        messages=[
            {
                "role": "user",
                "content": 
                f'''
                {prompt.INIT_WORKFLOW_PROMPT}\n

                For example:
                {prompt.INIT_WORKFLOW_TEMPLATE}\n

                Available Agents:
                {prompt.AGENT_LIST}

                Provide the output in the same format as the example above. Make sure you choose only the agents provided in the \"Available Agents\" to complete the task.

                Here is the task to be executed:

                '{query}'
                '''
        
            }
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='geo-olm agent')
    # parser.add_argument('--exp_id', default=0, help='run ID to choose')
    # args = parser.parse_args()
    # geo_path = f'./prompt_tests/benchmark/geo_{args.exp_id}'

    for i in range(22):
        geo_path = f'./prompt_tests/benchmark/geo_{i}'
        query = utils.load_json_file(geo_path + '/flow_gt.json')['query']

        response = generate_workflow(query)

        # Save the generated GeoFlow JSON
        cleaned_json_str = strip_json_code_block(response)
        parsed_json = json.loads(cleaned_json_str)

        new_data = OrderedDict()
        new_data["query"] = query  # Replace with actual query
        new_data["tasks"] = parsed_json.get("tasks", {})
        with open(f"prompt_tests/benchmark/geo_{i}/flow_exp_2.json", "w") as f:
            json.dump(new_data, f, indent=2)