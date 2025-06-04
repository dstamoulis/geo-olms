# main_stateflow.py

from agent_core.modules.messages import Messages, TextMessage
from agent_core.agents.assistant_agent import AssistantAgent
from agent_core.teams.single_agent import SingleAgent

from geoapps.geeo.database import Database
from geoapps.geeo.map_tools import MapTools
from geoapps.geeo.data_tools import DataTools
from geoapps.geeo.vision import Vision

from llm_clients.base_client import BaseClient
from evaluate.agent_run import AgentRun
from evaluate.agent_task import AgentTask

from geoplatform.platform import Platform
from collections import deque

import argparse

import json
import time


# ------------------------------
# Added
# ------------------------------
def load_json_file(file_path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The loaded JSON data as a Python dictionary or list, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {file_path}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return None
    

def main(args, workflow=None):
    
    model_client = BaseClient.from_cfg({
            "client": "openai",      # Options: "openai", "ollama", "vllm"
            "model": "gpt-4o-mini",    # Model name, e.g., "gpt-4o-mini" or "llama3.3:70b" for ollama
            "temperature": 0.1,        # Default temperature setting
        })
    messages = Messages()
    database = Database()    
    vision = Vision(database)    
    map_tools = MapTools(database, vision, map_style="open-street-map")
    data_tools = DataTools(database, vision)

    # Subagents
    database_agent = SingleAgent(
        api=args.api,
        name="database_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[database],
        system_message="You are the database agent!"
    )
    detector_agent = SingleAgent(
        api=args.api,
        name="detector_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[vision],
        system_message="You are the detector agent!"
    )
    map_agent = SingleAgent(
        api=args.api,
        name="map_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[map_tools],
        system_message="You are the map agent!"
    )
    verifier_agent = SingleAgent(
        api=args.api,
        name="verifier_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[],
        system_message="You are a result verifier agent that verify if the query has been executed successfully." \
        "You will receive an input of fomat \"[Objective]: xxx. [Response]: yyy\", and you will decide if the Response" \
        "successfully fulfill the ask from Objective. If it does, return \"COPMLETED\". If not, return \"ERROR\"." \
        "Example:\n"
        "[Objective]: Fetch images from the FAIR1M dataset and filter the images from the UK." \
        "[Response]: I have successfully fetched 14 images from the FAIR1M dataset for the UK.\n" \
        "In the above case, you will return \"COMPLETED\"." \
    )

    # Solving with an agent!
    # process tasks for StateFlow
    task_queue = deque()
    for task in workflow.values():
        task_queue.append(task)
    state_queue = deque([database_agent, detector_agent, map_agent])

    platform = Platform(model_client, messages, database, vision, map_tools, verifier_agent)
    start_time = time.time()
    response = platform.agent.run_stateflow(
        task_queue=task_queue,
        state_queue=state_queue,
    )
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)
    print("===\nelapsed time: ", elapsed_time, " s\n===")

    print(response)
    print(platform.database.images_gdf)
    print(platform.vision.detections_gdf)

    # agent_run.add_task_result(AgentTask(queries=[query], rounds=[{"query": query, "messages": platform.messages.to_list_dict()}]))
    # agent_run.save_inference_results()
    # platform.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    args = parser.parse_args()
    geo_flow = load_json_file('./workflows/geo_1/init_2.json')['tasks']
    main(args, geo_flow)