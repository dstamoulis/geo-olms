# main_agent.py
# GeoFlow

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

import argparse

import json
import time


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


def add_query_to_flowJSON(num_files):

    from collections import OrderedDict
    for i in range(0, num_files):
        # Load the JSON file
        geo_path = f'./prompt_tests/benchmark/geo_{i}'
        with open(geo_path + f"/flow.json", "r") as f:
            data = json.load(f)

        with open(geo_path + f"/query.txt", "r") as f:
            query = f.read()

        # Add or update the 'query' field
        new_data = OrderedDict()
        new_data["query"] = query  # Replace with your actual query value
        new_data["tasks"] = data.get("tasks", {})

        # Save it back to the same file (or a new one)
        with open(geo_path + "/flow_gt.json", "w") as f:
            json.dump(new_data, f, indent=4)
    

def main(args, workflow=None, query="No query provided"):
    
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
        system_message="Expert in fetching images from a database!"
    )
    map_agent = SingleAgent(
        api=args.api,
        name="map_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[map_tools],
        system_message="Expert in performing all kinds of operations on a map!"
    )
    detector_agent = SingleAgent(
        api=args.api,
        name="detector_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[vision],
        system_message="Expert in processing images fetched from a database, such as object detection!"
    )
    data_agent = SingleAgent(
        api=args.api,
        name="data_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[data_tools],
        system_message="Expert in all kinds of image analyzing tasks!"
    )
    orch_agent = SingleAgent(
        api=args.api,
        name="orch_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[],
        system_message="You are an orchastrating agent handing off tasks to subagents most suited for given a given task!"
    )
    
    single_agent = SingleAgent(
        api=args.api,
        name="single_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[database, vision, map_tools, data_tools],
        system_message="You are a geospatial agent helping with fetching images from a database!"
    )

    # Solving with an agent!
    platform = Platform(model_client, messages, database, vision, map_tools, orch_agent)
    # platform = Platform(model_client, messages, database, vision, map_tools, single_agent)
    agent_run = AgentRun(platform, results_output_file='./results/single_agent_test.json')
    
    start_time = time.time()
    # response = platform.agent.run_query(query)
    response = platform.agent.run_workflow(
        {"database_agent": database_agent, "map_agent": map_agent, "detector_agent": detector_agent, "data_agent": data_agent},
        workflow
    )
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)
    print("===elapsed time: ", elapsed_time, " s===")

    print(response)
    print(platform.database.images_gdf)
    print(platform.vision.detections_gdf)

    agent_run.add_task_result(AgentTask(queries=[query], rounds=[{"query": query, "messages": platform.messages.to_list_dict()}]))
    agent_run.save_inference_results()
    platform.reset()

if __name__ == "__main__":
    i = 0
    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    args = parser.parse_args()

    geo_path = f'./prompt_tests/benchmark/geo_{i}'
    geo_flow = load_json_file(geo_path + '/flow.json')['tasks']
    with open(geo_path + f"/query.txt", "r") as f:
        query = f.read()
    main(args, geo_flow, query)