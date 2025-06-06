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

from utils import load_json_file, get_results_path


def main(args, workflow=None, query="No query provided"):

    results_output_file = get_results_path(args)    
    model_client = BaseClient.from_cfg({
            "client": args.client,      # Options: "openai", "ollama", "vllm"
            "model": args.model,    # Model name, e.g., "gpt-4o-mini" or "llama3.3:70b" for ollama
            "temperature": args.temp,        # Default temperature setting
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
    agent_run = AgentRun(platform, results_output_file=results_output_file)
    
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
    
    # parser = argparse.ArgumentParser(description='geo-olm agent')
    # parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    # parser.add_argument('--exp_id', default=0, help='run ID to choose')
    # parser.add_argument('--client', default='openai', help='client to use')
    # parser.add_argument('--model', default= "gpt-4o-mini", help='model LLM to use')
    # parser.add_argument('--temp', default= 0.1, help='model LLM to use')
    # parser.add_argument('--agent', default= 'geoflow', help='agent to use')
    # args = parser.parse_args()

    # geo_path = f'./prompt_tests/benchmark/geo_{args.exp_id}'
    # geo_flow = load_json_file(geo_path + '/flow_gt.json')['tasks']
    # with open(geo_path + f"/query.txt", "r") as f:
    #     query = f.read()
    # main(args, geo_flow, query)
    from collections import OrderedDict

    # Load the original JSON file
    for exp_id in range(0, 22):
        geo_path = f'./prompt_tests/benchmark/geo_{exp_id}'
        with open(geo_path + f"/flow_expr.json", "r") as f:
            data = json.load(f)
        
        with open(geo_path + f"/query.txt", "r") as f:
            query = f.read()

        # Create a new OrderedDict with 'query' first
        new_data = OrderedDict()
        new_data["query"] = query
        new_data["tasks"] = data.get("tasks", {})

        # Save the modified JSON
        with open(geo_path + "/flow_expr.json", "w") as f:
            json.dump(new_data, f, indent=4)