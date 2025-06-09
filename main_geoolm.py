# main_agent.py
# Flow++

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

from utils import load_json_file, get_results_path, re_args_component
    

def main(args, workflow=None, query="No query provided"):
 
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
        system_message="You are the database agent!"
    )
    map_agent = SingleAgent(
        api=args.api,
        name="map_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[map_tools],
        system_message="You are the map agent!"
    )
    detector_agent = SingleAgent(
        api=args.api,
        name="detector_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[vision],
        system_message="You are the detector agent!"
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
        system_message="You are an orchastrating agent handing off tasks to subagents to complete given task!"
    )

    # Solving with an agent!
    platform = Platform(model_client, messages, database, vision, map_tools, orch_agent)
    agent_run = AgentRun(platform, results_output_filenames=get_results_path(args))

    start_time = time.time()
    response = platform.agent.run_flowPP(
        {"database_agent": database_agent, "map_agent": map_agent, "detector_agent": detector_agent, "data_agent": data_agent},
        workflow
    )
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)
    print("===elapsed time: ", elapsed_time, " s===")

    print(response)
    print(platform.database.images_gdf)
    print(platform.vision.detections_gdf)

    agent_run.save_agent_run_result(AgentTask(query= query, messages= platform.messages.to_list_dict()))
    platform.reset()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    parser.add_argument('--exp_id', default=0, help='run ID to choose')
    parser.add_argument('--client', default='openai', help='client to use')
    parser.add_argument('--model', default= "gpt-4o-mini", help='model LLM to use')
    parser.add_argument('--temp', default= 0.1, help='model LLM to use')
    parser.add_argument('--agent', default= 'geoolm', help='agent to use')
    parser.add_argument('--flow_ver', default= 'flow_gt', help='agent to use')
    args = parser.parse_args()

    geo_flow = load_json_file(f'./prompt_flows/flows/{re_args_component(args.flow_ver)}/{args.exp_id}.json')
    main(args, geo_flow['tasks'], geo_flow["query"])