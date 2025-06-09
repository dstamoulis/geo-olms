# main_stateflow.py
# StateFlow

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

from utils import load_json_file, get_results_path


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

    transition_message = "Based on your ability, please decide if you have the required tools to complete the task, which " \
        "will be in the format \'[Objective]: xxx\'.\nIf you DO NOT have the required tools, return \'PASS\'.\nIf you DO have the required tools for the task, " \
        "go on and try to complete the task/subtask.\n" \
        "At any point of completing the tasks/subtasks, if you encounter any error, return \'ERROR\'.\n Upon completing in your turn, ONLY REPLY WITH EITHER PASS or ERROR. Nothing else!!!" \
        
    # Subagents
    database_agent = SingleAgent(
        api=args.api,
        name="database_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[database],
        system_message=f"You are an expert in fetching images from a database!\n {transition_message}" \
        # system_message=f"You are an expert in fetching images from a database!\n {transition_message}" \
        # "Example:\n" \
        # "1. [Objective]: Zoom in on the capital of Texas.\n" \
        # "You DO NOT have the tools to complete the task, so you return 'PASS'.\n" \
        # "2. [Objective]: Fetch images from the FAIR1M dataset and filter the images from the UK.\n" \
        # "You DO have the tools to complete the task, so you go on and try to fetch the asked images. Now, if something happens that prevents you from fetching the images, " \
        # "you return 'ERROR'.\n"
    )
    detector_agent = SingleAgent(
        api=args.api,
        name="detector_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[vision],
        system_message=f"You are an expert in processing images fetched from a database, such as object detection! \n {transition_message}" \
        # system_message=f"You are an expert in processing images fetched from a database, such as object detection! \n {transition_message}" \
        # "Example:\n" \
        # "1. [Objective]: Zoom in on the capital of Texas.\n" \
        # "You DO NOT have the tools to complete the task, so you return 'PASS'.\n" \
        # "2. [Objective]: Run the Swin-L detector on xView1 images to detect Passenger Vehicles.\n" \
        # "You DO have the tools to complete the task, so you go on and try to run the detector.  Now, if something happens that prevents you from running the detector, " \
        # "you return 'ERROR'\n"
    )
    map_agent = SingleAgent(
        api=args.api,
        name="map_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[map_tools],
        system_message=f"You are an expert in performing all kinds of operations on a map! \n {transition_message}" \
        # system_message=f"You are an expert in performing all kinds of operations on a map! \n {transition_message}" \
        # "Example:\n" \
        # "1. [Objective]: Run the Swin-L detector on xView1 images to detect Passenger Vehicles.\n" \
        # "You DO NOT have the tools to complete the task, so you return 'PASS'.\n" \
        # "2. [Objective]: Zoom in on the capital of Texas.\n" \
        # "You DO have the tools to complete the task, so you go on and try to zoom the map.  Now, if something happens that prevents you from zooming the map, " \
        # "you return 'ERROR'.\n"
    )
    data_agent = SingleAgent(
        api=args.api,
        name="data_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[data_tools],
        system_message=f"You are an expert in all kinds of image analyzing tasks! \n {transition_message}" \
        # system_message=f"You are an expert in all kinds of image analyzing tasks! \n {transition_message}" \
        # "Example:\n" \
        # "1. [Objective]: Run the Swin-L detector on xView1 images to detect Passenger Vehicles.\n" \
        # "You DO NOT have the tools to complete the task, so you return 'PASS'.\n" \
        # "2. [Objective]: Count the number of filtered xView1 images from Greece.\n" \
        # "You DO have the tools to complete the task, so you go on and try to count the images. Now, if something happens that prevents you from counting the images, " \
        # "you return 'ERROR'.\n"
    )
    dummy_agent = SingleAgent(
        api=args.api,
        name="dummy_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[],
        system_message="Placeholder."
    )

    # process states for sequential StateFlow
    state_stack = deque([map_agent, data_agent, detector_agent, database_agent])
    state_list = [map_agent, data_agent, detector_agent, database_agent]

    platform = Platform(model_client, messages, database, vision, map_tools, dummy_agent)
    agent_run = AgentRun(platform, results_output_filenames=get_results_path(args))

    start_time = time.time()
    response = platform.agent.run_seq_stateflow(
        state_list=state_list,
        query=query,
        state_stack=state_stack,
    )
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)
    print("===\nelapsed time: ", elapsed_time, " s\n===")

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
    parser.add_argument('--agent', default= 'stateflow', help='agent to use')
    args = parser.parse_args()

    geo_flow = load_json_file(f'./prompt_tests/benchmark/geo_{args.exp_id}/flow_gt.json')
    main(args, geo_flow['tasks'], geo_flow["query"])