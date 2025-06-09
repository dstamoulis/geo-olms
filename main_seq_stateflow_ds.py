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


    # SYSTEM: You are the {agent_name} in a 3-step sequential pipeline:
    # database_agent → detector_agent → data_agent → map_agent

    transition_message = """\
        SYSTEM: You are the {agent_name} in a 3-step sequential pipeline:
        database_agent → detector_agent → map_agent

        REPLY FORMAT:
        - Your message must be exactly “NEXT” or “ERROR” (uppercase, no punctuation, no additional text).
        - Do NOT output anything else."""

    handoff_message = """\
        INSTRUCTIONS: You are part of a 3-step sequential pipeline:
        database_agent → detector_agent → map_agent

        On your turn, you must:
        1. Look at the current conversation state and decide if you have any tools or actions to perform.
            • If you have no relevant tools or your step is not needed, consider your turn finished immediately.
            • Otherwise, perform your operations (e.g. database queries, detections, plotting).
            • Your operations are self-contained, so you try to complete your part in a SINGLE go, without repeat the same step over and over!! If done, proceed with NEXT
        2. Monitor for any execution errors (e.g., calling a tool before its inputs exist).
            • If you detect any error or unfulfilled dependency problems from previous steps, you MUST signal ERROR.
        3. When you are done, you must signal exactly one token:
            • “NEXT” if you completed your work successfully or intentionally skipped because you had nothing to do.
            • “ERROR” if you encountered any problem or unfulfilled dependency.

        REPLY FORMAT:
        - Your message must be exactly “NEXT” or “ERROR” (uppercase, no punctuation, no additional text).
        - Do NOT output anything else.

        Example valid replies:
        NEXT
        ERROR

        Now it's your turn—carry out your step, then reply with NEXT or ERROR."""

        
    # Subagents
    database_agent = SingleAgent(
        api=args.api,
        name="database_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[database],
        system_message=f"You are an expert in fetching images from a database!\n {transition_message.format(agent_name='database_agent')}"
    )
    detector_agent = SingleAgent(
        api=args.api,
        name="detector_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[vision],
        system_message=f"You are an expert in processing images fetched from a database, such as object detection! \n {transition_message.format(agent_name='detector_agent')}"
    )
    map_agent = SingleAgent(
        api=args.api,
        name="map_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[map_tools],
        system_message=f"You are an expert in performing all kinds of operations on a map! \n {transition_message.format(agent_name='map_agent')}"
    )
    data_agent = SingleAgent(
        api=args.api,
        name="data_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[data_tools],
        system_message=f"You are an expert in all kinds of image analyzing tasks! \n {transition_message.format(agent_name='data_agent')}"
    )
    orch_agent = SingleAgent(
        api=args.api,
        name="orch_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[],
        system_message="Placeholder."
    )

    agents_dict = {"database_agent": database_agent, "map_agent": map_agent, "detector_agent": detector_agent, "data_agent": data_agent}
    agents_sequence = ["database_agent", "detector_agent", "data_agent", "map_agent"]

    agents_dict = {"database_agent": database_agent, "map_agent": map_agent, "detector_agent": detector_agent}
    agents_sequence = ["database_agent", "detector_agent", "map_agent"]
    platform = Platform(model_client, messages, database, vision, map_tools, orch_agent)
    agent_run = AgentRun(platform, results_output_filenames=get_results_path(args))

    start_time = time.time()
    response = platform.agent.run_seq_stateflow_ds(
        query=query,
        handoff_message=handoff_message,
        agents=agents_dict,
        agents_sequence=agents_sequence
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
    parser.add_argument('--agent', default= 'seq_stateflow', help='agent to use')
    args = parser.parse_args()

    geo_flow = load_json_file(f'./prompt_tests/benchmark/geo_{args.exp_id}/flow_gt.json')
    main(args, geo_flow['tasks'], geo_flow["query"])