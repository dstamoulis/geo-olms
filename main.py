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
    orch_agent = SingleAgent(
        api=args.api,
        name="orch_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[],
        system_message="You are an orchastrating agent handing off tasks to subagents to complete given task!"
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
    # platform = Platform(model_client, messages, database, vision, map_tools, single_agent)
    platform = Platform(model_client, messages, database, vision, map_tools, orch_agent)
    agent_run = AgentRun(platform, results_output_file='./results/single_agent_test.json')
    query = 'Fetch xView1 images from Athens International Airport, Greece. Consider a wide area. Then run the Swin-L detector and finally please zoom the map there!'
    query = 'Fetch xView1 and FAIR1M images from July 2017. Then run the Swin-L detector on each imagery source. '
    query = 'Fetch xView1 and FAIR1M images from July 2017. Then run the Swin-L detector on each imagery source. Last, from FAIR1M, plot the detections of category Van.'
    query = "Fetch BigEarthNet images from June 2017. Then run the ResNet-32 LCC classifer on the images. Last, plot the LCC classes of category 'Non-irrigated arable land'."
    query = 'Fetch xView1 images from Greece. Then run the Swin-L detector and plot the detections of category Passenger Vehicle!' # init_2.json
    # query = "Plot on the map the BigEarthNet, xView1 images in Germany from 2nd half of 2017"
    # query = "Plot on the map the xView1 images in Dar es-Salam, Tanzania from Summer 2017! Make sure you consider a very very wide area!"
    # query = "Fetch BigEarthNet in Switzerland for and run the ResNet-32 classifier. Please plot on the map the 'Vineyards' and 'Fruit trees and berry plantations' LCC classes"
    # query = "Zoom the map to the capital of UK please"
    # query = "Tell me a joke please"
    # query = "Fetch xView1 images from Athens International Airport, Greece."
    # query = "Zoom the map to the capital of UK please. What is the name of the capital?"
    # query = "Fetch xView1 images from Greece. How many images we got?"
    # query = "Run the YOLO-v6 LCC model on BigEarthNet from April 2018. How many 'Airports' LCC classification results we got?"
    # query = "Plot on the map the FAIR1M, BigEarthNet, and xView1 images in Greece from 2nd half of 2012"
    # query = "Let's start by getting the xView1 images from Prague, Czech Republic for Summer 2017. Consider a wide area. Then run the Swin-L detector. Last, can you please tell me how many 'Passenger Car' you got?"
    # query = "Where is 39.3434\u00b0 N, 117.3616\u00b0 E? I want to see it on the map; please zoom there!"
    # query = "First, let's load the xView1 images from Signapore for Summer 2017, but please make sure you consider a wide area. Then run the Swin-L detector. Based on the detection counts, which category showed up with more objects in that area: 'Oil Tanker', 'Ferry', or 'Sailboat'? Thanks in advance for the help!"
    # query = "Fetch xView1 images from Turkey. How many images are there?"
    # # added ------------------------------------
    # # make the query vague
    # query = "Zoom the map to the capital of the country where Liverpool is located!"
    # query = "Plot the Swin-L detected Passenger Vehicle on xView1 images in Greece!"
    # # take back the request and change to a new one
    # query = "Fetch xView1 images from Turkey, wait, don't do that, fetch from Greece actually, then run the Swin-L detector and plot the detections of category Passenger Vehicle!"
    # take back + vague
    # query = "Plot the detected Passenger Vehicle on xView1 images in Turkey. Wait, do Greece instead, and use Swin-L please!"

    start_time = time.time()
    # response = platform.agent.run_query(query)
    response = platform.agent.run_workflow(
        {"database_agent": database_agent, "map_agent": map_agent, "detector_agent": detector_agent},
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
    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--api', default='ChatCompletion', help='choose between Responses and ChatCompletion')
    args = parser.parse_args()
    geo_flow = load_json_file('./workflows/geo_1/init_2.json')['tasks']
    main(args, geo_flow)