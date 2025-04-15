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

from geoplatform.platform_studio import PlatformStudio

import json

import os
import json_outputs.keys as keys
os.environ["OPENAI_API_KEY"] = keys.get_openai_key()

def main():

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


    platform_studio = PlatformStudio(model_client, messages, database, vision, map_tools, data_tools)
    
    query = 'Fetch xView1 images from Athens International Airport, Greece. Consider a wide area. Then run the Swin-L detector and finally please zoom the map there!'

    workflow_json = 'results/gt1.json'
    with open(workflow_json, 'r') as workflow_file:
        workflow_dict = json.load(workflow_file)

    response = platform_studio.run_workflow(workflow_dict)

    print(response)
    print(platform_studio.database.images_gdf)
    print(platform_studio.vision.detections_gdf)


if __name__ == "__main__":
    main()