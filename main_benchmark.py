# main_agent.py

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
    
    single_agent = SingleAgent(
        name="single_agent",
        model_client=model_client,
        messages=messages,
        toolsets_list=[database, vision, map_tools, data_tools],
        system_message="You are a geospatial agent helping with fetching images from a database!"
    )

    # Solving with an agent!
    platform = Platform(model_client, messages, database, vision, map_tools, single_agent)
    agent_run = AgentRun(platform, results_output_file='results/geeo25_val_run2.json')
    agent_run.load_dataset("benchmarks/geeo25/geeo25_val.json")
    agent_run.run_all()
    agent_run.save_inference_results()
    platform.reset()


if __name__ == "__main__":
    main()