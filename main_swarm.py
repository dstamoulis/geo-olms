# main_agent.py

from agent_core.modules.messages import Messages, TextMessage
from agent_core.agents.assistant_agent import AssistantAgent
from agent_core.teams.swarm_agent import SwarmAgent

from geoapps.geeo.database import Database
from geoapps.geeo.map_ops import Map
from geoapps.geeo.detector import Detector
from geoapps.geeo.triage import Triage

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
    detector = Detector(database)    
    map_ops = Map(database, detector, map_style="open-street-map")
    triage = Triage()

    swarm_agent = SwarmAgent(
        name="swarm_agent",
        model_client=model_client,
        messages=messages,
        database=database, 
        detector=detector, 
        map_ops=map_ops,
        triage=triage
    )

    # Solving with an agent!
    platform = Platform(model_client, messages, database, detector, map_ops, swarm_agent)
    agent_run = AgentRun(platform, results_output_file='results/swarm_agent_test.json')
    query = 'Fetch xView1 images from Athens International Airport, Greece. Consider a wide area. Then run the Swin-L detector and finally please zoom the map there!'
    response = platform.agent.run_query(query)
    print(response)

    print(platform.database.images_gdf["xView1"])
    # print(platform.detector.detections_gdf["xView1"])
    # agent_run.add_task_result(AgentTask(queries=[query], rounds=[{"query": query, "messages": platform.messages.to_list_dict()}]))
    # agent_run.save_inference_results()
    # platform.reset()



if __name__ == "__main__":
    main()