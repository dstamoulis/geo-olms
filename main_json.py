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

from openai import OpenAI
from json_outputs.autogen_structure import AutogenWorkflow
import os
import json_outputs.keys as keys
import json

os.environ["OPENAI_API_KEY"] = keys.get_openai_key()
client = OpenAI()

# api fcn
def query_chatgpt_for_autogen_workflow(user_query: str):
    # This system prompt instructs the assistant to generate a JSON output
    # that conforms to an autogen workflow schema (like your provided travel_team.json).
    system_prompt = (
        "You are a helpful assistant that generates agentic workflows following the autogen documentation. "
        "Given a natural language description of a task, produce a strictly formatted JSON output that defines an autogen workflow. "
        "The JSON object must include keys such as 'provider', 'component_type', 'version', 'component_version', 'description', 'label', and 'config'. "
        "Do not include any extra text outside the JSON."
    )

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
        system_message=system_prompt,
    )
    platform = Platform(model_client, messages, database, vision, map_tools, single_agent)
    response = platform.agent.run_query(user_query)
    
    # Retrieve the generated output and strip extra whitespace
    output_text = response.choices[0].message.content

    return output_text

if __name__ == "__main__":

    user_query = \
        """Please create an agentic workflow json template that can solve this task: 
        
        Fetch xView1 images from Athens International Airport, Greece. Consider a 
        wide area. Then run the Swin-L detector and finally please zoom the map there! 

        Appropriately break down the task and assign relevant realistic agent 
        definitions including tools and a smart workflow coordination of these agents.
        """
    
#     json_mod_query = \
#         """Please edit this agentic workflow json template: 
        
#         I provide a agentic workflow task breakdown, please convert this to an autogen agentic workflow json template.

#         {
#   "tasks": {
#     "task0": {
#       "id": "task0",
#       "objective": "Identify and define the geographical bounds for the wide area around Athens International Airport for image fetching.",
#       "agent_id": 0,
#       "next": [
#         "task1",
#         "task3"
#       ],
#       "prev": [],
#       "status": "completed",
#       "history": "**Geographical Bounds for Athens International Airport, Greece:**\n\n- **Latitude Range:**\n  - Minimum Latitude: 37.88\n  - Maximum Latitude: 37.95\n\n- **Longitude Range:**\n  - Minimum Longitude: 23.78\n  - Maximum Longitude: 23.90\n\nThese bounds encompass a wide area around Athens International Airport, suitable for fetching xView1 images.",
#       "remaining_dependencies": 0,
#       "agent": "Geospatial Analyst"
#     },
#     "task1": {
#       "id": "task1",
#       "objective": "Fetch xView1 images based on the defined geographical bounds from Athens International Airport.",
#       "agent_id": 1,
#       "next": [
#         "task2"
#       ],
#       "prev": [
#         "task0"
#       ],
#       "status": "completed",
#       "history": "**Fetch xView1 Images from Athens International Airport, Greece:**\n\n**Geographical Bounds:**\n- **Latitude Range:**\n  - Minimum Latitude: 37.88\n  - Maximum Latitude: 37.95\n\n- **Longitude Range:**\n  - Minimum Longitude: 23.78\n  - Maximum Longitude: 23.90\n\n**Fetch Command:**\n- Fetch the xView1 images within the specified geographical bounds. Ensure the parameters are set to retrieve all relevant images captured in the area. \n\n**Data Output:**\n- Provide a comprehensive list or dataset of xView1 images that fall within the defined latitude and longitude ranges. This data should include image identifiers, timestamps, and any other pertinent metadata to facilitate the next steps in the workflow. \n\n**Map Zoom Instructions:**\n- Once the images are fetched, zoom into the area defined by the latitude and longitude bounds on a mapping platform to visualize the coverage of the images obtained. \n\nEnsure that the correct API or data retrieval methods for accessing xView1 images are used, and confirm the successful fetching of images before proceeding to the next task of applying the Swin-L detector.",
#       "remaining_dependencies": 0,
#       "agent": "Data Acquisition Specialist"
#     },
#     "task2": {
#       "id": "task2",
#       "objective": "Run the Swin-L detector on the fetched xView1 images to identify objects and generate a detailed report summarizing the detected objects and their locations.",
#       "agent_id": 2,
#       "next": [],
#       "prev": [
#         "task1"
#       ],
#       "status": "completed",
#       "history": "**Fetch xView1 Images from Athens International Airport, Greece:**\n\n**Geographical Bounds:**\n- **Latitude Range:**\n  - **Minimum:** 37.88\n  - **Maximum:** 37.95\n\n- **Longitude Range:**\n  - **Minimum:** 23.78\n  - **Maximum:** 23.90\n\n**Data Fetch Command:**\n- Execute the following command to retrieve the xView1 images: \n  ```\n  fetch_xview1_images(latitude_min=37.88, latitude_max=37.95, longitude_min=23.78, longitude_max=23.90)\n  ```\n\n**Expected Data Output:**\n- A comprehensive dataset of fetched xView1 images including:\n  - **Image Identifiers**\n  - **Timestamps**\n  - **Metadata (if available)**\n\n**Map Zoom Instructions:**\n- After images are fetched, utilize a mapping platform (e.g., Google Maps, OpenStreetMap) and input the following bounds to zoom into the area:\n  - Latitude: 37.88 to 37.95\n  - Longitude: 23.78 to 23.90\n\n**Swin-L Detector Execution:**\n- Process the fetched images through the Swin-L detector to identify objects. \n\n**Report Generation:**\n- Compile a detailed report summarizing:\n  - Detected objects\n  - Their respective locations (coordinates)\n\nEnsure successful completion of each step before proceeding.",
#       "remaining_dependencies": 0,
#       "agent": ""
#     },
#     "task3": {
#       "id": "task3",
#       "objective": "Zoom the map around Athens International Airport to visualize the area where images were fetched.",
#       "agent_id": 3,
#       "next": [],
#       "prev": [
#         "task0"
#       ],
#       "status": "completed",
#       "history": "**Zoomed Map Visualization for Athens International Airport, Greece:**\n\n- **Geographical Bounds:**\n  - **Latitude Range:** \n    - Minimum: 37.88\n    - Maximum: 37.95\n  - **Longitude Range:** \n    - Minimum: 23.78\n    - Maximum: 23.90\n\n- **Map Zoom Level:** Set to a level that provides a clear view of the area including the airport and surrounding regions. Recommended zoom level: 13-15 for adequate detail.\n\n- **Suggested Map Link:** Utilize a mapping service to visualize the area. For example: [OpenStreetMap](https://www.openstreetmap.org/#map=14/37.9254/23.8392). Adjust the zoom level within this link accordingly.\n\nEnsure to embed the given latitude and longitude ranges into the map interface to confirm coverage of the specified area.",
#       "remaining_dependencies": 0,
#       "agent": "Mapping Specialist"
#     }
#   }
# }
#         """
    
    result = query_chatgpt_for_autogen_workflow(user_query)

    with open("output.json", "w") as f:
        json.dump(json.loads(result), f, indent=2)
    
    # with open("output.txt", "w") as f:
    #     print(result, file=f)


# ERROR:
'''
  File "C:\Users\ranit\GeoLLMs\geo-olms\agent_core\agents\assistant_agent.py", line 84, in get_response
    chat_response = self.model_client.get_response(messages, tools=self.tool_schemas)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ranit\GeoLLMs\geo-olms\llm_clients\openai_client.py", line 39, in get_response
    response = self.api_client.beta.chat.completions.parse(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ranit\GeoLLMs\geo-olms\.venv\Lib\site-packages\openai\resources\beta\chat\completions.py", line 144, in parse
    _validate_input_tools(tools)
  File "C:\Users\ranit\GeoLLMs\geo-olms\.venv\Lib\site-packages\openai\lib\_parsing\_completions.py", line 53, in validate_input_tools
    raise ValueError(
ValueError: `query_dataset_images` is not strict. Only `strict` function tools can be auto-parsed
'''