import json
import time
import os
import re

DATABASE_AGENT_SYSTEM = "You are the database agent with tools for database queries and image filtering !"
MAP_AGENT_SYSTEM = "You are the map agent with map zooming, plotting, and visualization tools for satellite images and detections!"
DETECTOR_AGENT_SYSTEM = "You are the detector agent for detection and classification (land cover) on satellite images!"
DATA_AGENT_SYSTEM = "You are a data analysis agent with data-count and analytics tools for satellite images!"


DATABASE_AGENT_USER = (
    "You are the database_agent in a 4-step pipeline which typically (but not always as some parts might not needed) is "
    "database_agent → detector_agent → data_agent → map_agent. "
    "Your objective is self-contained: fetch and filter imagery by dataset, location, and date. "
    "Run only if the workflow needs data without overstepping your scope; if no data fetch is required, skip yourself."
    "Here's your current yours-specific objective (subtask) to help solve: "
)

DETECTOR_AGENT_USER = (
    "You are the detector_agent in a 4-step pipeline which typically (but not always as some parts might not needed) is "
    "database_agent → detector_agent → data_agent → map_agent. "
    "Your objective is self-contained: run object detection or classification on provided images. "
    "Execute only if detection is required without overstepping your scope; otherwise, skip yourself."
    "Here's your current yours-specific objective (subtask) to help solve: "
)

DATA_AGENT_USER = (
    "You are the data_agent in a 4-step pipeline which typically (but not always as some parts might not needed) is "
    "database_agent → detector_agent → data_agent → map_agent. "
    "Your objective is self-contained: perform counts or analytics on images or detections. "
    "Run only if data analysis is needed without overstepping your scope; otherwise, skip yourself."
    "Here's your current yours-specific objective (subtask) to help solve: "
)

MAP_AGENT_USER = (
    "You are the map_agent in a 4-step pipeline which typically (but not always as some parts might not needed) is"
    "database_agent → detector_agent → data_agent → map_agent. "
    "Your objective is self-contained: generate map visualizations (zoom, plot). "
    "Execute only if mapping is required without overstepping your scope; otherwise, skip yourself."
    "Here's your current yours-specific objective (subtask) to help solve: "
)

GROUP_MANAGER_INSTRUCTIONS = """

        You are an orchestrating agent acting as a `group manager` of a `group chat`.
        As part of the group chat (with the group of agents sharing a common thread of messages), you will 
        dynamically decompose the user task into smaller ones (subtasks) that can be handled by specialized 
        agents with well-defined roles. As a group manager, you will be orchestrating execution and handing 

        Your expertise is on geospatial tasks, where typical flows follow the following order:
            database_agent → detector_agent → data_agent → map_agent
        ** Typically, but not always! An agent might be skipped if not needed towards helping with the task
        For example, if there are not data-analysis operations or map/plot operations, the respective (sub)agents
        will not be needed.

        -----

        Think of the workflow of completing a given task modeled as a state-machine,
        so effectively your job is to decide the next agent (state) to transition to based on the current 
        state to continue with the next agent-specific subtask!

        INSTRUCTIONS: Select the next specific agent (only one at a time!!).

        Each task is self-contained and maps to a single agent, so you are choosing -- only one at a time -- from the following agents:

        > database_agent: tools for database queries and image filtering  
        > detector_agent: detection and classification on prior images  
        > data_agent: data-count and analytics tools  
        > map_agent: map zooming, plotting, and visualization tools
        > end_agent: agent responsible for finalizing the task and summarizing the results back to the user!!

        To do this, you need to properly think about the breakdown of the (global) task to the agent-specific objective related to its part!
        Each objective needs to be self-contained and make sure to include dataset names, locations, models, plot types and all 
        other info needed for the subagents to perform function calling!

        For example, consider the following scenario global task

        >> Task: Fetch BigEarthNet in Switzerland for and run the ResNet-32 classifier. Count how many 'Vineyards' LCC classes there are. Also, please plot on the map the 'Fruit trees and berry plantations' LCC classes!

        in that case you will return at SEPARATE ROUNDS

        >> To "Fetch images from the BigEarthNet dataset and filter the images specifically from Switzerland." you need: {"next_agent": "database_agent"}
        
        OR:

        >> To "Run the ResNet-32 classifier on BigEarthNet images to detect 'Vineyards' and 'Fruit trees and berry plantations'." you need: {"next_agent": "detector_agent"}

        OR:

        >> To "Count the ResNet-32 classification results for 'Vineyards' class." you need: {"next_agent": "data_agent"}
        
        OR:

        >> To "Plot ResNet-32 classification results highlighting the 'Fruit trees and berry plantations' class on the map." you need: {"next_agent": "map_agent"}

        OR:

        >> To "Successfully return and notify the user that you fetched BigEarthNet in Switzerland for ... plotted the 'Vineyards' and 'Fruit trees and berry plantations' LCC classes!"  you need: {"next_agent": "end_agent"}


        REPLY FORMAT:
        - To handoff to next agent your message should ONLY be a VALID dictionary with the key next_agent and a VALID agent name!

        Valid output example:
        {"next_agent": "database_agent"}
                    
        ATTENTION:
        - GIVE UP IF YOU FIND YOURSELF REPEATING THE SAME TOOL CALL OVER AND OVER!!

        Now it's your turn—carry out your step. As a reminder, the global user task is:

        """


SUBTASK_INSTRUCTIONS = """

    "You are the {next_agent_name} part of a 4-step pipeline which typically (but not always as some parts might not needed) is "
    "database_agent → detector_agent → data_agent → map_agent"

    INSTRUCTIONS: Given the global objective shared by all agents, determine which part corresponds to your tools. 
    Your goal is to execute the steps needed to solve your specific subtask.
    Think of your objective (subtask) as self-contained, i.e.: {scope}
    Run only if the workflow needs data without overstepping your scope; if your action is not required, skip yourself."

    For example, consider the following scenario

    >> Task: Fetch BigEarthNet in Switzerland for and run the ResNet-32 classifier. Count how many 'Vineyards' LCC classes there are. Also, please plot on the map the 'Fruit trees and berry plantations' LCC classes!

    in that case your part corresponds to solving for the following subobjective:

    >> Your subtask: {subtask_example}

    which you should help solving based on your tools.

    Now it's your turn—carry out your step. As a reminder, the global user task is:

    """


SUBAGENTS_INSTRUCTIONS = {
        "database_agent": {
            "example": "Fetch images from the BigEarthNet dataset and filter the images specifically from Switzerland.",
            "scope": "fetch and filter imagery by dataset, location, and date."
        },
        "detector_agent":  {
            "example": "Run the ResNet-32 classifier on BigEarthNet images to detect 'Vineyards' and 'Fruit trees and berry plantations'.",
            "scope": "run object detection or classification on provided images."
        },
        "map_agent":  {
            "example": "Plot ResNet-32 classification results highlighting the 'Fruit trees and berry plantations' class on the map.",
            "scope": "generate map visualizations (zoom, plot)."
        },
        "data_agent": {
            "example":  "Count the ResNet-32 classification results for 'Vineyards' class.",
            "scope": "perform counts or analytics on images or detections."
        }
}


MAGENTIC_PROMPT = """
        You are an orchestrating agent acting as a `group manager` of a `group chat`.
        As part of the group chat (with the group of agents sharing a common thread of messages), you will 
        dynamically decompose the user task into smaller ones (subtasks) that can be handled by specialized 
        agents with well-defined roles. As a group manager, you will be orchestrating execution and handing 

        Your expertise is on geospatial tasks, where typical flows follow the following order:
            database_agent → detector_agent → data_agent → map_agent
        ** Typically, but not always! An agent might be skipped if not needed towards helping with the task
        For example, if there are not data-analysis operations or map/plot operations, the respective (sub)agents
        will not be needed.

        -----

        Think of the workflow of completing a given task modeled as a state-machine,
        so effectively your job is to decide the next agent (state) to transition to based on the current state, 
        as well as what's their exact subtask objective!

        INSTRUCTIONS: Your job it twofold

        1. Explicilty select the next specific agent (only one at a time!!).

        Each task is self-contained and maps to a single agent, so you are choosing -- only one at a time -- from the following agents:

        > database_agent: tools for database queries and image filtering  
        > detector_agent: detection and classification on prior images  
        > data_agent: data-count and analytics tools  
        > map_agent: map zooming, plotting, and visualization tools
        > end_agent: agent responsible for finalizing the task and summarizing the results back to the user!!

        2. Spell out explicitly the subtask for the agent as its self-contained specific objective.

        To do this, you need to properly breakdown the (global) task to the agent-specific objective related to its part!

        Each objective needs to be self-contained and make sure to include dataset names, locations, models, plot types and all 
        other info needed for the subagents to perform function calling!

        For example, consider the following scenario

        >> Task: Fetch BigEarthNet in Switzerland for and run the ResNet-32 classifier. Count how many 'Vineyards' LCC classes there are. Also, please plot on the map the 'Fruit trees and berry plantations' LCC classes!

        in that case you will return at SEPARATE ROUNDS

        >> {"database_agent": "Fetch images from the BigEarthNet dataset and filter the images specifically from Switzerland."}
        
        OR:

        >> {"detector_agent": "Run the ResNet-32 classifier on BigEarthNet images to detect 'Vineyards' and 'Fruit trees and berry plantations'."}

        
        OR:

        >> {"data_agent": "Count the ResNet-32 classification results for 'Vineyards' class."}
        
        OR:

        >> {"map_agent": "Plot ResNet-32 classification results highlighting the 'Fruit trees and berry plantations' class on the map."}

        OR:

        >> {"end_agent": "Successfully fetched BigEarthNet in Switzerland for and run the ResNet-32 classifier and plotted the 'Vineyards' and 'Fruit trees and berry plantations' LCC classes!"}

        REPLY FORMAT:
        - To handoff to next agent your message should ONLY be a VALID dictionary with the agent name and its objective!

        Valid output example:
        {"database_agent": "Objective is to ..."}
                    
        ATTENTION:
        - GIVE UP IF YOU FIND YOURSELF REPEATING THE SAME TOOL CALL OVER AND OVER!!
    """


def re_args_component(value: str) -> str:
    return re.sub(r"[.\-\:]", "_", value)

def get_results_path(args, base_dir="results") -> str:
    """
    Construct and create a directory hierarchy and filename based on the provided argparse Namespace:
        results/{client}/{model}/{agent}/{client}_{model}_{temp}_{agent}_{exp_id}.json

    Returns the full file path to the JSON file, creating any missing folders.
    """
    client = re_args_component(str(args.client))
    model = re_args_component(str(args.model))
    temp = re_args_component(str(args.temp))
    agent = re_args_component(str(args.agent))
    exp_id = re_args_component(str(args.exp_id))
    flow_ver = re_args_component(str(args.flow_ver))

    # Directory hierarchy: results/{client}/{model}/{agent}/{exp_id}
    results_path = os.path.join(base_dir, flow_ver, client, model, agent, exp_id)
    os.makedirs(results_path, exist_ok=True)

    # Build the base filenames
    results_filenames = {
        "results_path": str(results_path),
        "result": os.path.join(results_path, "result.json"),
        "images_gdf": os.path.join(results_path, "images_gdf.json"),
        "detections_gdf": os.path.join(results_path, "detections_gdf.json"),
        "map_state": os.path.join(results_path, "map_state.json"),
    }
    return results_filenames 

def strip_json_code_block(raw_content):
    # Remove leading/trailing whitespace
    raw_content = raw_content.strip()
    
    # Remove triple backtick wrapper if present
    if raw_content.startswith("```json") or raw_content.startswith("```"):
        # Split by lines, remove the first and last lines
        lines = raw_content.splitlines()
        # Handle empty or malformed content defensively
        if len(lines) >= 3:
            return "\n".join(lines[1:-1])
    return raw_content

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
    

def parse_llm_delimiter_message(raw: str, delimiter_str) -> bool:
    """
    Return True if the LLM response ends with delimiter_str (ignoring quotes, <think>...</think>, 
    or trailing explanation).  Handles delimiter_str like ERROR, DONE, etc. 
    """
    # 1) Remove any <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)

    # 2) Split into lines and keep only non-blank lines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    if not lines:
        return False

    # 3) Consider the last non-blank line
    last = lines[-1]

    # 4) Strip surrounding quotes or backticks
    last = last.strip(' "\'`')

    # 5) Check if it equals ERROR
    return last.upper() == delimiter_str


def extract_json_object(raw: str) -> str:
    """
    Extracts the first JSON object from `raw` text, ignoring any <think>...</think> sections.
    1) Strip out any <think>…</think> blocks entirely.
    2) If there's a ```json...``` code block, return its contents.
    3) Otherwise, find the first “{” and grab the balanced {...} substring.
    Returns the JSON string or raises ValueError if none found.
    """
    # 0) Remove any <think>...</think> sections
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Trim whitespace
    raw = raw.strip()

    # 1) Try to pull from a ```json …``` block
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return m.group(1)

    # 2) Otherwise look for the first “{” and match braces
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("Unbalanced braces in response")
