import json
import time
import os
import re

def re_args_component(value: str) -> str:
    return re.sub(r"[.\-]", "_", value)

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
    