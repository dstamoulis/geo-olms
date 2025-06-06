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

    # Build the base filename (without extension)
    filename = f"{client}_{model}_{temp}_{agent}_{exp_id}.json"

    # Directory hierarchy: results/{client}/{model}/{agent}
    dir_path = os.path.join(base_dir, client, model, agent)
    os.makedirs(dir_path, exist_ok=True)

    # Full path to the .json file
    return os.path.join(dir_path, filename)



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
    