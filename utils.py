import json
import time
import os
import re

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
