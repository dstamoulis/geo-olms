#!/usr/bin/env python3
import os
import json

def create_flow_gt(base_dir="prompt_tests/benchmark"):
    """
    For each subfolder geo_0 through geo_25 in base_dir, read 'flow.json' and 'query.txt',
    then create 'flow_gt.json' containing both the original 'tasks' and the 'query' string.
    """
    for i in range(22):
        folder_name = f"geo_{i}"
        folder_path = os.path.join(base_dir, folder_name)
        flow_json_path = os.path.join(folder_path, "flow.json")
        query_txt_path = os.path.join(folder_path, "query.txt")
        output_path = os.path.join(folder_path, "flow_gt.json")

        # Skip if the folder or required files don't exist
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_path}: not a directory")
            continue
        if not os.path.isfile(flow_json_path):
            print(f"Skipping {folder_path}: 'flow.json' not found")
            continue
        if not os.path.isfile(query_txt_path):
            print(f"Skipping {folder_path}: 'query.txt' not found")
            continue

        # Load the existing flow.json
        with open(flow_json_path, "r") as f:
            flow_data = json.load(f)

        # Read the first line of query.txt
        with open(query_txt_path, "r") as f:
            query_line = f.readline().rstrip("\n")

        # Construct the combined dictionary
        combined = {
            "query": query_line,
            **flow_data  # assumes flow_data has a top-level "tasks" key
        }

        # Write out flow_gt.json
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"Created {output_path}")

if __name__ == "__main__":
    create_flow_gt()
