from openai import OpenAI
client = OpenAI()
import prompt
import json
import argparse
import sys
import os
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import utils

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

def generate_workflow(query):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        # Removed {prompt.TASK_EXECUTION_PROMPT}\n
        messages=[
            {
                "role": "user",
                "content": 
                f'''
                {prompt.INIT_WORKFLOW_PROMPT}\n

                For example:
                {prompt.INIT_WORKFLOW_TEMPLATE}\n

                Available Agents:
                {prompt.AGENT_LIST}

                Provide the output in the same format as the example above. Make sure you choose only the agents provided in the \"Available Agents\" to complete the task.

                Here is the task to be executed:

                '{query}'
                '''
        
            }
        ]
    )
    return completion.choices[0].message.content

# Benchmark queries
queries = [
    'Fetch xView1 images from Athens International Airport, Greece. Consider a wide area. Then run the Swin-L detector and finally please zoom the map there!',
    'Fetch xView1 and FAIR1M images from July 2017. Then run the Swin-L detector on each imagery source. ',
    'Fetch xView1 and FAIR1M images from July 2017. Then run the Swin-L detector on each imagery source. Last, from FAIR1M, plot the detections of category Van.',
    "Fetch BigEarthNet images from June 2017. Then run the ResNet-32 LCC classifer on the images. Last, plot the LCC classes of category 'Non-irrigated arable land'.",
    'Fetch xView1 images from Greece. Then run the Swin-L detector and plot the detections of category Passenger Vehicle!',
    "Plot on the map the BigEarthNet, xView1 images in Germany from 2nd half of 2017",
    "Plot on the map the xView1 images in Dar es-Salam, Tanzania from Summer 2017! Make sure you consider a very very wide area!",
    "Fetch BigEarthNet in Switzerland for and run the ResNet-32 classifier. Please plot on the map the 'Vineyards' and 'Fruit trees and berry plantations' LCC classes",
    "Zoom the map to the capital of UK please",
    "Fetch xView1 images from Greece. How many images we got?",
    "Run the YOLO-v6 LCC model on BigEarthNet from April 2018. How many 'Airports' LCC classification results we got?",
    "Plot on the map the FAIR1M, BigEarthNet, and xView1 images in Greece from 2nd half of 2012",
    "Let's start by getting the xView1 images from Prague, Czech Republic for Summer 2017. Consider a wide area. Then run the Swin-L detector. Last, can you please tell me how many 'Passenger Car' you got?",
    "Where is 39.3434\u00b0 N, 117.3616\u00b0 E? I want to see it on the map; please zoom there!",
    "First, let's load the xView1 images from Signapore for Summer 2017, but please make sure you consider a wide area. Then run the Swin-L detector. Based on the detection counts, which category showed up with more objects in that area: 'Oil Tanker', 'Ferry', or 'Sailboat'? Thanks in advance for the help!",
    "Fetch xView1 images from Turkey. How many images are there?",
    "Run the YOLO-v6 LCC model on BigEarthNet from March 2018. Plot the 'Airports' LCC classification results on the map",
    "Plot on the map the xView1 images in (wider area around) Ouagadougou, Burkina Faso, for August 2017",
    "Scatter plot the xView1 images in Kaohsiung, Taiwan from June 2017. I want to make sure that you consider a wide area",
    "Plot on the map the xView1 images in Derby Airport, Australia from August 2017",
    "Run the ResNet-32 LCC model on BigEarthNet from 2nd half of April 2018. How many 'Airports' LCC classification results we got?",
    "Which country has more xView1 images, Greece or Turkey?"
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='geo-olm agent')
    parser.add_argument('--exp_id', default=0, help='run ID to choose')
    args = parser.parse_args()

    geo_path = f'./prompt_tests/benchmark/geo_{args.exp_id}'
    query = utils.load_json_file(geo_path + '/flow_gt.json')['query']

    response = generate_workflow(query)

    # Save the generated GeoFlow JSON
    cleaned_json_str = strip_json_code_block(response)
    parsed_json = json.loads(cleaned_json_str)

    new_data = OrderedDict()
    new_data["query"] = query  # Replace with actual query
    new_data["tasks"] = parsed_json.get("tasks", {})
    with open(f"prompt_tests/benchmark/geo_{args.exp_id}/flow_exp_2.json", "w") as f:
        json.dump(new_data, f, indent=2)