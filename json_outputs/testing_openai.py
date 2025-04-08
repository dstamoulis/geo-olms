from openai import OpenAI
from autogen_structure import AutogenWorkflow
import os
import keys
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
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=AutogenWorkflow
    )
    
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
    
    result = query_chatgpt_for_autogen_workflow(user_query)

    with open("output.json", "w") as f:
        json.dump(json.loads(result), f, indent=4)
    
    # with open("output.txt", "w") as f:
    #     print(result, file=f)