import json

# with open('output.txt') as file:
#     file_content = file.read()

# # print(file_content)

# with open("output.json", "w") as f:
#     json.dump(json.loads(file_content), f, indent=4)

# json.dump(json.load(open("output.json")), open("past_jsons/output_4.json", "w"), indent=2)  # Format JSON file with indentation

def transfigure_tools(input_json):
    def transform_tool(tool_name):
        return {
            "provider": "autogen_core.tools.FunctionTool",
            "component_type": "tool",
            "version": 1,
            "component_version": 1,
            "description": "A tool that performs Google searches",
            "label": tool_name,
            "config": {
                "source_code": "",
                "name": tool_name,
                "description": "Perform Google searches",
                "has_cancellation_support": False
            }
        }

    def recurse_and_transform(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if key == "tools" and isinstance(value, list):
                    new_obj[key] = [transform_tool(tool) for tool in value]
                else:
                    new_obj[key] = recurse_and_transform(value)
            return new_obj
        elif isinstance(obj, list):
            return [recurse_and_transform(item) for item in obj]
        else:
            return obj

    return recurse_and_transform(input_json)
