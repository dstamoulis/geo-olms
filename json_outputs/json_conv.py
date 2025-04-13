import json

# with open('output.txt') as file:
#     file_content = file.read()

# # print(file_content)

# with open("output.json", "w") as f:
#     json.dump(json.loads(file_content), f, indent=4)

json.dump(json.load(open("output.json")), open("past_jsons/output_4.json", "w"), indent=2)  # Format JSON file with indentation