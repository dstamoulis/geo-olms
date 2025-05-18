from agent_core.modules.messages import ChatResponseMessage, ToolCall, ToolCallRequestMessage
import json

response = {}
def call_function(name, args):
    pass

for tool in response.output:
    if tool.type != "function_call":
        continue

    message_type = ToolCallRequestMessage
    response_tool_calls = []

    tool_call = ToolCall(
        id=tool.id,
        name=tool.name,
        arguments=json.loads(tool.arguments),
        tool_type=tool.type
    )
    response_tool_calls.append(tool_call)
