"""
Update (March 2025):
The OpenAI Agents SDK (https://github.com/openai/openai-agents-python) now offers native API support 
for tool definition via the `@function_tool` decorator 
(https://openai.github.io/openai-agents-python/tools/#function-tools).

Roadmap:
Migrate to the OpenAI Agents API.
"""

def agent_tool(func):
    """
    Decorator that marks a function as exposable to an agent.
    """
    func.is_agent_tool = True
    return func

def agent_toolset(obj) -> list:
    """
    Returns a list with callable methods that are marked with
    the 'is_agent_tool' attribute on the given object.

    Args:
        obj: The object (typically an agent instance) to inspect.

    Returns:
        A list where items callable methods.
    """
    tools = []
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if callable(attr) and getattr(attr, "is_agent_tool", False):
            tools.append(attr)
    return tools