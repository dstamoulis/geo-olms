from __future__ import annotations

"""
Note (Jan 2025):
This module follows the logic outlined by Ollama:
    - https://ollama.com/blog/functions-as-tools
    - https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py
which, in turn, leverages Pydantic and docstring parsing for JSON generation 
(as described in the Google Python Style Guide: 
    https://google.github.io/styleguide/pyguide.html#doc-function-raises).
We retain the Pydantic + docstring parsing approach while removing the Ollama-specific tool logic 
(e.g., removing the import of Tool-related classes from ollama._types).

Update (March 2025):
The OpenAI Agents SDK (https://github.com/openai/openai-agents-python) now offers native API support 
for Pydantic-based tool parsing and definition via the `function_tool` import and `@function_tool` decorator 
(https://openai.github.io/openai-agents-python/tools/#function-tools).

Roadmap:
Migrate to the OpenAI Agents API.
"""

import inspect
import re
import json
from collections import defaultdict
from typing import Callable, Union

import pydantic


def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
    parsed_docstring = defaultdict(str)
    if not doc_string:
        return parsed_docstring

    key = hash(doc_string)
    for line in doc_string.splitlines():
        lowered_line = line.lower().strip()
        if lowered_line.startswith('args:'):
            key = 'args'
        elif lowered_line.startswith('returns:') or lowered_line.startswith('yields:') or lowered_line.startswith('raises:'):
            key = '_'
        else:
            # maybe change to a list and join later
            parsed_docstring[key] += f'{line.strip()}\n'

    last_key = None
    for line in parsed_docstring['args'].splitlines():
        line = line.strip()
        if ':' in line:
            # Split the line on either:
            # 1. A parenthetical expression like (integer) - captured in group 1
            # 2. A colon :
            # Followed by optional whitespace. Only split on first occurrence.
            parts = re.split(r'(?:\(([^)]*)\)|:)\s*', line, maxsplit=1)

            arg_name = parts[0].strip()
            last_key = arg_name

            # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
            arg_description = parts[-1].strip()
            if len(parts) > 2 and parts[1]:
                # Has parenthetical content
                arg_description = parts[-1].split(':', 1)[-1].strip()

            parsed_docstring[last_key] = arg_description

        elif last_key and line:
            parsed_docstring[last_key] += ' ' + line

    return parsed_docstring


def convert_function_to_tool(func: Callable) -> Tool:
    doc_string_hash = hash(inspect.getdoc(func))
    parsed_docstring = _parse_docstring(inspect.getdoc(func))
    schema = type(
        func.__name__,
        (pydantic.BaseModel,),
        {
            '__annotations__': {k: v.annotation if v.annotation != inspect._empty else str for k, v in inspect.signature(func).parameters.items()},
            '__signature__': inspect.signature(func),
            '__doc__': parsed_docstring[doc_string_hash],
        },
    ).model_json_schema()

    for k, v in schema.get('properties', {}).items():
        # If type is missing, the default is string
        types = {t.get('type', 'string') for t in v.get('anyOf')} if 'anyOf' in v else {v.get('type', 'string')}
        if 'null' in types:
            schema['required'].remove(k)
            types.discard('null')

        schema['properties'][k] = {
            'description': parsed_docstring[k],
            'type': ', '.join(types),
        }

    return func, schema


def function_to_tool_json(func: Callable) -> str:
    # Assume convert_function_to_tool returns a tuple (func, schema)
    _func, schema = convert_function_to_tool(func)
    
    # Extract the top-level description for the tool.
    tool_description = schema.get("description", "")
    
    # Remove extraneous keys from the schema for parameters.
    schema.pop("title", None)
    schema.pop("description", None)
    
    # Build the final tool schema dictionary.
    tool_dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": tool_description,
            "parameters": schema
        }
    }
    
    # Return the JSON string formatted with an indent of 4 spaces.
    # return json.dumps(tool_dict, indent=4)
    # print(json.dumps(tool_dict, indent=4))
    return tool_dict
