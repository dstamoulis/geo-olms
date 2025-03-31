"""
Note (Jan 2025):
This module follows the educational implementation outlined in the OpenAI Cookbook example
for orchestrating agents: https://cookbook.openai.com/examples/orchestrating_agents.
Each handoff path must be explicitly defined in code (e.g., transfer_to_x, transfer_to_y, etc.).

Update (March 2025):
Swarm has been replaced by the OpenAI Agents SDK 
(https://github.com/openai/openai-agents-python), which now provides native API support for handoffs 
(https://openai.github.io/openai-agents-python/handoffs/) as an input to the new Agent Class 
(https://openai.github.io/openai-agents-python/agents/).

Roadmap:
Migrate to the OpenAI Agents API.
"""

import os
import pandas as pd
import datetime
import geopandas as gpd
from typing import Optional

from agent_core.modules.toolset import agent_tool

from agent_core.agents.assistant_agent import AssistantAgent

class Triage:
    def __init__(self) -> None:
        """
        Initialize with Triage object that implements "programmable" handoffs 
        """
        self.database_agent = None
        self.map_agent = None
        self.detector_agent = None
        self.name = "triage"

    @agent_tool
    def handoff_transfer_to_database_agent(self) -> AssistantAgent:
        """
        Transfer agentic procress to the database_agent responsible for database tasks.

        Returns:
            AssistantAgent: The database_agent object.
        """
        return self.database_agent

    @agent_tool
    def handoff_transfer_to_map_agent(self) -> AssistantAgent:
        """
        Transfer agentic procress to the map_agent responsible for map/plot tasks.

        Returns:
            AssistantAgent: The map_agent object.
        """
        return self.map_agent

    @agent_tool
    def handoff_transfer_to_detector_agent(self) -> AssistantAgent:
        """
        Transfer agentic procress to the detector_agent responsible for detection tasks.

        Returns:
            AssistantAgent: The detector_agent object.
        """
        return self.detector_agent