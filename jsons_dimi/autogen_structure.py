from pydantic import BaseModel, Field
from typing import List, Optional


class ModelClientConfig(BaseModel):
    model: str = Field(..., description="The specific LLM model name, e.g., 'gpt-4o-mini'")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {"model": "gpt-4o-mini"}
        }


class ModelClient(BaseModel):
    provider: str = Field(..., description="Path or name of the LLM provider implementation.")
    component_type: str = Field(..., description="Component type, e.g., 'model'.")
    version: int = Field(..., description="Component spec version.")
    component_version: int = Field(..., description="Implementation version of the component.")
    description: str = Field(..., description="Human-readable description of the model client.")
    label: str = Field(..., description="Short name of the model client.")
    config: ModelClientConfig = Field(..., description="Configuration block with model name.")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "provider": "llm_clients.openai_client.OpenAIClient",
                "component_type": "model",
                "version": 1,
                "component_version": 1,
                "description": "OpenAI LLM client",
                "label": "OpenAIClient",
                "config": {"model": "gpt-4o-mini"}
            }
        }


class AgentConfig(BaseModel):
    name: str = Field(..., description="Unique internal name for the agent.")
    model_client: ModelClient = Field(..., description="Model client configuration.")
    tools: List[str] = Field(..., description="List of callable tool names assigned to this agent.")
    handoffs: List[str] = Field(..., description="List of downstream agents to hand off to.")
    description: str = Field(..., description="Agent-specific description of its task.")
    system_message: Optional[str] = Field(None, description="Optional system message for the agent.")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "name": "vision_agent",
                "model_client": ModelClient.Config.json_schema_extra["example"],
                "tools": ["run_detector", "run_land_coverage_classifier"],
                "handoffs": ["map_agent"],
                "description": "Processes fetched images and runs the Swin-L object detection model.",
                "system_message": "You will analyze the fetched images with the Swin-L detector."
            }
        }


class Agent(BaseModel):
    provider: str = Field(..., description="Full path or identifier of the agent provider.")
    component_type: str = Field(..., description="Type of this component (e.g., 'Agent').")
    version: int = Field(..., description="Spec version for this component.")
    component_version: int = Field(..., description="Implementation version for the agent.")
    description: str = Field(..., description="Human-readable summary of what this agent does.")
    label: str = Field(..., description="Short name or label for this agent.")
    config: AgentConfig = Field(..., description="Configuration settings for this agent.")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "provider": "geoapps.geeo.agents.VisionAgent",
                "component_type": "Agent",
                "version": 1,
                "component_version": 1,
                "description": "Agent to run detector vision models on satellite images.",
                "label": "VisionAgent",
                "config": AgentConfig.Config.json_schema_extra["example"]
            }
        }


class WorkflowConfig(BaseModel):
    participants: List[Agent] = Field(..., description="List of all agents in the workflow.")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "participants": [
                    Agent.Config.json_schema_extra["example"]
                ]
            }
        }


class AutogenWorkflow(BaseModel):
    provider: str = Field(..., description="The provider of this workflow, e.g., 'autogen'.")
    component_type: str = Field(..., description="The component type, should be 'Workflow'.")
    version: int = Field(..., description="The spec version of this workflow.")
    component_version: int = Field(..., description="The implementation version of this workflow.")
    description: str = Field(..., description="Description of the full workflow and its intent.")
    label: str = Field(..., description="Label used to identify this workflow.")
    config: WorkflowConfig = Field(..., description="The agent composition used in the workflow.")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "provider": "autogen",
                "component_type": "Workflow",
                "version": 1,
                "component_version": 1,
                "description": "Agentic workflow to fetch satellite images, run a detector, and zoom the map based on user queries.",
                "label": "Geospatial Satellite Imagery Workflow",
                "config": WorkflowConfig.Config.json_schema_extra["example"]
            }
        }