from pydantic import BaseModel, Field
from typing import List, Optional, Literal
 
class ModelClientConfig(BaseModel):
    model: str = Field(..., description="The specific LLM model name, e.g., 'gpt-4o-mini'")

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {"model": "gpt-4o-mini"}
        }

class ModelClient(BaseModel):
    provider: Literal["autogen_ext.models.openai.OpenAIChatCompletionClient"]
    component_type: Literal["model"]
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
    tools: List[Literal['plot_images_scatter_map', 'plot_detections_catergory_scatter_map', 'plot_land_cover_class_scatter_map', 'reset_map', 'zoom_map',\
                        'query_dataset_images', 'query_images_by_aoi_coords', 'count_filtered_images', 'count_detections_by_category', 'count_lcc_classification_results_by_category',\
                        'run_detector', 'run_land_coverage_classifier']] = \
        Field(..., description="List of callable tool names assigned to this agent.")
    handoffs: List[str] = Field(..., description="List of downstream agent names indicating agents to hand off to.")
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
    provider: Literal["autogen_agentchat.agents.AssistantAgent"]
    component_type: Literal["agent"]
    version: int = Field(..., description="Spec version for this component.")
    component_version: int = Field(..., description="Implementation version for the agent.")
    description: str = Field(..., description="Human-readable summary of what this agent does.")
    label: Literal['DatabaseAgent', 'VisionAgent', 'MapAgent', 'DataAgent'] = \
        Field(..., description="Short name or label for this agent.")
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
    provider: Literal["autogen_agentchat.teams.RoundRobinGroupChat"]
    component_type: Literal["team"]
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