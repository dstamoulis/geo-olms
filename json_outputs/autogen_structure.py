from pydantic import BaseModel

# pydantic output formatting stuff
class ModelClientConfig(BaseModel):
    model: str

class ModelClient(BaseModel):
    provider: str
    component_type: str
    version: int
    component_version: int
    description: str
    label: str
    config: ModelClientConfig

class ModelContext(BaseModel):
    provider: str
    component_type: str
    version: int
    component_version: int
    description: str
    label: str
    config: ModelClientConfig

class AgentConfig(BaseModel):
    name: str
    model_client: ModelClient
    tools: list[str]  
    handoffs: list[str]
    # model_context: ModelContext
    description: str
    # system_message: str
    # model_client_stream: bool
    # reflect_on_tool_use: bool
    # tool_call_summary_format: str

class Agent(BaseModel):
    provider: str
    component_type: str
    version: int
    component_version: int
    description: str
    label: str
    config: AgentConfig

class TerminationConditionConfig(BaseModel):
    text: str

class TerminationCondition(BaseModel):
    provider: str
    component_type: str
    version: int
    component_version: int
    description: str
    label: str
    config: TerminationConditionConfig

class WorkflowConfig(BaseModel):
    participants: list[Agent]
    # termination_condition: TerminationCondition

class AutogenWorkflow(BaseModel):
    provider: str
    component_type: str
    version: int
    component_version: int
    description: str
    label: str
    config: WorkflowConfig
