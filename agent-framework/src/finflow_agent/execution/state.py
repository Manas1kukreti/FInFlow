from typing import Any, Dict, List, Literal, Optional, Annotated
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    step_id: str
    agent: str
    params: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    input_from: List[str] = Field(default_factory=list)
    output_key: Optional[str] = None

class ExecutionPlan(BaseModel):
    steps: List[PlanStep]

def merge_data(left: dict, right: dict) -> dict:
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}

class PipelineState(BaseModel):
    """
    Shared state bag. Keyed by `step_id`, not agent name. 
    This prevents namespace collisions.
    """
    data: Annotated[Dict[str, Any], merge_data] = Field(default_factory=dict)

class AgentResult(BaseModel):
    """
    Standard envelope for all agent results.
    """
    status: Literal["success", "partial", "failed"]
    error_message: Optional[str] = None
    data: Any = None
    summary: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    operations_applied: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)

class ExecutionOutput(BaseModel):
    """
    Standard envelope for all deterministic executor operations.
    """
    data: Any = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    operations_applied: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
