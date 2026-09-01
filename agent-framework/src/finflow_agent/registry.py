from typing import Any, Callable, Dict, List, Literal, Optional, Type
from pydantic import BaseModel, Field

class AgentSpec(BaseModel):
    name: str = Field(description="Unique identifier for the agent")
    description: str = Field(description="Plain text description for the LLM Orchestrator")
    stage: Literal["ingest", "transform", "analyze", "visualize", "deliver"]
    accepts: List[str] = Field(description="Required input data types (e.g., ['dataframe'])")
    produces: List[str] = Field(description="Output data types (e.g., ['filtered_dataframe'])")
    params_schema: Dict[str, Any] = Field(description="JSON Schema for expected parameters")
    # Whether the agent is enabled and may be executed by the engine. Disabled
    # agents (currently only ``visualization_agent`` when ``ENABLE_VISUALIZATION``
    # is false) remain registered so the validator's "Unknown agent: <name>"
    # error never fires for legitimately-named slots, but the validator and
    # compiler MUST refuse plans that reference a disabled agent. The flag is
    # mutated by ``bootstrap_agents`` based on the ``ENABLE_VISUALIZATION``
    # environment variable; agents themselves leave it at the ``True`` default
    # and never read environment configuration.
    enabled: bool = Field(
        default=True,
        description=(
            "Whether the agent is enabled. Disabled agents stay in the "
            "registry but must not be executed; the validator and compiler "
            "reject plans that reference them."
        ),
    )

class Registry:
    def __init__(self):
        self._agents: Dict[str, Type] = {}
        self._specs: Dict[str, AgentSpec] = {}
        # Maps agent name -> Pydantic params model. Populated when an agent class
        # declares a `params_model` class attribute. Consumed by the plan validator
        # and the execution engine to gate every step's `params` dict before the
        # agent is invoked.
        self._param_models: Dict[str, Type[BaseModel]] = {}

    def register(self, cls: Type) -> Type:
        if not hasattr(cls, 'spec') or not isinstance(cls.spec, AgentSpec):
            raise ValueError(f"Agent {cls.__name__} must define a 'spec' attribute of type AgentSpec")

        spec = cls.spec
        if spec.name in self._agents:
            raise ValueError(f"Agent name {spec.name} is already registered")

        self._agents[spec.name] = cls
        self._specs[spec.name] = spec

        # Optionally register a Pydantic params model alongside the agent class.
        # Agent classes opt in by declaring `params_model = SomeAgentParams`.
        params_model = getattr(cls, 'params_model', None)
        if params_model is not None:
            if not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
                raise ValueError(
                    f"Agent {cls.__name__}.params_model must be a Pydantic BaseModel subclass"
                )
            self._param_models[spec.name] = params_model

        return cls

    def describe_all(self) -> List[Dict[str, Any]]:
        return [spec.model_dump() for spec in self._specs.values()]

    def get_spec(self, name: str) -> AgentSpec:
        if name not in self._specs:
            raise ValueError(f"Agent {name} not found in registry")
        return self._specs[name]

    def get_agent_class(self, name: str) -> Type:
        if name not in self._agents:
            raise ValueError(f"Agent {name} not found in registry")
        return self._agents[name]

    def get_params_model(self, name: str) -> Type[BaseModel]:
        """Return the registered Pydantic params model for an agent.

        Raises ValueError if no params model has been registered for `name`.
        Callers that prefer a non-raising lookup should use `has_params_model`
        first or read `AGENT_PARAM_MODELS` directly.
        """
        if name not in self._param_models:
            raise ValueError(f"No params model registered for agent {name}")
        return self._param_models[name]

    def has_params_model(self, name: str) -> bool:
        return name in self._param_models

    def is_enabled(self, name: str) -> bool:
        """Return whether the named agent is currently enabled.

        Raises ``ValueError`` when ``name`` is not registered, mirroring
        ``get_spec``. The validator and compiler use this helper (added for
        the agent-pipeline-hardening spec, task 13.1 plumbing) to reject
        plans that reference a registered-but-disabled agent without having
        to reach into the spec model directly.
        """
        return self.get_spec(name).enabled

    @property
    def param_models(self) -> Dict[str, Type[BaseModel]]:
        """Read-only view of the agent-name -> params-model mapping."""
        return dict(self._param_models)

registry = Registry()

# Module-level alias so the validator and engine can do a direct lookup
# (e.g. `AGENT_PARAM_MODELS["ingestion_agent"]`) without going through the
# Registry instance. The Registry instance remains the source of truth; this
# proxy reflects whatever is currently registered.
class _AgentParamModelsProxy:
    """Read-only mapping proxy backed by the live `registry._param_models`."""

    def __getitem__(self, name: str) -> Type[BaseModel]:
        return registry.get_params_model(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and registry.has_params_model(name)

    def get(self, name: str, default: Optional[Type[BaseModel]] = None) -> Optional[Type[BaseModel]]:
        if registry.has_params_model(name):
            return registry.get_params_model(name)
        return default

    def keys(self):
        return registry.param_models.keys()

    def items(self):
        return registry.param_models.items()

    def values(self):
        return registry.param_models.values()

    def __iter__(self):
        return iter(registry.param_models)

    def __len__(self) -> int:
        return len(registry.param_models)

AGENT_PARAM_MODELS = _AgentParamModelsProxy()
