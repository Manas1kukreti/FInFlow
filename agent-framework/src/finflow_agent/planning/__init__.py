"""Planning package — compilation of intents into execution plans.

Exports both legacy compilation (compile_intent_to_plan) and the refactored
Compiler class from the semantic-grounding-refactor spec.
"""

# Lazy imports to avoid circular dependency chains in the existing codebase.
# Use: from finflow_agent.planning import Compiler
# Or:  from finflow_agent.planning.compiler import Compiler


def __getattr__(name: str):
    """Lazy module-level attribute access for planning package exports."""
    _compiler_exports = {
        "CompilerError",
        "ExecutionStep",
        "RefactoredExecutionPlan",
        "Compiler",
    }
    _trigger_exports = {
        "TriggerDetector",
        "TriggerResult",
    }
    _enricher_exports = {
        "enrich_intent_with_visualization",
        "should_produce_visualization",
    }
    if name in _compiler_exports:
        from finflow_agent.planning.compiler import (
            CompilerError,
            ExecutionStep,
            RefactoredExecutionPlan,
            Compiler,
        )
        _mapping = {
            "CompilerError": CompilerError,
            "ExecutionStep": ExecutionStep,
            "RefactoredExecutionPlan": RefactoredExecutionPlan,
            "Compiler": Compiler,
        }
        return _mapping[name]
    if name in _trigger_exports:
        from finflow_agent.planning.trigger_detector import (
            TriggerDetector,
            TriggerResult,
        )
        _mapping = {
            "TriggerDetector": TriggerDetector,
            "TriggerResult": TriggerResult,
        }
        return _mapping[name]
    if name in _enricher_exports:
        from finflow_agent.planning.intent_enricher import (
            enrich_intent_with_visualization,
            should_produce_visualization,
        )
        _mapping = {
            "enrich_intent_with_visualization": enrich_intent_with_visualization,
            "should_produce_visualization": should_produce_visualization,
        }
        return _mapping[name]
    raise AttributeError(f"module 'finflow_agent.planning' has no attribute {name!r}")


__all__ = [
    "CompilerError",
    "ExecutionStep",
    "RefactoredExecutionPlan",
    "Compiler",
    "TriggerDetector",
    "TriggerResult",
    "enrich_intent_with_visualization",
    "should_produce_visualization",
]
