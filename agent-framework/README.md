# FinFlow Agent Framework

The `finflow_agent` package implements the FinFlow agent pipeline:

```
natural-language intent
  → semantic grounding
  → ambiguity detection
  → clarification
  → deterministic execution plan / DAG
  → worker execution
  → results
```

## Layout

```
agent-framework/
├── src/finflow_agent/     # the installable package
├── tests/                 # unit tests
├── architecture_tests/    # architecture / behavior tests
├── Dockerfile             # agent service image
├── requirements.txt       # pinned runtime dependencies
└── pyproject.toml         # packaging + editable install config
```

## Installation

Install the package (editable) from the repository root:

```bash
pip install -e ./agent-framework
```

This makes `finflow_agent` importable without any `sys.path` manipulation:

```python
from finflow_agent.engine import ExecutionEngine
```

## Running tests

```bash
pip install -e "./agent-framework[test]"
pytest agent-framework
```
