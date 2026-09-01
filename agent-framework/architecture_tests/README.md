# FinFlow Architecture Test Bundle

Copy the `tests/` folder into the root of your FinFlow Agent Service project.

Run:

```bash
pip install pytest pytest-asyncio pandas openpyxl xlsxwriter
pytest -q
```

These tests are intentionally architecture-focused. Some tests may fail on the current code because they are designed to catch the remaining issues:

- LLM must not output direct `steps`
- ARQ worker must bootstrap agents
- API must preserve engine failure details
- compiler must reject incomplete `PlanIntent`
- plan validator must validate `input_from`
- engine must pass visualization artifacts to reporting
- ingestion must reject legacy `file_path`
- output module must stay deprecated
- FileStore must reject traversal
- dataframe profiling must not include samples by default

If a test fails, treat it as a useful architecture signal, not just a unit-test annoyance.
