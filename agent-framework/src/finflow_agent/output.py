# DEPRECATED: This module is deprecated.
# Use ReportingAgent -> execute_reporting_plan -> reporting_handlers instead.

def generate_output(*args, **kwargs):
    raise RuntimeError(
        "generate_output is deprecated. Use ReportingAgent and execute_reporting_plan instead."
    )
