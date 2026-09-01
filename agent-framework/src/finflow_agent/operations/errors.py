class FinFlowError(Exception):
    """Base exception for all FinFlow agent operations."""
    pass

class OperationExecutionError(FinFlowError):
    """Raised when an operation fails during execution."""
    pass

class UnsupportedOperationError(FinFlowError):
    """Raised when an unsupported operation type is encountered."""
    pass

class OperationValidationError(FinFlowError):
    """Raised when an operation plan fails structural or logical validation."""
    pass

class UnsafeOutputPathError(FinFlowError):
    """Raised when an output path is potentially unsafe (e.g. path traversal)."""
    pass

class ReportGenerationError(FinFlowError):
    """Raised when report generation fails."""
    pass

class UnsafeInputPathError(FinFlowError):
    """Raised when an input path is potentially unsafe (e.g. path traversal)."""
    pass

class UnsafeFilterPrepOperationError(FinFlowError):
    """Raised when an operation outside the SAFE_FILTER_PREP_OPERATIONS whitelist
    is requested in `cleaning_agent` `filter_prep` mode.

    The Compiler emits a `filter_prep` step (Component 7) realized as a
    `cleaning_agent` invocation whose params restrict execution to the seven
    safe, non-destructive operations defined in requirement 2.7. Any attempt
    to execute an operation outside that whitelist must be rejected before it
    can mutate the dataframe; this error is the controlled signal used by the
    `assert_safe_for_filter_prep` guard in `operations.cleaning_handlers`.
    """
    pass
