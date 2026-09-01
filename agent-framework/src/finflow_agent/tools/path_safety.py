import os
from pathlib import Path
from finflow_agent.operations.errors import UnsafeOutputPathError, UnsafeInputPathError

def get_safe_input_path(base_dir: str, file_path: str) -> Path:
    """
    Validate that ``file_path`` resolves inside ``base_dir`` and points to an
    existing regular file.

    Rejects:
      - empty / missing inputs
      - absolute paths whose resolved location is outside ``base_dir``
      - relative paths whose resolved location escapes ``base_dir`` (e.g. ".." segments)
      - paths that don't resolve to an existing regular file (missing or directory)

    Raises ``UnsafeInputPathError`` on rejection. Returns the resolved ``Path``
    on success.
    """
    if not base_dir:
        raise UnsafeInputPathError("base_dir must be provided.")
    if not file_path:
        raise UnsafeInputPathError("file_path must be provided.")

    base_path = Path(base_dir).resolve()

    # Treat relative paths as relative to the configured upload directory so a
    # caller passing "data.csv" maps to "<upload_dir>/data.csv". Absolute paths
    # are kept as-is and validated by the sandbox check below.
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = base_path / candidate

    try:
        candidate_resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise UnsafeInputPathError(
            f"Could not resolve input path '{file_path}': {exc}"
        ) from exc

    # Sandbox boundary: the resolved candidate must be a descendant of base_dir.
    try:
        candidate_resolved.relative_to(base_path)
    except ValueError:
        raise UnsafeInputPathError(
            f"Path traversal detected or path outside upload directory: {file_path}"
        )

    if not candidate_resolved.exists():
        raise UnsafeInputPathError(f"Input file does not exist: {file_path}")

    if not candidate_resolved.is_file():
        raise UnsafeInputPathError(f"Input path is not a regular file: {file_path}")

    return candidate_resolved


def get_safe_output_path(base_dir: str, file_name: str) -> Path:
    """
    Safely resolves a file path within a specific base directory.
    Prevents path traversal attacks.
    """
    if not base_dir:
        raise ValueError("base_dir must be provided.")
    if not file_name:
        raise ValueError("file_name must be provided.")
        
    base_path = Path(base_dir).resolve()
    
    safe_name = os.path.basename(file_name)
    if safe_name != file_name:
        raise UnsafeOutputPathError(f"Invalid characters or traversal attempts in file name: {file_name}")
        
    candidate_path = (base_path / safe_name).resolve()
    
    # Verify the candidate path is strictly within the base_dir
    try:
        candidate_path.relative_to(base_path)
    except ValueError:
        raise UnsafeOutputPathError(f"Path traversal detected or path outside base directory: {file_name}")
        
    # Ensure directory exists
    base_path.mkdir(parents=True, exist_ok=True)
    
    return candidate_path
