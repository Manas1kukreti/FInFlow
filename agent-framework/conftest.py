from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
TEMP_ROOT = PROJECT_ROOT / ".pytest-tmp-local"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)

# Prefer an editable/regular install of the ``finflow_agent`` package
# (``pip install -e ./agent-framework``). Fall back to the local ``src``
# directory so the tests remain runnable from a fresh checkout without an
# install step.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

for name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[name] = str(TEMP_ROOT)

tempfile.tempdir = str(TEMP_ROOT)
