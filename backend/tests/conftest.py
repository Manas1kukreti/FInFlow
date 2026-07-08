from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT_STR = str(BACKEND_ROOT)
BACKEND_TEMP_ROOT = BACKEND_ROOT / ".pytest-tmp"
BACKEND_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

if BACKEND_ROOT_STR not in sys.path:
    sys.path.insert(0, BACKEND_ROOT_STR)

for name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[name] = str(BACKEND_TEMP_ROOT)

tempfile.tempdir = str(BACKEND_TEMP_ROOT)
