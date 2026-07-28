from __future__ import annotations

import sys
from pathlib import Path


CAST_ROOT = Path(__file__).resolve().parents[1]
if str(CAST_ROOT) not in sys.path:
    sys.path.insert(0, str(CAST_ROOT))

from src.paths import configure_local_environment, ensure_vendor_on_path

configure_local_environment()
ensure_vendor_on_path()

