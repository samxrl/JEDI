from __future__ import annotations

import sys
from pathlib import Path


QWEN3GUARD_ROOT = Path(__file__).resolve().parents[1]
if str(QWEN3GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3GUARD_ROOT))
