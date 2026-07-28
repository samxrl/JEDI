from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


CAST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CAST_ROOT.parents[2]
VENDOR_ROOT = CAST_ROOT / "third_party" / "activation-steering"


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve relative paths from the JEDI repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_cast_path(value: str | Path) -> Path:
    """Reject any write path that escapes the CAST directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    root = CAST_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"CAST refuses to write outside the baseline directory: {resolved}; allowed root is {root}"
        )
    return resolved


def configure_local_environment(*, create: bool = True) -> Mapping[str, str]:
    """Confine caches, temporary files, and third-party logs to the CAST directory."""
    cache_root = require_cast_path(CAST_ROOT / "cache")
    paths = {
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TMPDIR": cache_root / "tmp",
        "TEMP": cache_root / "tmp",
        "TMP": cache_root / "tmp",
        "CAST_ACTIVATION_STEERING_LOG_DIR": cache_root / "activation_steering_logs",
    }
    if create:
        for path in set(paths.values()):
            require_cast_path(path).mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    # Transformers v5 will remove TRANSFORMERS_CACHE; HF_HOME already covers the model-cache root.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    # These environment variables are normally read only at interpreter startup.
    # Update runtime state as well so later dynamic imports of JEDI evaluation modules
    # do not create caches or temporary files outside the CAST directory.
    sys.pycache_prefix = str(paths["PYTHONPYCACHEPREFIX"])
    tempfile.tempdir = str(paths["TMPDIR"])
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return {name: str(path) for name, path in paths.items()}


def ensure_vendor_on_path() -> None:
    """Prioritize the pinned official source over a same-named package installed in the environment."""
    import sys

    vendor = str(VENDOR_ROOT)
    if vendor in sys.path:
        sys.path.remove(vendor)
    sys.path.insert(0, vendor)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256_text(encoded)


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"The top level of YAML must be a mapping: {resolved}")
    return value


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = require_cast_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    # Path.write_text lacks the newline parameter on older Python versions; Path.open
    # has supported it longer and remains compatible with Python 3.8/3.9 common on Linux clusters.
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(target)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    return atomic_write_text(path, text)


def write_yaml(path: str | Path, payload: Any) -> Path:
    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return atomic_write_text(path, text)


def write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows
    )
    return atomic_write_text(path, text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object.")
            rows.append(value)
    return rows
