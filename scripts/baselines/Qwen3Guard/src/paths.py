from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


QWEN3GUARD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = QWEN3GUARD_ROOT.parents[2]


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve relative paths from the JEDI repository root; use only for read-only inputs."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_local_write(value: str | Path) -> Path:
    """Reject any write path that escapes ``scripts/baselines/Qwen3Guard``."""
    resolved = Path(value).expanduser().resolve()
    root = QWEN3GUARD_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"Qwen3Guard refuses to write outside the baseline directory: {resolved}; "
            f"allowed root is {root}"
        )
    return resolved


def resolve_local_path(value: str | Path) -> Path:
    """Resolve relative paths from this baseline's root and enforce write isolation."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = QWEN3GUARD_ROOT / path
    return require_local_write(path)


def configure_local_environment(*, create: bool = True) -> Mapping[str, str]:
    """Confine caches and temporary directories to this baseline before importing torch/transformers."""
    cache_root = require_local_write(QWEN3GUARD_ROOT / "cache")
    paths = {
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
        "TORCH_HOME": cache_root / "torch",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TMPDIR": cache_root / "tmp",
        "TEMP": cache_root / "tmp",
        "TMP": cache_root / "tmp",
    }
    if create:
        for path in set(paths.values()):
            require_local_write(path).mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.pycache_prefix = str(paths["PYTHONPYCACHEPREFIX"])
    tempfile.tempdir = str(paths["TMPDIR"])
    return {name: str(path) for name, path in paths.items()}


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
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"The top level of YAML must be a mapping: {resolved}")
    return value


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = require_local_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_local_write(target.with_suffix(target.suffix + ".tmp"))
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(target)
    return target


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Append UTF-8 JSONL for resumable prefix-scoring logs."""
    target = require_local_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, default=str) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    return target


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


def write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def write_yaml(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
    )


def write_jsonl_gz(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write UTF-8 JSONL with a fixed gzip mtime."""
    target = require_local_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_local_write(target.with_suffix(target.suffix + ".tmp"))
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(
                        json.dumps(dict(row), ensure_ascii=False, default=str)
                        + "\n"
                    )
    temporary.replace(target)
    return target


def read_jsonl_gz(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(Path(path), mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object.")
            rows.append(value)
    return rows
