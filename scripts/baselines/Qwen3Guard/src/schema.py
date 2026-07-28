from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GuardDecision:
    """Qwen3Guard three-level severity and the policy-derived release decision."""

    raw_output: str
    is_safe: bool
    verdict: str
    categories: Sequence[str]
    parse_status: str
    refusal: str | None = None
    risk_probability: float | None = None
    category_probability: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class GuardBatchResult:
    decisions: Sequence[GuardDecision]
    generate_ms: float
    e2e_ms: float
    batch_size: int
    retry_count: int = 0
