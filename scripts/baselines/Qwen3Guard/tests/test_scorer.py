from __future__ import annotations

from src.backend import ScriptedBackend
from src.scorer import score_prepared_samples


class FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "|".join(str(value) for value in token_ids)


def test_scorer_uses_user_prompt_and_cumulative_prefix() -> None:
    seen: list[tuple[str, str]] = []

    def respond(prompt: str, prefix: str) -> str:
        seen.append((prompt, prefix))
        if prefix.endswith("4"):
            return (
                "Safety: Unsafe\n"
                "Categories: Violent\n"
                "Refusal: No"
            )
        return "Safety: Safe\nCategories: None\nRefusal: No"

    sample = {
        "sample_id": "a",
        "target_model": "model",
        "source_record": {"prompt": "original prompt"},
        "prompt_sha256": "hash",
        "target_token_ids": [1, 2, 3, 4],
        "total_target_tokens": 4,
        "checkpoint_ends": [2, 4],
    }
    rows = score_prepared_samples(
        [sample],
        tokenizer=FakeTokenizer(),
        backend=ScriptedBackend(respond),
        batch_size=2,
    )

    assert seen == [
        ("original prompt", "1|2"),
        ("original prompt", "1|2|3|4"),
    ]
    assert [row["label"] for row in rows] == ["safe", "unsafe"]
    assert rows[1]["categories"] == ["Violent"]
    assert rows[1]["refusal"] == "no"


def test_scorer_skips_completed_checkpoint() -> None:
    calls = 0

    def respond(prompt: str, prefix: str) -> str:
        del prompt, prefix
        nonlocal calls
        calls += 1
        return "Safety: Safe\nCategories: None\nRefusal: No"

    sample = {
        "sample_id": "a",
        "target_model": "model",
        "source_record": {"prompt": "prompt"},
        "prompt_sha256": "hash",
        "target_token_ids": [1, 2, 3, 4],
        "total_target_tokens": 4,
        "checkpoint_ends": [2, 4],
    }
    rows = score_prepared_samples(
        [sample],
        tokenizer=FakeTokenizer(),
        backend=ScriptedBackend(respond),
        batch_size=2,
        completed={("a", 2)},
    )

    assert calls == 1
    assert rows[0]["checkpoint_end"] == 4
