# sample_jailbreaks_fixed.py
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

# Fixed list of attack methods (you can hardcode 5 method names here).
ATTACK_METHODS = [
    "GCG",
    "AutoDAN",
    "PAIR",
    "TAP",
    "HumanJailbreaks",
]


def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def load_by_attack(root, llm, behaviors_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read per-attack JSON for the specified LLM and filter samples with label==1.
    Return candidate lists bucketed by attack method.
    """
    root = _as_path(root)
    buckets: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ATTACK_METHODS}

    for attack in ATTACK_METHODS:
        if attack == "HumanJailbreaks":
            # Human jailbreaks are stored at root/human_jailbreaks/llm/results/llm.json.
            json_path = root / attack / "default" / "results" / f"{llm}.json"
        else:
            json_path = root / attack / llm / "results" / f"{llm}.json"
        if not json_path.exists():
            print(f"[WARN] 文件不存在：{json_path}")
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取失败：{json_path}，错误：{e}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] 非预期结构（顶层不是 dict）：{json_path}")
            continue

        for key, items in data.items():
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                if not isinstance(item, dict) or item.get("label", 0) != 1:
                    continue
                prompt_text = item.get("test_case") or item.get("prompt")
                if not prompt_text:
                    continue

                # Look up the behavior in behaviors_df.
                try:
                    behavior = behaviors_df.loc[key, 'Behavior']
                    FunctionalCategory = behaviors_df.loc[key, 'FunctionalCategory']
                except KeyError:
                    print(f"[WARN] 在行为文件中未找到 BehaviorID '{key}'。")
                    continue  # Skip behaviors that are not found.

                buckets[attack].append({
                    "behavior": behavior,
                    "prompt": prompt_text,
                    "FunctionalCategory": FunctionalCategory,
                    "ContextString": behaviors_df.loc[key, 'ContextString'] if FunctionalCategory == 'contextual' else "",
                    "source": {
                        "attack": attack,
                        "file": str(json_path),
                        "category_key": key,
                    },
                })
    return buckets


def _plan_quota(buckets: Dict[str, List[Dict[str, Any]]], n_total: int, seed: int) -> Dict[str, int]:
    """
    Compute the sample count for each attack method:
    1) Start with equal allocation (including remainder, distributed to larger buckets)
    2) If a bucket lacks candidates, cap it and reassign the deficit to buckets with spare
    """
    random.seed(seed)
    attacks = list(buckets.keys())
    k = len(attacks)
    base = n_total // k
    rem = n_total % k

    # Initial allocation: base per bucket, remainder to larger buckets.
    # Sorting only affects remainder allocation and not later redistribution.
    by_capacity = sorted(attacks, key=lambda a: len(buckets[a]), reverse=True)
    quota = {a: base for a in attacks}
    for a in by_capacity[:rem]:
        quota[a] += 1

    # If there is a shortage, redistribute.
    while True:
        shortage = 0
        donors: List[Tuple[str, int]] = []  # (attack, additional available count)
        for a in attacks:
            cap = len(buckets[a])
            if quota[a] > cap:
                shortage += quota[a] - cap
                quota[a] = cap
        if shortage == 0:
            break

        for a in attacks:
            cap = len(buckets[a])
            spare = max(0, cap - quota[a])
            if spare > 0:
                donors.append((a, spare))

        if not donors:
            # All buckets are at capacity and cannot satisfy n_total.
            break

        # Sort by spare capacity and fill from the largest.
        donors.sort(key=lambda x: x[1], reverse=True)
        i = 0
        while shortage > 0 and donors:
            a, spare = donors[i % len(donors)]
            if spare > 0:
                quota[a] += 1
                spare -= 1
                donors[i % len(donors)] = (a, spare)
                shortage -= 1
            i += 1

    return quota


def sample_equal(llm: str, root: str, behaviors_path: str, n: int = 100, seed: int = 42, out=None):
    root = _as_path(root)
    out = _as_path(out) if out else Path(f"{llm}_sampled_equal.json")
    random.seed(seed)

    # Load behaviors CSV.
    try:
        behaviors_df = pd.read_csv(behaviors_path)
        # Assume the CSV has 'BehaviorID' and 'Behavior' columns.
        # JSON keys (e.g., 'queer_lgbtq_harassment_text') are BehaviorID values.
        behaviors_df.set_index('BehaviorID', inplace=True)
    except FileNotFoundError:
        print(f"[ERROR] 在路径中未找到行为文件： {behaviors_path}")
        return
    except KeyError:
        print(f"[ERROR] 行为文件必须包含 'BehaviorID' 列。")
        return

    buckets = load_by_attack(root, llm, behaviors_df)
    total_candidates = sum(len(v) for v in buckets.values())
    if total_candidates == 0:
        raise RuntimeError("未找到任何 label==1 的越狱提示，请检查路径与数据。")

    target_n = min(n, total_candidates)
    quota = _plan_quota(buckets, target_n, seed)

    # Sample without replacement based on per-bucket quotas.
    records: List[Dict[str, Any]] = []
    for attack, q in quota.items():
        if q <= 0 or len(buckets[attack]) == 0:
            continue
        # Shuffle first for reproducibility, then take the first q.
        items = buckets[attack][:]
        random.shuffle(items)
        records.extend(items[:q])

    # If extreme shortages prevent reaching target_n (rare), fill globally.
    if len(records) < target_n:
        remaining = []
        for attack, items in buckets.items():
            used = quota.get(attack, 0)
            remaining.extend(items[used:])
        random.shuffle(remaining)
        records.extend(remaining[: target_n - len(records)])

    output = {
        "llm": llm,
        "requested_sample_size": n,
        "actual_sample_size": len(records),
        "total_candidates": total_candidates,
        "attacks": ATTACK_METHODS,
        "seed": seed,
        "quota_per_attack": quota,
        "records": records,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"完成：总候选 {total_candidates}，按等量策略采样 {len(records)} 条 → {out}")
    print("每个攻击方法分配：", {k: v for k, v in quota.items() if v > 0})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample jailbreak prompts by attack method.")
    parser.add_argument("--llm", required=True, help="LLM name used to locate result files.")
    parser.add_argument(
        "--behaviors-csv",
        default="../../harmbench_results_initial_release/harmbench_results_initial_release/harmbench_behaviors_text_all.csv",
        help="Path to harmbench behaviors CSV.",
    )
    parser.add_argument(
        "--root",
        default="../../harmbench_results_initial_release/harmbench_results_initial_release/results_text",
        help="Root directory for attack results.",
    )
    parser.add_argument("--n", type=int, default=100, help="Sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Defaults to data/raw/<llm>_sampled_jailbreaks.json.",
    )
    args = parser.parse_args()

    output_path = Path(args.out) if args.out else Path("data/raw") / f"{args.llm}_sampled_jailbreaks.json"

    sample_equal(
        llm=args.llm,  # LLM name.
        root=args.root,  # Root directory path.
        behaviors_path=args.behaviors_csv,  # Path to behaviors CSV.
        n=args.n,  # Sample size.
        seed=args.seed,  # Random seed.
        out=output_path,
    )
