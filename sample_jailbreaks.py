# sample_jailbreaks_fixed.py
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

# 固定的攻击方法列表（你可以在这里写死 5 个方法名）
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
    读取每个攻击方法下、指定 LLM 的 json，筛选 label==1 的样本。
    返回按攻击方法分桶的候选列表。
    """
    root = _as_path(root)
    buckets: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ATTACK_METHODS}

    for attack in ATTACK_METHODS:
        if attack == "HumanJailbreaks":
            # 人工越狱单独存放在 root/human_jailbreaks/llm/results/llm.json
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

                # 从 behaviors_df 中查找行为
                try:
                    behavior = behaviors_df.loc[key, 'Behavior']
                    FunctionalCategory = behaviors_df.loc[key, 'FunctionalCategory']
                except KeyError:
                    print(f"[WARN] 在行为文件中未找到 BehaviorID '{key}'。")
                    continue # 跳过未找到的行为

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
    计算每个攻击方法应采样的数量：
    1) 初始等额平均分配（含余数，按回合分配给样本更充足的桶）
    2) 若某桶候选不足，取其上限，缺口再按“剩余可用量”大的桶分配
    """
    random.seed(seed)
    attacks = list(buckets.keys())
    k = len(attacks)
    base = n_total // k
    rem = n_total % k

    # 初次分配：每桶 base，余数 rem 轮流分配（优先候选更多者）
    # 排序仅用于分配余数，不影响后续再分配
    by_capacity = sorted(attacks, key=lambda a: len(buckets[a]), reverse=True)
    quota = {a: base for a in attacks}
    for a in by_capacity[:rem]:
        quota[a] += 1

    # 若不足，再分配
    while True:
        shortage = 0
        donors: List[Tuple[str, int]] = []  # (attack, 可再给出的数量)
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
            # 所有桶都已经到上限了，无法满足 n_total，结束
            break

        # 依据 spare 大小排序，从 spare 多的开始补
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

    # 加载 behaviors csv
    try:
        behaviors_df = pd.read_csv(behaviors_path)
        # 假设 CSV 文件有 'BehaviorID' 和 'Behavior' 列
        # 并且 json 文件中的键（如 'queer_lgbtq_harassment_text'）是 BehaviorID
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

    # 按每桶配额进行无放回随机抽样
    records: List[Dict[str, Any]] = []
    for attack, q in quota.items():
        if q <= 0 or len(buckets[attack]) == 0:
            continue
        # 为了可复现，先打乱，再取前 q
        items = buckets[attack][:]
        random.shuffle(items)
        records.extend(items[:q])

    # 若因极端不足导致凑不满 target_n（极少见），再全局补齐
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
    # 修改 root 和 llm 名称即可
    llm = "llama2_7b"

    # 定义 behaviors csv 文件路径
    # 用户应将 'harmbench_behaviors_text_all.csv' 文件放在此处
    behaviors_csv_path = "../../harmbench_results_initial_release/harmbench_results_initial_release/harmbench_behaviors_text_all.csv"

    sample_equal(
        llm=llm,  # 指定 LLM 名称
        root="../../harmbench_results_initial_release/harmbench_results_initial_release/results_text",  # 根目录路径
        behaviors_path=behaviors_csv_path,  # behaviors csv 的路径
        n=100,  # 采样数
        seed=42,  # 随机种子
        out=Path("data/raw/" + llm + "_sampled_jailbreaks.json"),
    )