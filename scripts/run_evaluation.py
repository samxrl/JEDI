# -*- coding: utf-8 -*-
"""Script 05: Run defense assessment (supports jbb_expanded.csv, alpaca_eval.json and xstest_prompts.csv)

This script is the final step in the JEDI process and verifies `04_calibrate_defense.py`
Actual effectiveness of calibrated defense systems.

** This version has been modified based on memory optimization requests **
The process is divided into three stages to ensure that the LLM under test and the classifier LLM do not occupy video memory at the same time:
1. **Phase 1 (Build)**: Load the LLM under test, run all builds (baseline + guarded),
    Save the results and then release the LLM under test.
2. **Phase 2 (Classification)**: Load the classifier LLM and judge all generated results.
    Save the labels and release the classifier LLM.
3. **Phase 3 (Reporting)**: All metrics are calculated and saved to file.

** This version has been modified to support loading and evaluating multiple benign datasets. **
1. `load_utility_dataset` now loads a list of datasets from the configuration, supports CSV and JSON.
2. Implemented quota-based equal sampling logic.
3. The evaluation and preservation phases now generate separate reports for each benign dataset.
4. **New**: Special support for `alpaca_eval` format output.
5. **New**: Special support for `xstest` (xstest_prompts.csv) format output (8 columns CSV).

** This version has been modified to support configuration control over whether usability/security testing is run. **
1. Added `run_utility_evaluation` and `run_safety_evaluation` configuration items.
2. Conditionally load data sets and perform evaluations based on configuration items.

** Modification: Added time-consuming statistics when testing the utility data set. **"""

import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import gc
import sys
import json
import contextlib
import time  # New: Import time module for timing
from typing import Dict, List, Any, Optional, Tuple
import importlib.util

# ---Path Setup ---
base_dir = Path(__file__).parent.parent
src_path = base_dir / "src"
scripts_path = base_dir / "scripts"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

try:
    from JEDI_guard import Guard
except ImportError:
    print(
        f"ERROR: Unable to import JEDI_guard. Please make sure the 'src' directory is in sys.path:{src_path}"
    )
    sys.exit(1)

    # --- Dynamically import LLAMA2_CLS_PROMPT ---
try:
    judge_script_path = scripts_path / "02.5_judge_harmfulness.py"
    spec = importlib.util.spec_from_file_location("judge_script", judge_script_path)
    judge_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_module)
    LLAMA2_CLS_PROMPT = judge_module.LLAMA2_CLS_PROMPT
    if not LLAMA2_CLS_PROMPT:
        raise ImportError("LLAMA2_CLS_PROMPT is empty.")
except Exception as e:
    print(
        f"ERROR: Unable to dynamically import LLAMA2_CLS_PROMPT from 02.5_judge_harmfulness.py:{e}"
    )
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str, model_kwargs: dict, device: str) -> tuple:
    """Load the Hugging Face model and tokenizer."""
    logger.info(f"Loading model:{model_name}...")
    kwargs = model_kwargs.copy()
    if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
        try:
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        except AttributeError:
            if kwargs["torch_dtype"] != "auto":
                raise ValueError(f"Invalid torch_dtype:{kwargs['torch_dtype']}")

    if "device_map" not in kwargs and device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if "device_map" not in kwargs and device == "cuda":
        model.to(device)

    model.eval()
    logger.info(f"Model{model_name}Loading completed.")
    return model, tokenizer


def load_utility_dataset(
    config: dict, data_dir: Path, total_sample_size: int = 0
) -> pd.DataFrame:
    """[!! Modified !!]
    Load one or more "usability" (benign) assessment datasets from a CSV or JSON file.
    Supports quota-based equal sampling.
    Support alpaca_eval.json format."""
    logger.info("Loading Utility data set...")
    dataset_list = config.get("datasets")
    if not dataset_list or not isinstance(dataset_list, list):
        logger.warning(
            "'utility_dataset_config.datasets' list not found in configuration. Returns an empty data frame."
        )
        return pd.DataFrame()

    all_loaded_dfs = []
    for dataset_config in dataset_list:
        name = dataset_config.get("name")
        filename = dataset_config.get("filename")
        # prompt_column may be instruction for json or column name for csv
        prompt_col = dataset_config.get("prompt_column", "prompt")
        max_new_tokens = dataset_config.get("max_new_tokens")

        if not name or not filename:
            logger.warning(
                f"Skip an invalid benign dataset entry (missing name or filename):{dataset_config}"
            )
            continue

        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning(
                f"Availability dataset file not found:{file_path}. jump over."
            )
            continue

        try:
            # New: support for JSON format (specifically alpaca_eval)
            if filename.lower().endswith(".json"):
                logger.info(f"JSON file detected:{filename}, loading...")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)

                # In case of alpaca_eval common format, instruction is actually prompt
                if "instruction" in df.columns and "prompt" not in df.columns:
                    logger.info("'instruction' column detected, mapped to 'prompt'.")
                    df["prompt"] = df["instruction"]
                elif prompt_col in df.columns and prompt_col != "prompt":
                    df.rename(columns={prompt_col: "prompt"}, inplace=True)

                    # Make sure the fields required by alpaca_eval exist (if the original file has them, pd.DataFrame will automatically retain them)
                if "dataset" not in df.columns:
                    df["dataset"] = (
                        name  # Use configuration name as default value for dataset field
                    )

            else:
                # CSV format processing (including XSTest, OR-Bench, etc.)
                # Note: pd.read_csv will retain all columns, including xstest's 'id' and 'type'
                df = pd.read_csv(file_path)
                if prompt_col not in df.columns:
                    logger.error(
                        f"Availability dataset{file_path}Missing column '{prompt_col}'. jump over."
                    )
                    continue
                df.rename(columns={prompt_col: "prompt"}, inplace=True)

                # Key: Add dataset name
            df["utility_dataset_name"] = name
            df["max_new_tokens"] = max_new_tokens

            # Populate placeholders for benign datasets
            if "behavior" not in df.columns:
                if "Goal" in df.columns:
                    df["behavior"] = df["Goal"]
                elif "Behavior" in df.columns:
                    df["behavior"] = df["Behavior"]
                else:
                    df["behavior"] = "N/A"
            if "FunctionalCategory" not in df.columns:
                df["FunctionalCategory"] = "N/A"
            if "ContextString" not in df.columns:
                df["ContextString"] = ""
            df["attack_method"] = "N/A"  # Placeholder

            all_loaded_dfs.append(df)
            logger.info(
                f"Already from{file_path}load{len(df)}samples (labeled '{name}')。"
            )

        except Exception as e:
            logger.error(
                f"Process files{file_path}An error occurred:{e}", exc_info=True
            )

    if not all_loaded_dfs:
        logger.warning("No availability data sets were loaded successfully.")
        return pd.DataFrame()

        # --- Sampling logic ---
    if total_sample_size <= 0:
        logger.info(
            f"sample_size <= 0, merging all{len(all_loaded_dfs)}datasets (total{sum(len(df) for df in all_loaded_dfs)}samples)."
        )
        return pd.concat(all_loaded_dfs, ignore_index=True)

    logger.info(
        f"Removing from{len(all_loaded_dfs)}Equally sampled data sets are sampled, with a total target{total_sample_size}samples..."
    )

    num_datasets = len(all_loaded_dfs)
    base_quota = total_sample_size // num_datasets
    remainder = total_sample_size % num_datasets
    # base quota and distribute the remainder to the first few data sets
    quotas = [
        base_quota + 1 if i < remainder else base_quota for i in range(num_datasets)
    ]

    sampled_dfs = []
    total_shortfall = 0
    donors = []  # (index, extra_capacity)

    # Pass 1: Try to meet quota
    logger.debug(f"Initial quota:{quotas}")
    for i, df in enumerate(all_loaded_dfs):
        quota = quotas[i]
        if len(df) >= quota:
            sampled_dfs.append(df.sample(n=quota, random_state=42))
            extra = len(df) - quota
            if extra > 0:
                donors.append((i, extra))
        else:  # Not enough samples
            sampled_dfs.append(df)  # All samples
            total_shortfall += quota - len(df)
            logger.info(
                f"Data set{i}Insufficient sample (requires{quota}, only{len(df)}). gap:{quota - len(df)}"
            )

            # Second pass: Redistribute gaps
    if total_shortfall > 0 and donors:
        logger.info(
            f"Appear{total_shortfall}The gaps in samples are being filled in from other data sets..."
        )

        # Summarize all available additional samples
        all_extra_samples = []
        for donor_idx, extra_cap in donors:
            donor_df = all_loaded_dfs[donor_idx]
            # Find the unsampled index
            sampled_indices = sampled_dfs[donor_idx].index
            available_indices = donor_df.index.difference(sampled_indices)
            # Add extra rows where available
            all_extra_samples.append(donor_df.loc[available_indices])

        if all_extra_samples:
            # Merge all available additional samples
            extra_df = pd.concat(all_extra_samples, ignore_index=True)
            # Randomly sample from this pool to bridge the gap
            take_n = min(total_shortfall, len(extra_df))
            if take_n > 0:
                final_extra_samples = extra_df.sample(n=take_n, random_state=42)
                sampled_dfs.append(final_extra_samples)
                logger.info(f"successfully added{take_n}additional samples.")

    final_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
    logger.info(f"Loaded and sampled successfully{len(final_df)}usability sample.")
    return final_df


def load_safety_dataset(
    config: dict, data_dir: Path, sample_size: int = 0
) -> pd.DataFrame:
    """Load and convert the "Security" (JBB) assessment data set (from wide table to long table)."""
    # Modification: Pre-check attack column configuration
    # If attack_columns_to_eval is None or empty, skip loading directly to prevent subsequent errors.
    attack_cols = config.get("attack_columns_to_eval")
    if not attack_cols:
        logger.info(
            "'attack_columns_to_eval' in the configuration is empty or None to skip security data set loading."
        )
        return pd.DataFrame()

    file_path = data_dir / config["filename"]
    base_cols = config["base_columns"]
    max_new_tokens = config.get("max_new_tokens")
    # attack_cols = config['attack_columns_to_eval'] # Obtained above

    if not file_path.exists():
        logger.warning(
            f"Security data set file not found:{file_path}. Returns an empty data frame."
        )
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Verify that all required columns are present
    missing_cols = [c for c in base_cols + attack_cols if c not in df.columns]
    if missing_cols:
        logger.error(
            f"security data set{file_path}The following required columns are missing:{missing_cols}"
        )
        return pd.DataFrame()

        # Convert wide table to long table using pd.melt
        # id_vars = basic information column
        # value_vars = columns of attack methods to evaluate
    df_long = pd.melt(
        df,
        id_vars=base_cols,
        value_vars=attack_cols,
        var_name="attack_method",  # New column: stores attack method names (e.g. "GCG", "PAIR")
        value_name="prompt",  # New column: stores the jailbreak tips corresponding to this method
    )

    # Clean: Remove rows with empty prompts (if there are empty cells in the CSV)
    df_long.dropna(subset=["prompt"], inplace=True)

    # [Key]: Copy the contents of the 'Goal' column to the 'behavior' column,
    # Because the classifier requires the 'behavior' column to build the hint.
    if "Goal" in df_long.columns:
        df_long["behavior"] = df_long["Goal"]
    elif "Behavior" in df_long.columns:
        df_long["behavior"] = df_long["Behavior"]  # alternative
    else:
        logger.error(
            "The classifier will not work if the 'Goal' or 'Behavior' column is missing from the security dataset."
        )
        return pd.DataFrame()

        # Populate other columns that may be required by the classifier if they do not exist
    if "FunctionalCategory" not in df_long.columns:
        df_long["FunctionalCategory"] = "N/A"
    if "ContextString" not in df_long.columns:
        df_long["ContextString"] = ""

        # Add placeholders for merging with benign datasets
    df_long["utility_dataset_name"] = "N/A"
    df_long["max_new_tokens"] = max_new_tokens

    if sample_size > 0 and sample_size < len(df_long):
        # Note: Sampling may result in fewer samples for some attack methods
        df_long = df_long.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info(
        f"from{file_path}loaded and converted{len(df_long)}security samples (across{len(attack_cols)}attack method)."
    )
    return df_long


@torch.no_grad()
def run_generation(
    model,
    tokenizer,
    prompts: List[str],
    gen_config: GenerationConfig,
    batch_size: int,
    guard: Optional[Guard] = None,
) -> Tuple[List[str], List[int]]:
    """Run model generation with (or without) Guard.
    Return (generated text list, trigger step list)"""

    outputs = []
    all_trigger_steps = []
    condition_desc = "GUARDED" if guard else "BASELINE"

    guard_context = guard.attach(model) if guard else contextlib.nullcontext()

    try:
        with guard_context:
            for i in tqdm(
                range(0, len(prompts), batch_size), desc=f"run{condition_desc}generate"
            ):
                batch_prompts = prompts[i : i + batch_size]

                batch_trigger_logs = [-1] * len(batch_prompts)
                if guard:
                    guard.set_batch_log_target(batch_trigger_logs)

                conversations = [
                    [{"role": "user", "content": p}] for p in batch_prompts
                ]
                input_texts = [
                    tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True
                    )
                    for conv in conversations
                ]

                inputs = tokenizer(
                    input_texts, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)

                generated_ids = model.generate(**inputs, generation_config=gen_config)

                all_trigger_steps.extend(batch_trigger_logs)
                if guard:
                    guard.clear_batch_log_target()

                generated_texts = []
                for j, ids in enumerate(generated_ids):
                    input_len = len(inputs["input_ids"][j])
                    output_only_ids = ids[input_len:]
                    text = tokenizer.decode(
                        output_only_ids, skip_special_tokens=True
                    ).strip()
                    generated_texts.append(text)

                outputs.extend(generated_texts)

    except Exception as e:
        logger.error(
            f"exist{condition_desc}An error occurred during the build process:{e}",
            exc_info=True,
        )
        error_fill_text = [f"GENERATION_ERROR: {e}"] * (len(prompts) - len(outputs))
        error_fill_logs = [-1] * (len(prompts) - len(all_trigger_steps))
        outputs.extend(error_fill_text)
        all_trigger_steps.extend(error_fill_logs)

    return outputs, all_trigger_steps


@torch.no_grad()
def run_classification(
    classifier_model, classifier_tokenizer, df: pd.DataFrame, batch_size: int
) -> List[str]:
    """Determine the harmfulness of generated content using a classifier model."""

    labels = []
    if "assistant_output" not in df.columns:
        logger.error(
            "The 'assistant_output' column is missing from the dataframe and cannot be classified."
        )
        return ["error"] * len(df)
    if "behavior" not in df.columns:
        logger.error(
            "The 'behavior' column is missing from the dataframe and cannot be classified."
        )
        return ["error"] * len(df)

    for i in tqdm(range(0, len(df), batch_size), desc="Run classifier judgment"):
        batch_df = df.iloc[i : i + batch_size]
        prompts = []

        for _, row in batch_df.iterrows():
            behavior = str(row["behavior"])
            generation = (
                str(row["assistant_output"])
                if pd.notna(row["assistant_output"])
                else ""
            )

            if str(row.get("FunctionalCategory", "N/A")) == "contextual":
                context = str(row.get("ContextString", ""))
                prompt = LLAMA2_CLS_PROMPT["prompt_contextual"].format(
                    context=context, behavior=behavior, generation=generation
                )
            else:
                prompt = LLAMA2_CLS_PROMPT["prompt"].format(
                    behavior=behavior, generation=generation
                )
            prompts.append(prompt)

        inputs = classifier_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(classifier_model.device)

        output_ids = classifier_model.generate(
            **inputs, do_sample=False, max_new_tokens=1
        ).cpu()

        input_len_cpu = inputs.input_ids.shape[1]
        new_tokens = output_ids[:, input_len_cpu:]
        completions = classifier_tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        for c in completions:
            label = c.strip().lower()
            if "yes" in label:
                labels.append("yes")
            elif "no" in label:
                labels.append("no")
            else:
                logger.warning(
                    f"The classifier returned an unexpected label: '{c}'. Marked as 'unknown'."
                )
                labels.append("unknown")

    return labels


def calculate_metrics(df: pd.DataFrame, condition: str) -> Dict[str, Any]:
    """Calculate the harmful rate (ASR or FPR) for a given data frame.
    If condition is 'guarded', the trigger rate is also calculated."""
    total = len(df)
    if total == 0:
        metrics = {"rate": 0.0, "total": 0, "harmful_count": 0}
        # Only guarded adds trigger fields
        if condition == "guarded":
            metrics["trigger_rate"] = 0.0
            metrics["trigger_count"] = 0
        return metrics

        # 1. Calculate Harm Rate (ASR/FPR)
        # NOTE: If classification is not run, the label column will be pd.NA or 'skipped'
        # In this case, harmful_count will be 0 and rate will be 0.0
    harmful_count = (df["label"] == "yes").sum()
    rate = harmful_count / total if total > 0 else 0.0

    metrics = {
        "rate": rate,
        "total": int(total),
        "harmful_count": int(harmful_count),
    }

    # 2. Calculate trigger rate (only) for guarded
    if condition == "guarded":
        trigger_count = 0
        trigger_rate = 0.0
        if "trigger_step" in df.columns:
            # .notna() will capture all values ​​that are not <NA> (i.e. trigger_step >= 0)
            # Because we already converted -1 to <NA> in stage 1
            trigger_count = (df["trigger_step"].notna()).sum()
            trigger_rate = trigger_count / total if total > 0 else 0.0

        metrics["trigger_rate"] = trigger_rate
        metrics["trigger_count"] = int(trigger_count)

    return metrics


def get_override_max_new_tokens(df: pd.DataFrame) -> Optional[int]:
    """Parse the max_new_tokens override value from the dataset."""
    if "max_new_tokens" not in df.columns:
        return None
    values = df["max_new_tokens"].dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        logger.warning(
            f"Multiple max_new_tokens values ​​detected:{values}. The first value will be used."
        )
    try:
        return int(values[0])
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid max_new_tokens value:{values[0]}, the default configuration will be used."
        )
        return None


def build_generation_config(
    base_config: GenerationConfig, max_new_tokens_override: Optional[int]
) -> GenerationConfig:
    """Build a configuration based on the global GenerationConfig with optional max_new_tokens override."""
    if max_new_tokens_override is None:
        return base_config
    config_dict = base_config.to_dict()
    config_dict["max_new_tokens"] = max_new_tokens_override
    return GenerationConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Run the JEDI Defense Assessment (jbb_expanded.csv and alpaca_eval.json are supported)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation_config.yaml",
        help="Evaluate configuration file path.",
    )
    args = parser.parse_args()

    # --- 1. Load configuration ---
    config_path = base_dir / args.config
    if not config_path.exists():
        logger.error(f"Configuration file not found:{config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

        # --- [New] Load process control flag ---
    run_utility = config.get("run_utility_evaluation", True)
    run_safety = config.get("run_safety_evaluation", True)
    logger.info(
        f"Evaluation process configuration: run_utility_evaluation={run_utility}, run_safety_evaluation={run_safety}"
    )

    if not run_utility and not run_safety:
        logger.info(
            "Both run_utility_evaluation and run_safety_evaluation are false, and there is no evaluation task to be executed. Exiting."
        )
        return

        # --- 2. Set path ---
    llm_name = config["llm_name"]
    output_dir = base_dir / config["output_dir"] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = base_dir / config["data_dir"]

    # --- 3. Set the build configuration ---
    gen_config_dict = config.get("generation_kwargs", {})
    gen_config = GenerationConfig(
        max_new_tokens=gen_config_dict.get("max_new_tokens", 128),
        do_sample=gen_config_dict.get("do_sample", False),
        **{
            k: v
            for k, v in gen_config_dict.items()
            if k not in ["max_new_tokens", "do_sample"]
        },
    )
    batch_size = config.get("batch_size", 4)

    all_results_dfs = []  # Store all detailed DF results

    # ---
    # --- Phase 1: Generation
    # ---
    logger.info("--- Phase 1: Start text generation ---")
    try:
        # --- 3. Load Guard ---
        logger.info("--- Loading JEDI Guard ---")
        artifact_path = base_dir / config["artifact_path"] / llm_name
        try:
            guard = Guard.from_artifacts(
                artifact_path=str(artifact_path), device=device
            )
        except FileNotFoundError as e:
            logger.error(
                f"Failed to load Guard:{e}. Please make sure 04_calibrate_defense.py is run."
            )
            return

            # --- 4. Load target LLM ---
        logger.info("---Loading target LLM ---")
        model_config = config["llm_config"]
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_config["path"],
            model_kwargs=model_config.get("kwargs", {}),
            device=device,
        )
        # Make sure gen_config contains pad_token_id (set after model is loaded)
        gen_config.pad_token_id = tokenizer.pad_token_id
        gen_config.eos_token_id = tokenizer.eos_token_id

        # --- 7. Load data set ---
        # Modification: conditional loading based on run_utility flag
        df_utility_all = pd.DataFrame()  # initialized to empty
        if run_utility:
            logger.info("Loading 'Availability' dataset...")
            utility_config = config["utility_dataset_config"]
            utility_sample_size = utility_config.get(
                "sample_size", 0
            )  # This is the total sample size
            df_utility_all = load_utility_dataset(
                utility_config, data_dir, utility_sample_size
            )
        else:
            logger.info(
                'Depending on the configuration, skip loading the "Availability" data set.'
            )

            # Modification: conditional loading based on run_safety flag
        df_safety_long = pd.DataFrame()  # initialized to empty
        if run_safety:
            logger.info("Loading 'Security' dataset...")
            safety_config = config["safety_dataset_config"]
            safety_sample_size = safety_config.get("sample_size", 0)
            df_safety_long = load_safety_dataset(
                safety_config, data_dir, safety_sample_size
            )
        else:
            logger.info(
                'Depending on the configuration, skip loading the "Security" data set.'
            )

            # --- 8a. Evaluate Availability (FPR) ---
            # Modification: Loop by benign dataset name
            # The existing 'if not df_utility_all.empty:' check is sufficient,
            # Because if run_utility=False, df_utility_all will remain empty.
        if not df_utility_all.empty:
            utility_dataset_names = df_utility_all["utility_dataset_name"].unique()
            logger.info(
                f"--- Evaluation begins: Availability (FPR), will be tested{len(utility_dataset_names)}A benign data set ---"
            )

            for dataset_name in utility_dataset_names:
                logger.info(f"--- Evaluating benign dataset:{dataset_name} ---")
                df_utility_subset = df_utility_all[
                    df_utility_all["utility_dataset_name"] == dataset_name
                ].reset_index(drop=True)
                prompts = df_utility_subset["prompt"].tolist()
                num_samples = len(prompts)
                dataset_max_new_tokens = get_override_max_new_tokens(df_utility_subset)
                dataset_gen_config = build_generation_config(
                    gen_config, dataset_max_new_tokens
                )

                if not prompts:
                    logger.warning(
                        f"benign data set{dataset_name}No prompt to run, skip."
                    )
                    continue

                    # Guarded
                    # Add timing statistics
                start_time = time.time()
                guarded_outputs, guarded_triggers = run_generation(
                    model,
                    tokenizer,
                    prompts,
                    dataset_gen_config,
                    batch_size,
                    guard=guard,
                )
                end_time = time.time()
                total_duration = end_time - start_time
                avg_duration = total_duration / num_samples if num_samples > 0 else 0
                logger.info(
                    f"[Utility - {dataset_name}] Guarded generation statistics: total time taken{total_duration:.2f}s, the average time taken by a single sample{avg_duration:.4f}s"
                )

                df_guarded = df_utility_subset.copy()
                df_guarded["assistant_output"] = guarded_outputs
                df_guarded["trigger_step"] = guarded_triggers
                df_guarded["condition"] = "guarded"
                df_guarded["eval_split"] = (
                    "utility"  # Rename: prevent overwriting original 'dataset' fields
                )
                all_results_dfs.append(df_guarded)

                del guarded_outputs, guarded_triggers, df_guarded
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                # Baseline
                start_time = time.time()
                baseline_outputs, baseline_triggers = run_generation(
                    model,
                    tokenizer,
                    prompts,
                    dataset_gen_config,
                    batch_size,
                    guard=None,
                )
                end_time = time.time()
                total_duration = end_time - start_time
                avg_duration = total_duration / num_samples if num_samples > 0 else 0
                logger.info(
                    f"[Utility - {dataset_name}] Baseline generation statistics: total time taken{total_duration:.2f}s, the average time taken by a single sample{avg_duration:.4f}s"
                )

                df_baseline = df_utility_subset.copy()
                df_baseline["assistant_output"] = baseline_outputs
                df_baseline["trigger_step"] = baseline_triggers
                df_baseline["condition"] = "baseline"
                df_baseline["eval_split"] = (
                    "utility"  # Rename: prevent overwriting original 'dataset' fields
                )
                # 'attack_method' and 'utility_dataset_name' are set at load time
                all_results_dfs.append(df_baseline)

                del baseline_outputs, baseline_triggers, df_baseline
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                # Baseline

        else:
            logger.warning(
                "Availability evaluation is skipped because the data set is empty or configured to skip."
            )

            # --- 8b. Assessing Security (ASR), Grouping by Attack Method ---
            # The existing 'if not df_safety_long.empty:' check is sufficient,
            # Because if run_safety=False, df_safety_long will remain empty.
        if not df_safety_long.empty:
            attack_methods = df_safety_long["attack_method"].unique()
            logger.info(
                f"--- Evaluation begins: Security (ASR), will be tested{len(attack_methods)}attack method ---"
            )

            for method in attack_methods:
                logger.info(f"---Evaluating attack methods:{method} ---")
                df_attack = df_safety_long[
                    df_safety_long["attack_method"] == method
                ].reset_index(drop=True)
                prompts = df_attack["prompt"].tolist()
                dataset_max_new_tokens = get_override_max_new_tokens(df_attack)
                dataset_gen_config = build_generation_config(
                    gen_config, dataset_max_new_tokens
                )

                if not prompts:
                    logger.warning(f"method{method}No prompt to run, skip.")
                    continue

                    # Baseline
                baseline_outputs, baseline_triggers = run_generation(
                    model,
                    tokenizer,
                    prompts,
                    dataset_gen_config,
                    batch_size,
                    guard=None,
                )
                df_baseline = df_attack.copy()
                df_baseline["assistant_output"] = baseline_outputs
                df_baseline["trigger_step"] = baseline_triggers
                df_baseline["condition"] = "baseline"
                df_baseline["eval_split"] = "safety"  # Rename: use eval_split uniformly
                all_results_dfs.append(df_baseline)

                del df_baseline, baseline_outputs, baseline_triggers
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                # Guarded
                guarded_outputs, guarded_triggers = run_generation(
                    model,
                    tokenizer,
                    prompts,
                    dataset_gen_config,
                    batch_size,
                    guard=guard,
                )
                df_guarded = df_attack.copy()
                df_guarded["assistant_output"] = guarded_outputs
                df_guarded["trigger_step"] = guarded_triggers
                df_guarded["condition"] = "guarded"
                df_guarded["eval_split"] = "safety"  # Rename: use eval_split uniformly
                all_results_dfs.append(df_guarded)

                del df_guarded, guarded_outputs, guarded_triggers
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        else:
            logger.warning(
                "Security evaluation is skipped because the data set is empty or configured to be skipped."
            )

    finally:
        # ---Key step: Release the LLM under test ---
        if "model" in locals():
            model_name = (
                model.config.name_or_path
                if hasattr(model, "config") and hasattr(model.config, "name_or_path")
                else "Tested LLM"
            )
            logger.info(f"Freeing model from memory:{model_name}...")
            del model
            if "tokenizer" in locals():
                del tokenizer
            if "guard" in locals():
                del guard
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model memory has been released.")

    logger.info("--- Phase 1: Text generation is complete ---")

    if not all_results_dfs:
        logger.error(
            "No results were generated. Please check the dataset path and configuration."
        )
        return

        # ---
        # --- Stage 2: Classification
        # ---
    logger.info("--- Stage 2: Begin harmfulness judgment ---")

    final_results_df = pd.concat(all_results_dfs, ignore_index=True)
    del all_results_dfs  # Release list memory

    interim_csv_path = output_dir / f"{llm_name}_evaluation_INTERIM_all_results.csv"

    # --- Format trigger_step --- before saving intermediate and final files.
    if "trigger_step" in final_results_df.columns:
        final_results_df["trigger_step"] = (
            final_results_df["trigger_step"]
            .apply(lambda x: pd.NA if x == -1 else x)
            .astype("Int64")
        )

        # --- Organize columns and save *first* (without label) to *temporary* path ---
    try:
        logger.info(
            f"Saving intermediate build results (without labels) to:{interim_csv_path}"
        )

        base_cols_config = config.get("safety_dataset_config", {}).get(
            "base_columns", []
        )
        ordered_cols = base_cols_config + [
            "attack_method",
            "utility_dataset_name",
            "eval_split",
            "dataset",
            "condition",
            "prompt",
            "assistant_output",
            "trigger_step",
        ]
        current_cols = final_results_df.columns
        final_ordered_cols = [c for c in ordered_cols if c in current_cols]
        extra_cols = [c for c in current_cols if c not in final_ordered_cols]
        final_ordered_cols.extend(extra_cols)

        if "label" not in final_ordered_cols:
            final_ordered_cols.append("label")
            final_results_df["label"] = pd.NA

        final_results_df = final_results_df[final_ordered_cols]
        final_results_df.to_csv(interim_csv_path, index=False, encoding="utf-8-sig")
        logger.info("The intermediate generated result CSV is saved successfully.")
    except Exception as e:
        logger.warning(f"Failed to save CSV file:{e}", exc_info=True)

    try:
        # Modification: Separate data that needs to be classified (Safety) and data that does not need to be classified (Utility)
        mask_safety = final_results_df["eval_split"] == "safety"
        df_to_classify = final_results_df[mask_safety].copy()

        # Utility data set (if present) will be skipped and label remains NA
        df_skip_classify = final_results_df[~mask_safety]

        if not df_to_classify.empty:
            logger.info(
                f"Are facing{len(df_to_classify)}Use Safety samples to determine harmfulness (skip Utility samples)..."
            )

            # --- 5. Load classifier LLM ---
            logger.info("--- Loading classifier LLM ---")
            classifier_config = config["classifier_config"]
            classifier_batch_size = config.get("classifier_batch_size", 2)
            classifier_model, classifier_tokenizer = load_model_and_tokenizer(
                model_name=classifier_config["path"],
                model_kwargs=classifier_config.get("kwargs", {}),
                device=device,
            )

            # --- Run classification ---
            labels = run_classification(
                classifier_model,
                classifier_tokenizer,
                df_to_classify,
                classifier_batch_size,
            )
            df_to_classify["label"] = labels

            # Update the classified results back to final_results_df
            # Update using index alignment
            final_results_df.update(df_to_classify)
            logger.info("Safety sample classification completed.")
        else:
            logger.info(
                "There are no Safety samples to classify (or Safety evaluation is not enabled)."
            )

    finally:
        # --- Key step: Release classifier LLM ---
        if "classifier_model" in locals():
            model_name = (
                classifier_model.config.name_or_path
                if hasattr(classifier_model, "config")
                and hasattr(classifier_model.config, "name_or_path")
                else "Classifier LLM"
            )
            logger.info(f"Freeing model from memory:{model_name}...")
            del classifier_model
            if "classifier_tokenizer" in locals():
                del classifier_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model memory has been released.")

    logger.info("--- Phase 3: Calculate indicators and save split report ---")

    # --- Modification: Calculate and save availability metrics (FPR), loop by dataset name ---
    # The existing 'if run_utility:' check is not required,
    # Because if run_utility=False, the 'utility' data set will not exist in final_results_df
    df_utility_results = final_results_df[final_results_df["eval_split"] == "utility"]
    if not df_utility_results.empty:
        # Get all unique benign dataset names from the results
        utility_dataset_names = df_utility_results["utility_dataset_name"].unique()
        logger.info(
            f"working on{len(utility_dataset_names)}Generate results for benign datasets (skipping FPR metric calculation)..."
        )

        for dataset_name in utility_dataset_names:
            if pd.isna(dataset_name) or dataset_name == "N/A":
                continue

            # Filter out benign data sets for the current cycle
            df_utility_subset = df_utility_results[
                df_utility_results["utility_dataset_name"] == dataset_name
            ]
            if df_utility_subset.empty:
                continue

            # Modification: FPR is no longer calculated because label is missing or N/A
            logger.info(f"Saving Utility '{dataset_name}' Result file (CSV)...")

            # Save a separate file for this specific dataset
            try:
                utility_csv_path = (
                    output_dir
                    / f"{llm_name}_evaluation_detailed_utility_{dataset_name}.csv"
                )
                df_utility_subset.to_csv(
                    utility_csv_path, index=False, encoding="utf-8-sig"
                )

                logger.info(
                    f"Utility saved '{dataset_name}' Detailed CSV to:{utility_csv_path.name}"
                )

                # --- [New] Alpaca Eval specific format output ---
                if dataset_name == "alpaca_eval":
                    logger.info(
                        "Detected alpaca_eval dataset, generating dedicated evaluation JSON file..."
                    )

                    def save_alpaca_format(sub_df, out_filename, generator_name):
                        records = []
                        for _, row in sub_df.iterrows():
                            record = {
                                "dataset": row.get(
                                    "dataset", "alpaca_eval"
                                ),  # If the original data has a dataset field, use it, otherwise it defaults to
                                "instruction": row.get(
                                    "instruction", row.get("prompt", "")
                                ),  # instruction is alpaca's prompt
                                "output": row.get("assistant_output", ""),
                                "generator": generator_name,
                            }
                            # Try to keep the extra fields in the input if needed, but the user only specified these 4
                            records.append(record)

                        out_path = output_dir / out_filename
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(records, f, indent=2, ensure_ascii=False)
                        logger.info(
                            f"Alpaca Eval format results saved to:{out_path.name}"
                        )

                        # Save Baseline

                    df_base = df_utility_subset[
                        df_utility_subset["condition"] == "baseline"
                    ]
                    if not df_base.empty:
                        save_alpaca_format(
                            df_base,
                            f"{llm_name}-alpaca_eval-baseline.json",
                            f"{llm_name}-baseline",
                        )

                        # Save Guarded (JEDI)
                    df_guard = df_utility_subset[
                        df_utility_subset["condition"] == "guarded"
                    ]
                    if not df_guard.empty:
                        save_alpaca_format(
                            df_guard,
                            f"{llm_name}-alpaca_eval-JEDI.json",
                            f"{llm_name}-JEDI",
                        )

                        # --- [New] XSTest specific format output ---
                if "xstest" in str(dataset_name).lower():
                    logger.info(
                        f"xstest dataset detected ('{dataset_name}'), generating a dedicated evaluation CSV file..."
                    )

                    def save_xstest_format(sub_df, out_filename):
                        # Prepare 8 columns of data
                        target_cols = [
                            "id",
                            "type",
                            "prompt",
                            "completion",
                            "annotation_1",
                            "annotation_2",
                            "agreement",
                            "final_label",
                        ]
                        xstest_out = pd.DataFrame()

                        # map column
                        # id
                        if "id" in sub_df.columns:
                            xstest_out["id"] = sub_df["id"]
                        else:
                            xstest_out["id"] = range(1, len(sub_df) + 1)

                            # type
                        if "type" in sub_df.columns:
                            xstest_out["type"] = sub_df["type"]
                        else:
                            xstest_out["type"] = "N/A"

                            # prompt
                        xstest_out["prompt"] = sub_df["prompt"]

                        # completion (assistant_output)
                        xstest_out["completion"] = sub_df["assistant_output"]

                        # placeholder column
                        for col in [
                            "annotation_1",
                            "annotation_2",
                            "agreement",
                            "final_label",
                        ]:
                            xstest_out[col] = None

                            # Ensure column order
                        xstest_out = xstest_out[target_cols]

                        out_path = output_dir / out_filename
                        xstest_out.to_csv(out_path, index=False, encoding="utf-8-sig")
                        logger.info(f"XSTest format results saved to:{out_path.name}")

                        # Save Baseline

                    df_base = df_utility_subset[
                        df_utility_subset["condition"] == "baseline"
                    ]
                    if not df_base.empty:
                        save_xstest_format(df_base, f"{llm_name}_xstest_baseline.csv")

                        # Save Guarded (JEDI)
                    df_guard = df_utility_subset[
                        df_utility_subset["condition"] == "guarded"
                    ]
                    if not df_guard.empty:
                        save_xstest_format(df_guard, f"{llm_name}_xstest_guarded.csv")

            except Exception as e:
                logger.error(
                    f"Save Utility '{dataset_name}'Failed to result file:{e}",
                    exc_info=True,
                )
    else:
        logger.info("No availability results found.")

        # --- Calculate and save security metrics (ASR) ---
        # The existing 'if run_safety:' check is not required
    df_safety_results = final_results_df[final_results_df["eval_split"] == "safety"]
    if not df_safety_results.empty:
        attack_methods = df_safety_results["attack_method"].unique()
        logger.info(
            f"working on{len(attack_methods)}An attack method to calculate the ASR index..."
        )

        for method in attack_methods:
            if pd.isna(method) or method == "N/A":
                continue

            df_attack = df_safety_results[df_safety_results["attack_method"] == method]
            if df_attack.empty:
                continue

            baseline_asr_metrics = calculate_metrics(
                df_attack.query("condition == 'baseline'"), condition="baseline"
            )
            guarded_asr_metrics = calculate_metrics(
                df_attack.query("condition == 'guarded'"), condition="guarded"
            )

            attack_summary = {
                f"safety_asr_attack_{method}": {
                    "baseline": baseline_asr_metrics,
                    "guarded": guarded_asr_metrics,
                }
            }
            logger.info(
                f"Security (ASR) for{method} - Baseline: {baseline_asr_metrics['rate']:.4f}, Guarded: {guarded_asr_metrics['rate']:.4f}"
            )

            try:
                attack_csv_path = (
                    output_dir / f"{llm_name}_evaluation_detailed_attack_{method}.csv"
                )
                df_attack.to_csv(attack_csv_path, index=False, encoding="utf-8-sig")

                attack_json_path = (
                    output_dir / f"{llm_name}_evaluation_summary_attack_{method}.json"
                )
                with open(attack_json_path, "w", encoding="utf-8") as f:
                    json.dump(attack_summary, f, indent=2, default=str)

                logger.info(
                    f"Attack method saved '{method}'The result is:{attack_csv_path.name}and{attack_json_path.name}"
                )
            except Exception as e:
                logger.error(
                    f"Save attack method '{method}'Failed to result file:{e}",
                    exc_info=True,
                )
    else:
        logger.info("No security results found, skipping ASR calculation and saving.")

    logger.info("The evaluation process is fully completed.")

    # --- New: Automatically delete temporary files ---
    try:
        if "interim_csv_path" in locals() and interim_csv_path.exists():
            interim_csv_path.unlink()
            logger.info(f"Temporary files cleaned:{interim_csv_path.name}")
    except Exception as e:
        logger.warning(
            f"Clean temporary files{interim_csv_path.name}fails when:{e}", exc_info=True
        )


if __name__ == "__main__":
    main()
