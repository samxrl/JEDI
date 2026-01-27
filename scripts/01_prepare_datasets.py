# -*- coding: utf-8 -*-
"""Script 01: Dataset preparation and slicing

The script follows the first stage in the "Methodflow.md" document for preparing the dataset required to train the representation vectors.

Main functions:
1. Load the original jailbreak tip data set (for example, JSONL format).
2. Based on the "compliance" and "refusal" prefix lists defined in the configuration file.
3. For each jailbreak prompt, two types of conversation samples are generated:
    - A1 (satisfied type): user prompt + prefix that the model satisfies.
    - A2 (rejection type): user prompt + prefix for model rejection.
4. Process the benign data set, pair it with the "satisfied" prefix, and generate B1 (benign-satisfied) samples.
    - Supports CSV format (reading 'prompt' or 'Goal' columns).
    - Support JSONL format (specific support for justeval structure, filter source_id!=alpaca_eval and category==regular).
5. Save all generated conversation samples in a structured CSV format to the processed data directory.
    - Each sample contains the original prompt, the prefix used, the category, and the conversation structure that matches the model input (the conversation structure will be stored as a JSON string).


How to run:
python scripts/01_prepare_datasets.py --config configs/data_prep_config.yaml"""

import json
import argparse
import csv
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd
import random


def create_paired_samples(
    prompt: str,
    behavior: str,
    functional_category: str,
    context_string: str,
    compliance_prefixes: list,
    refusal_prefixes: list,
    compliance_category: str = "compliance",
    refusal_category: str = "refusal",
) -> tuple[list, list]:
    """Create pairs of samples for a given prompt word.

    Args:
        prompt (str): The user's original prompt.
        behavior (str): Behavior description.
        functional_category (str): Functional category.
        context_string (str): context information string.
        compliance_prefixes (list): A list of prefixes indicating that the model satisfies the intent.
        refusal_prefixes (list): A list of prefixes representing the model's rejection intent.
        compliance_category (str): The category label of the compliance sample.
        refusal_category (str): The category label of the rejection sample.

    Returns:
        tuple[list, list]: A tuple containing two lists: (satisfied sample list, rejected sample list).
                           Each sample is a dictionary containing the conversation history.
    """
    compliance_samples = []
    refusal_samples = []

    # ---Create satisfying samples ---
    for prefix in compliance_prefixes:
        # Build an input structure that conforms to the model dialogue template
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix},
        ]
        compliance_samples.append(
            {
                "prompt": prompt,
                "prefix": prefix,
                "category": compliance_category,
                "conversation": conversation,
                "behavior": behavior,
                "FunctionalCategory": functional_category,
                "ContextString": context_string,
            }
        )

        # ---Create rejection samples ---
    for prefix in refusal_prefixes:
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix},
        ]
        refusal_samples.append(
            {
                "prompt": prompt,
                "prefix": prefix,
                "category": refusal_category,
                "conversation": conversation,
                "behavior": behavior,
                "FunctionalCategory": functional_category,
                "ContextString": context_string,
            }
        )

    return compliance_samples, refusal_samples


def process_dataset(config: dict):
    """Process all specified data sets according to the configuration file.

    Args:
        config (dict): Configuration dictionary loaded from YAML file."""
    config_path = Path(config["__config_path__"])
    base_dir = config_path.parent.parent
    output_dir = base_dir / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"All processed files will be saved in:{output_dir.resolve()}")

    # --- Load prefix list uniformly ---
    prefixes_config = config.get("prefixes", {})
    source_file = prefixes_config.get("source_file")
    compliance_prefixes = []
    refusal_prefixes = []

    if source_file:
        source_file_path = base_dir / source_file
        print(f"Load prefix from file:{source_file_path.resolve()}")
        try:
            with open(source_file_path, "r", encoding="utf-8") as f:
                prefix_data = json.load(f)

            compliance_key = prefixes_config.get("compliance_key")
            refusal_key = prefixes_config.get("refusal_key")

            if not compliance_key or not refusal_key:
                raise ValueError(
                    "When 'source_file' is specified, both 'compliance_key' and 'refusal_key' must be provided in the configuration file."
                )

            compliance_prefixes = prefix_data.get(compliance_key, [])
            refusal_prefixes = prefix_data.get(refusal_key, [])
            print(
                f"loaded{len(compliance_prefixes)}The prefix sum satisfies{len(refusal_prefixes)}Denied prefix."
            )

        except FileNotFoundError:
            print(f"ERROR: Prefix source file not found:{source_file_path.resolve()}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"ERROR: Parsing prefix file{source_file_path.resolve()}Or an error occurred while looking for the specified key:{e}"
            )
            return
    else:
        # If no source file is specified, attempts to read directly from the configuration (legacy compatibility)
        print("Load prefix list directly from configuration file...")
        compliance_prefixes = prefixes_config.get("compliance", [])
        refusal_prefixes = prefixes_config.get("refusal", [])
        print(
            f"loaded{len(compliance_prefixes)}The prefix sum satisfies{len(refusal_prefixes)}Denied prefix."
        )

        # --- Processing jailbreak data sets ---
    if "jailbreak_dataset" in config:
        if not compliance_prefixes or not refusal_prefixes:
            raise ValueError(
                "When processing jailbroken datasets, the 'compliance' and 'refusal' prefix lists must be loaded successfully."
            )

        print("Starting to process the jailbreak dataset...")
        jb_config = config["jailbreak_dataset"]
        input_path = base_dir / jb_config["input_file"]

        compliance_output_path = (
            output_dir / f"{config['llm_name']}_jailbreak_compliance.csv"
        )
        refusal_output_path = output_dir / f"{config['llm_name']}_jailbreak_refusal.csv"

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data["records"]
        except FileNotFoundError:
            print(f"ERROR: Input file not found:{input_path.resolve()}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"ERROR: Parse file{input_path.resolve()}An error occurred:{e}. Please make sure the file is in valid JSON format and that each item in the 'records' list has a 'prompt' key."
            )
            return

        print(f"from{input_path.resolve()}loaded{len(records)}jailbreak tips.")

        with open(
            compliance_output_path, "w", encoding="utf-8-sig", newline=""
        ) as f_comply, open(
            refusal_output_path, "w", encoding="utf-8-sig", newline=""
        ) as f_refuse:

            comply_writer = csv.writer(f_comply)
            refuse_writer = csv.writer(f_refuse)

            header = [
                "prompt",
                "prefix",
                "category",
                "conversation",
                "behavior",
                "FunctionalCategory",
                "ContextString",
            ]
            comply_writer.writerow(header)
            refuse_writer.writerow(header)

            for record in tqdm(records, desc="Processing jailbreak prompts"):
                prompt = record["prompt"]
                behavior = record["behavior"]
                functional_category = record["FunctionalCategory"]
                context_string = (
                    record["ContextString"]
                    if functional_category == "contextual"
                    else ""
                )

                compliance_samples, refusal_samples = create_paired_samples(
                    prompt,
                    behavior,
                    functional_category,
                    context_string,
                    compliance_prefixes,
                    refusal_prefixes,
                )
                for sample in compliance_samples:
                    conversation_str = json.dumps(
                        sample["conversation"], ensure_ascii=False
                    )
                    row = [
                        sample["prompt"],
                        sample["prefix"],
                        sample["category"],
                        conversation_str,
                        sample["behavior"],
                        sample["FunctionalCategory"],
                        sample["ContextString"],
                    ]
                    comply_writer.writerow(row)
                for sample in refusal_samples:
                    conversation_str = json.dumps(
                        sample["conversation"], ensure_ascii=False
                    )
                    row = [
                        sample["prompt"],
                        sample["prefix"],
                        sample["category"],
                        conversation_str,
                        sample["behavior"],
                        sample["FunctionalCategory"],
                        sample["ContextString"],
                    ]
                    refuse_writer.writerow(row)

        print(
            f"Successfully write the satisfying type (A1) sample into:{compliance_output_path.resolve()}"
        )
        print(
            f"Successfully written rejection type (A2) sample to:{refusal_output_path.resolve()}"
        )

        # --- Processing benign data sets ---
    if (
        "benign_dataset" in config
        and config["benign_dataset"].get("input_files") is not None
    ):
        if not compliance_prefixes:
            raise ValueError(
                "When working with benign datasets, the 'compliance' prefix list must be loaded successfully."
            )

        print("Start processing benign data sets...")
        benign_config = config["benign_dataset"]
        input_files = benign_config["input_files"]
        sample_size = benign_config.get("sample_size", 100)
        output_path = output_dir / f"{config['llm_name']}_benign_compliance.csv"

        all_prompts = []
        for file_info in input_files:
            file_path = base_dir / Path(file_info)
            prompts_from_file = []

            try:
                # Check file suffix to distinguish processing logic
                if file_path.suffix.lower() == ".jsonl":
                    # Processing JSONL format (specifically for justeval type structures)
                    print(
                        f"JSONL file detected, processed according to JustEval structure:{file_path.resolve()}"
                    )
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    candidates = []
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            # Filter criteria:
                            # 1. source_id does not contain 'alpaca_eval'
                            # 2. category is 'regular'
                            source_id = str(item.get("source_id", ""))
                            category = str(item.get("category", ""))

                            if "alpaca_eval" not in source_id and category == "regular":
                                # To extract prompt, use 'instruction' first, followed by 'prompt'
                                p_text = item.get("instruction") or item.get("prompt")
                                if p_text:
                                    candidates.append(str(p_text))
                        except json.JSONDecodeError:
                            print(
                                f"WARNING: Skipping unparsable JSON lines in{file_path.name}"
                            )
                            continue

                            # Sampling filtered data
                    if candidates:
                        num_candidates = len(candidates)
                        actual_sample_size = min(sample_size, num_candidates)
                        sampled_list = random.sample(candidates, actual_sample_size)
                        for p in sampled_list:
                            prompts_from_file.append((p, file_path.name))
                        print(
                            f"from{file_path.resolve()}filter out{num_candidates}valid samples and sampled{len(prompts_from_file)}strip."
                        )
                    else:
                        print(
                            f"WARNING: in{file_path.name}No matching sample was found (source_id!=alpaca_eval, category==regular)."
                        )

                else:
                    # Process CSV format by default (original logic)
                    df = pd.read_csv(file_path)
                    prompt_column = "prompt"
                    if prompt_column not in df.columns:
                        if "Goal" in df.columns:
                            prompt_column = "Goal"
                        else:
                            raise ValueError(
                                f"exist{file_path}No suitable prompt column ('prompt' or 'Goal') found in ."
                            )

                    num_rows = len(df)
                    if num_rows > 0:
                        actual_sample_size = min(sample_size, num_rows)
                        sampled_df = df.sample(n=actual_sample_size, random_state=42)
                        for _, row in sampled_df.iterrows():
                            prompt = row[prompt_column]
                            if pd.notna(prompt):
                                # Save prompt and its source file name
                                prompts_from_file.append((str(prompt), file_path.name))
                    print(
                        f"from{file_path.resolve()}(CSV) Sampled successfully{len(prompts_from_file)}A good reminder."
                    )

                all_prompts.extend(prompts_from_file)

            except FileNotFoundError:
                print(f"ERROR: Input file not found:{file_path.resolve()}")
            except Exception as e:
                print(
                    f"ERROR: processing file{file_path.resolve()}An error occurred:{e}"
                )

        if all_prompts:
            # Scramble samples from all sources again
            random.shuffle(all_prompts)
            print(f"Collect in total{len(all_prompts)}benign tips for processing.")

            with open(output_path, "w", encoding="utf-8-sig", newline="") as f_out:
                header = ["prompt", "prefix", "category", "conversation", "source"]
                writer = csv.writer(f_out)
                writer.writerow(header)

                total_samples_written = 0
                for prompt, source in tqdm(
                    all_prompts, desc="Processing benign prompts"
                ):
                    compliance_samples, _ = create_paired_samples(
                        prompt=prompt,
                        behavior="",
                        functional_category="",
                        context_string="",
                        compliance_prefixes=compliance_prefixes,
                        refusal_prefixes=[],  # Benign samples do not need to be paired with reject prefixes
                        compliance_category="benign_compliance",
                    )

                    for sample in compliance_samples:
                        conversation_str = json.dumps(
                            sample["conversation"], ensure_ascii=False
                        )
                        row = [
                            sample["prompt"],
                            sample["prefix"],
                            sample["category"],
                            conversation_str,
                            source,
                        ]
                        writer.writerow(row)
                        total_samples_written += 1

            print(
                f"Success will{total_samples_written}The sample of benign satisfaction type (B1) is written:{output_path.resolve()}"
            )


def main():
    """The main function is used to parse command line parameters and start the data set processing process."""
    parser = argparse.ArgumentParser(
        description="Prepare data sets for characterization projects based on profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/data_prep_config.yaml",
        help="Specifies the YAML configuration file path for the data preparation phase. \\nDefault: ../configs/data_prep_config.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file does not exist:{config_path.resolve()}")
        return

    print(f"Load configuration file:{config_path.resolve()}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config["__config_path__"] = str(
            config_path.resolve()
        )  # Inject the configuration file path to facilitate calculation of relative paths

    process_dataset(config)

    print("The data set preparation process is complete.")


if __name__ == "__main__":
    main()
