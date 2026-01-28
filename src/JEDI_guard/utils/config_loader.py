# -*- coding: utf-8 -*-
"""
Utility: defense artifact loader

This file provides the core helper function `load_defense_artifacts`, which
loads all offline artifacts required by the JEDI defense from a given directory.

According to `PROJECT_STRUCTURE.md` and scripts dated 03/04, the artifacts include:
- `defense_params.yaml`: calibrated parameters (theta, mu_hat, kappa, alpha, best_layer).
- `transforms.pt`: whitening/centering transforms for "early" and "content" windows.
- `condition_vectors.pt`: condition vectors (c_l) for all layers, used for detection.
- `intervention_vectors.pt`: intervention vectors (v_l) for all layers, used for steering.

The `Guard.from_artifacts` method depends on this loader to initialize the defense system.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_defense_artifacts(
        artifact_path: str,
        device: Optional[str] = 'cpu'
) -> Dict[str, Any]:
    """
    Load all JEDI defense artifacts from the specified directory.

    Args:
        artifact_path (str):
            Directory path containing all artifact files.
        device (str, optional):
            Target device to load PyTorch tensors onto. Defaults to 'cpu'.

    Returns:
        Dict[str, Any]:
            A dictionary containing all loaded artifacts with the structure:
            {
                'defense_params': {...},      // from defense_params.yaml
                'transforms': {...},          // from transforms.pt
                'condition_vectors': {...},   // from condition_vectors.pt
                'intervention_vectors': {...} // from intervention_vectors.pt
            }

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    base_path = Path(artifact_path)
    if not base_path.is_dir():
        raise FileNotFoundError(
            f"The specified artifact path is not a valid directory: {artifact_path}"
        )

    files_to_load = {
        'defense_params': base_path / 'defense_params.yaml',
        'transforms': base_path / 'transforms.pt',
        'condition_vectors': base_path / 'condition_vectors.pt',
        'intervention_vectors': base_path / 'intervention_vectors.pt',
    }

    loaded_artifacts = {}
    map_location = torch.device(device)

    # Check that all files exist
    for key, path in files_to_load.items():
        if not path.exists():
            raise FileNotFoundError(f"Required defense artifact file not found: {path}")

    # 1. Load YAML config file
    try:
        with open(files_to_load['defense_params'], 'r', encoding='utf-8') as f:
            loaded_artifacts['defense_params'] = yaml.safe_load(f)
        logger.info("Loaded defense params: %s", files_to_load['defense_params'])
    except Exception as e:
        logger.error("Error loading or parsing %s: %s", files_to_load['defense_params'], e)
        raise

    # 2. Load PyTorch tensor files
    try:
        loaded_artifacts['transforms'] = torch.load(
            files_to_load['transforms'], map_location=map_location
        )
        logger.info("Loaded transform matrices: %s", files_to_load['transforms'])

        loaded_artifacts['condition_vectors'] = torch.load(
            files_to_load['condition_vectors'], map_location=map_location
        )
        logger.info("Loaded condition vectors: %s", files_to_load['condition_vectors'])

        loaded_artifacts['intervention_vectors'] = torch.load(
            files_to_load['intervention_vectors'], map_location=map_location
        )
        logger.info("Loaded intervention vectors: %s", files_to_load['intervention_vectors'])

    except Exception as e:
        logger.error("Error loading PyTorch artifacts (device: %s): %s", device, e)
        raise

    return loaded_artifacts
