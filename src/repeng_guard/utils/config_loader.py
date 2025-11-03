# -*- coding: utf-8 -*-
"""
工具：防御产物加载器

该文件提供了一个核心辅助函数 `load_defense_artifacts`，
用于从指定的目录中加载所有 SARC 防御所需的离线产物。

根据 `PROJECT_STRUCTURE.md` 和 `03/04` 号脚本，这些产物包括：
- `defense_params.yaml`: 包含校准后的参数 (theta, mu_hat, kappa, h, best_layer)。
- `transforms.pt`: 包含用于“早期窗口”和“内容窗口”的白化/中心化变换。
- `condition_vectors.pt`: 包含所有层的条件向量 (c_l)，用于检测。
- `intervention_vectors.pt`: 包含所有层的干预向量 (v_l)，用于转向。

`Guard` 类的 `from_artifacts` 方法依赖此加载器来初始化防御系统。
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
    从指定目录加载所有 SARC 防御产物。

    Args:
        artifact_path (str):
            包含所有产物文件的目录路径。
        device (str, optional):
            将 PyTorch 张量加载到的目标设备。默认为 'cpu'。

    Returns:
        Dict[str, Any]:
            一个字典，包含了所有加载的产物，结构如下：
            {
                'defense_params': {...},      // 来自 defense_params.yaml
                'transforms': {...},          // 来自 transforms.pt
                'condition_vectors': {...},   // 来自 condition_vectors.pt
                'intervention_vectors': {...} // 来自 intervention_vectors.pt
            }

    Raises:
        FileNotFoundError: 如果缺少任何必需的文件。
    """
    base_path = Path(artifact_path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"指定的产物路径不是一个有效的目录: {artifact_path}")

    files_to_load = {
        'defense_params': base_path / 'defense_params.yaml',
        'transforms': base_path / 'transforms.pt',
        'condition_vectors': base_path / 'condition_vectors.pt',
        'intervention_vectors': base_path / 'intervention_vectors.pt',
    }

    loaded_artifacts = {}
    map_location = torch.device(device)

    # 检查所有文件是否存在
    for key, path in files_to_load.items():
        if not path.exists():
            raise FileNotFoundError(f"必需的防御产物文件未找到: {path}")

    # 1. 加载 YAML 配置文件
    try:
        with open(files_to_load['defense_params'], 'r', encoding='utf-8') as f:
            loaded_artifacts['defense_params'] = yaml.safe_load(f)
        logger.info(f"成功加载防御参数: {files_to_load['defense_params']}")
    except Exception as e:
        logger.error(f"加载或解析 {files_to_load['defense_params']} 时出错: {e}")
        raise

    # 2. 加载 PyTorch 张量文件
    try:
        loaded_artifacts['transforms'] = torch.load(
            files_to_load['transforms'], map_location=map_location
        )
        logger.info(f"成功加载变换矩阵: {files_to_load['transforms']}")

        loaded_artifacts['condition_vectors'] = torch.load(
            files_to_load['condition_vectors'], map_location=map_location
        )
        logger.info(f"成功加载条件向量: {files_to_load['condition_vectors']}")

        loaded_artifacts['intervention_vectors'] = torch.load(
            files_to_load['intervention_vectors'], map_location=map_location
        )
        logger.info(f"成功加载干预向量: {files_to_load['intervention_vectors']}")

    except Exception as e:
        logger.error(f"加载 PyTorch 产物时出错 (设备: {device}): {e}")
        raise

    return loaded_artifacts
