# -*- coding: utf-8 -*-
"""
工具：日志设置

提供一个简单的函数来配置 SARC 包的日志记录。
这有助于在 `Guard` 运行时提供标准化的、可控的日志输出，
方便调试和审计。
"""

import logging
import sys


def setup_logging(level=logging.INFO, stream=sys.stdout):
    """
    配置 SARC 包（或根日志记录器）的日志记录。

    Args:
        level (int, optional):
            日志级别 (例如, logging.INFO, logging.DEBUG)。
            默认为 logging.INFO。
        stream (IO, optional):
            日志输出流。默认为 sys.stdout。
    """
    # 获取 'repeng_guard' 包的根日志记录器
    # 如果在包外部使用，可以改为获取根日志记录器：
    # logger = logging.getLogger()

    logger = logging.getLogger('repeng_guard')
    if logger.hasHandlers():
        # 如果已经配置过，则不再重复配置
        return

    logger.setLevel(level)

    # 创建一个流处理器 (StreamHandler)
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    # 创建一个格式化器 (Formatter) 并添加到处理器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(handler)

    # 防止日志消息传播到根日志记录器 (如果根有自己的处理器)
    logger.propagate = False

    logger.info("RepEng-Guard 日志记录器已初始化。")
