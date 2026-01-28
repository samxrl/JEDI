# -*- coding: utf-8 -*-
"""
Utility: logging setup

Provides a simple function to configure logging for the JEDI package.
This helps produce standardized, controllable log output when `Guard` runs,
which is useful for debugging and auditing.
"""

import logging
import sys


def setup_logging(level=logging.INFO, stream=sys.stdout):
    """
    Configure logging for the JEDI package (or root logger).

    Args:
        level (int, optional):
            Log level (e.g., logging.INFO, logging.DEBUG).
            Defaults to logging.INFO.
        stream (IO, optional):
            Log output stream. Defaults to sys.stdout.
    """
    # Get the root logger for the 'JEDI_guard' package
    # If using outside the package, you can get the root logger instead:
    # logger = logging.getLogger()

    logger = logging.getLogger('JEDI_guard')
    if logger.hasHandlers():
        # If already configured, do not reconfigure
        return

    logger.setLevel(level)

    # Create a stream handler
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Prevent log messages from propagating to the root logger (if it has handlers)
    logger.propagate = False

    logger.info("RepEng-Guard logger initialized.")
