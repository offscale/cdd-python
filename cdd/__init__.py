#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Root __init__
"""

import logging
from logging import Logger
from logging import getLogger as get_logger

__author__ = "Samuel Marks"  # type: str
__version__ = "0.0.99rc46"  # type: str
__description__ = (
    "Open API to/fro routes, models, and tests. "
    "Convert between docstrings, classes, methods, argparse, pydantic, and SQLalchemy."
)  # type: str


root_logger: Logger = get_logger()
logging.getLogger("blib2to3").setLevel(logging.WARNING)

__all__ = [
    "get_logger",
    "root_logger",
    "__description__",
    "__version__",
]  # type: list[str]
