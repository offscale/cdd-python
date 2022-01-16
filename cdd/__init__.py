#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Root __init__
"""

import logging
from logging import getLogger as get_logger

__author__ = "Samuel Marks"
__version__ = "0.0.81"
__description__ = (
    "Open API to/fro routes, models, and tests. "
    "Convert between docstrings, classes, methods, argparse, and SQLalchemy."
)


root_logger = get_logger()
logging.getLogger("blib2to3").setLevel(logging.WARNING)

__all__ = ["get_logger", "root_logger", "__version__"]
