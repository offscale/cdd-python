#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging.config import dictConfig as _dictConfig
from os import path

import yaml

__author__ = 'Samuel Marks'
__version__ = '0.0.1'


def get_logger(name=None):
    with open(path.join(path.dirname(__file__), '_data', 'logging.yml'), 'rt') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    _dictConfig(data)
    return logging.getLogger(name=name)


root_logger = get_logger()
