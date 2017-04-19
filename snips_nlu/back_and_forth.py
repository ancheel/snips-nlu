#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging


def log_levels_mapping(verbose):
    if verbose==0: return logging.WARNING
    if verbose==1: return logging.INFO
    if verbose>=2: return logging.DEBUG


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)


def error(msg, code=1):
    """Log an error message and exit with given code (default: 1)."""
    logger.error(msg)
    exit(code)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output JSON file (required)")
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase verbosity")
    args = parser.parse_args()
    logger.setLevel(log_levels_mapping(args.verbosity))
    
