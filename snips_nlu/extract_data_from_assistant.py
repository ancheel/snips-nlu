# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import logging
import json

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


def load_assistant(fname):
    try:
        with open(fname) as f:
            logger.info("Loading assistant from '{}'".format(fname))
            return json.load(f)
    except:
        error("Could not load assistant from '{}'".format(fname))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Assistant to load (JSON file, required)")
    parser.add_argument("output", help="Output JSON file (required)")
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase verbosity")
    args = parser.parse_args()
    logger.setLevel(log_levels_mapping(args.verbosity))
    if args.input is None or args.output is None:
        parser.print_help()
        error("Invalid command line")
    
    assistant = load_assistant(args.input)
    
    data = dict(
        [
            (intent, intent_data["utterances"]) for intent, intent_data in assistant["data"]["intents"].items()
        ]
    )
    
    with open(args.output, "w") as f: json.dump(data, f, indent=2)
