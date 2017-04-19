# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import logging
import json
from translator.assistant_translator import AssistantTranslator

def log_levels_mapping(verbose):
    if verbose == 0: return logging.WARNING
    if verbose == 1: return logging.INFO
    if verbose >= 2: return logging.DEBUG

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)

def error(msg, code=1):
    """Log an error message and exit with given code (default: 1)."""
    logger.error(msg)
    exit(code)


def load_data(fname):
    try:
        with open(fname) as f:
            logger.info("Loading data from '{}'".format(fname))
            return json.load(f)
    except:
        error("Could not load data from '{}'".format(fname))

def save_data(data, fname):
    try:
        with open(fname, "w") as f:
            logger.info("Saving translated data to '{}'".format(fname))
            json.dump(data, f, indent=2)
    except:
        error("Could not save assistant to '{}'".format(fname))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Data to be translated (JSON file, required)")
    parser.add_argument("source_language", help="Source language (ISO code, required)")
    parser.add_argument("target_language", help="Target language (ISO code, required)")
    parser.add_argument("output", help="Where to save translated data (JSON file, required))")
    parser.add_argument("-m", "--model", help=  "Translation model:\n"
                                                "\tgoogle-neural or gn: Google's neural translation model (default)\n"
                                                "\tgoogle-phrase or gp: Google's phrase based translation model\n"
                                                "\tsystran-neural or sn: Systran's neural translation model\n"
                                                "\tsystran-rule or sr: Systran's rule based translation model\n"
                        )
    parser.add_argument("-c", "--cache", help="Translation cache file")
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase verbosity")
    args = parser.parse_args()
    logger.setLevel(log_levels_mapping(args.verbosity))
    
    if args.input is None or args.output is None or args.source_language is None or args.target_language is None:
        parser.print_help()
        error("Invalid command line")
    
    data_s = load_data(args.input)
    
    translator = AssistantTranslator(args.source_language.lower(),
                                     args.target_language.lower(),
                                     args.model,
                                     args.cache,
                                     logger.getEffectiveLevel())
    
    logger.info("Translating assistant")
    data_t = translator.translate_data(data_s)
    
    save_data(data_t, args.output)