# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import logging
import json
from translator.assistant_translator import AssistantTranslator
from translator.apis.dummy_translator import DummyTranslator
from translator.apis.gcloud_translator import GcloudTranslator

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


def get_assistant_language(assistant):
    return assistant["language"]

def load_assistant(fname):
    try:
        with open(fname) as f:
            logger.info("Loading assistant from '{}'".format(fname))
            return json.load(f)
    except:
        error("Could not load assistant from '{}'".format(fname))

def save_assistant(assistant, fname):
    try:
        with open(fname, "w") as f:
            logger.info("Saving translated assistant to '{}'".format(fname))
            json.dump(assistant, f, indent=4)
    except:
        error("Could not save assistant to '{}'".format(fname))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Assistant to translated (JSON file, required)")
    parser.add_argument("language", help="Target language (ISO code, required)")
    parser.add_argument("output", help="Where to save translated assistant (JSON file, required))")
    parser.add_argument("-c", "--cache", help="Translation cache file")
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase verbosity")
    args = parser.parse_args()
    logger.setLevel(log_levels_mapping(args.verbosity))
    
    if args.input is None or args.output is None or args.language is None:
        parser.print_help()
        error("Invalid command line")
    
    assistant_s = load_assistant(args.input)
    
    #translation_backend = DummyTranslator()
    translation_backend = GcloudTranslator()
    translator = AssistantTranslator(translation_backend,
                                     get_assistant_language(assistant_s),
                                     args.language.lower(),
                                     args.cache,
                                     logger.getEffectiveLevel())
    
    logger.info("Translating assistant")
    assistant_t = translator.translate_assistant(assistant_s)
    
    save_assistant(assistant_t, args.output)