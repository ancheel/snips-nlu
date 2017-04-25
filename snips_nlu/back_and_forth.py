#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import logging
from os import getenv
from os.path import join
from translator.assistant_translator import AssistantTranslator
from translate_data import load_data, save_data

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
    default_systran_auth = join(getenv("HOME"), ".systran", "auth.json")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="Data to be translated (JSON file, required). Output will be written to <filename>_<model>_<s>_<t>_<s>.json")
    parser.add_argument("source_language", help="Source language (ISO code, required)")
    parser.add_argument("pivot_language", help="Pivot language (ISO code, required)")
    parser.add_argument("--auth", default=default_systran_auth, help="Auth file (default: {})".format(default_systran_auth))
    parser.add_argument("-c", "--cache", help="Translation cache directory")
    parser.add_argument("-m", "--model", help= "Translation model:\n"
                                               "\tgoogle-neural or gn: Google's neural translation model (default)\n"
                                               "\tgoogle-phrase or gp: Google's phrase based translation model\n"
                                               "\tsystran-neural or sn: Systran's neural translation model\n"
                                               "\tsystran-rule or sr: Systran's rule based translation model\n"
                        )
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase verbosity")
    parser.add_argument("-t", "--time-data", metavar="file basename", help="Store query processising times in <file>_<model>_<s>_<t>.json.")
    args = parser.parse_args()
    logger.setLevel(log_levels_mapping(args.verbosity))
    
    if args.input is None or args.source_language is None or args.pivot_language is None:
        parser.print_help()
        error("Invalid command line")
    
    s = args.source_language.lower()
    t = args.pivot_language.lower()
    
    T_forth = AssistantTranslator(s,
                                  t,
                                  modelname=args.model,
                                  authfile=args.auth,
                                  cache=join(args.cache,
                                            "cache_{}_{}_{}.json".format(args.model,
                                                                        s,
                                                                        t)) if args.cache is not None else None,
                                  log_level=logger.getEffectiveLevel(),
                                  time_stats_file="{}_{}_{}.json".format(args.time_data,
                                                                              args.model,
                                                                              s,
                                                                              t
                                                                              )
                                  )
        
    T_back = AssistantTranslator(t,
                                 s,
                                 modelname=args.model,
                                 authfile=args.auth,
                                 cache=join(args.cache,
                                            "cache_{}_{}_{}.json".format(args.model,
                                                                        t,
                                                                        s)) if args.cache is not None else None,
                                 log_level=logger.getEffectiveLevel(),
                                 time_stats_file="{}_{}_{}.json".format(args.time_data,
                                                                             args.model,
                                                                             t,
                                                                             s
                                                                             )
                                 )
    
    
    back_filename = "{}_{}_{}_{}_{}.json".format(args.input.replace(".json", ""),
                                                 args.model,
                                                 s,
                                                 t,
                                                 s)

    data_s = load_data(args.input)

    logger.info("Translating data {}->{}".format(s, t))
    data_s_t = T_forth.translate_data(data_s)

    logger.info("Translating data {}->{}".format(t, s))
    data_s_t_s = T_back.translate_data(data_s_t)

    save_data(data_s_t_s, back_filename)