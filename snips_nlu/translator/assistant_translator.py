# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from progressbar import ProgressBar
from snips_nlu.tokenization import tokenize
from snips_nlu.languages import Language
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.snowball import FrenchStemmer
from apis.systran_translator import SystranTranslator
from apis.gcloud_translator import GcloudTranslator
import HTMLParser
import json
from time import time
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)

def get_stemmer_for_language(lang):
    return SnowballStemmer(Language.name_from_iso_code(lang))

def _assemble_query(utterance):
    return "".join(part["text"] for part in utterance["data"])

def _find_in_seq_with_map(sub, seq, mapping):
    for i, _ in enumerate(seq):
        if i + len(sub) <= len(seq) and not any(mapping[i:i + len(sub)]):
            if seq[i : i + len(sub)] == sub:
                return i
    return -1

def _same_slot(el1, el2):
    return el1.get("slot_name") == el2.get("slot_name") and el1.get("entity") == el2.get("entity")

def _mk_translation_backend(model_name, authfile):
    if model_name=="google-neural" or model_name=="gn": return GcloudTranslator(model="nmt")
    if model_name=="google-phrase" or model_name=="gp": return GcloudTranslator(model="base")
    if model_name=="systran-neural" or model_name=="sn": return SystranTranslator(authfile, model="neural")
    if model_name=="systran-rule" or model_name=="sr": return SystranTranslator(authfile, model="rule")
    raise Exception("Unknown model {}".format(model_name))
    

class AssistantTranslator():

    def __init__(self,
                 source_language,
                 target_language,
                 backend=None,
                 modelname=None,
                 authfile=None,
                 cache=None,
                 log_level=None,
                 time_stats_file=None):
        
        if backend is not None and modelname is not None:
            logger.warning("Backend and model name both provided. Ignoring model name.")
        if backend is None and modelname is None:
            logger.critical("No backend or model name provided. Aborting.")
            raise Exception("No backend or model name provided.")
        
        self.translator = backend if backend is not None else _mk_translation_backend(modelname, authfile)
        self.source_language = source_language
        self.target_language = target_language
        self.stemmer = get_stemmer_for_language(target_language)
        self.html_parser = HTMLParser.HTMLParser()
        self.translation_cache_file = cache
        self.translation_cache = {}
        self.time_stats_file = time_stats_file
        self.time_stats = []
        
        if cache is not None:
            try:
                with open(cache, "r") as f:
                    logger.info("Loading translation cache from '{}'".format(cache))
                    self.translation_cache = json.load(f)
            except:
                logger.warning("Could not load cache file '{}'".format(cache))
        
        if log_level is not None: logger.setLevel(log_level)

    def _save_cache(self):
        if self.translation_cache_file is not None:
            logger.info("Saving translation cache to '{}'".format(self.translation_cache_file))
            try:
                with open(self.translation_cache_file, "w") as f:
                    json.dump(self.translation_cache, f, indent=2)
            except:
                logger.warning("Could not write cache file '{}'".format(self.translation_cache_file))
    
    def _save_time_stats(self):
        if self.time_stats_file is not None:
            with open(self.time_stats_file, "w") as f:
                json.dump(self.time_stats, f, indent=2)
    
    def translate_text(self, text):
        t_0 = time()
        t = self.html_parser.unescape(self.translator.translate(text, self.source_language, self.target_language))
        t_1 = time()
        self.time_stats.append(
            {
                "text": text,
                "duration": t_1 - t_0
            }
        )
        logger.info("Processing time for {} characters: {}".format(len(text), t_1 - t_0))
        return t
    
    def translate(self, phrase):
        if phrase in self.translation_cache:
            return self.translation_cache[phrase]
        else:
            phrase_t = self.translate_text(phrase)
            self.translation_cache[phrase] = phrase_t
            return phrase_t

    def stem_text(self, text):
        return self.stemmer.stem(text)

    def slot_values_match(self, slot_tokens_stemmed, translated_token_values_stemmed):
        match = True
        startIndex = 0
        if len(slot_tokens_stemmed) > 0:
            token = slot_tokens_stemmed[0]
            if token in translated_token_values_stemmed:
                startIndex = translated_token_values_stemmed.index(token)

        for ind_token, token in enumerate(slot_tokens_stemmed):
            if not translated_token_values_stemmed.count(token) == 1:
                match = False
            else:
                if translated_token_values_stemmed.index(token) != startIndex + ind_token:
                    match = False
        return match

    def _map_stemmed_slots(self, text, slots):
        utt = []
        unassigned_slots = []

        tokens = tokenize(text)
        stems = [ self.stem_text(token.value) for token in tokens ]
        slots_map = [ None ] * len(stems)
        
        # map stemmed slots to positions in stemmed tokenized text
        for slot in sorted(slots, key = lambda el: len(el["text"]), reverse=True):
            tokenized_slot = tokenize(slot["text"])
            stemmed_slot = [ self.stem_text(token.value) for token in tokenized_slot ]
            position = _find_in_seq_with_map(stemmed_slot, stems, slots_map)
            if position >= 0 and not any(slots_map[position:position + len(stemmed_slot)]):
                slots_map[position:position+len(stemmed_slot)] = [ {"text": token.value,
                                                                    "entity": slot.get("entity"),
                                                                    "slot_name": slot.get("slot_name"),
                                                                    "id": slot.get("id")
                                                                    }
                                                                   for token in tokenized_slot ]
            else:
                unassigned_slots.append(slot)
        
        for i, token in enumerate(tokens):
            if slots_map[i] is None:
                slots_map[i] = { "text": token.value }
        
        # merge sequences of tokens of same type
        utt = [ slots_map[0] ]
        for token in slots_map[1:]:
            if _same_slot(utt[-1], token):
                utt[-1]["text"] += " " + token["text"]
            else:
                utt.append(token)
        
        return utt, unassigned_slots

    def _map_slots(self, text, slots):
        utt, unassigned_slots = self._map_stemmed_slots(text, slots)
        
        # add spaces before and after text parts that aren't slots
        for i, part in enumerate(utt):
            if "entity" not in part and "slot_name" not in part:
                if i>0:               part["text"] = " "+part["text"]
                if i<len(utt)-1:    part["text"] = part["text"]+" "
        
        # set ranges in translated query
        offset = 0
        for part in utt:
            part["range"] = { "start": offset, "end": offset + len(part["text"]) }
            offset += len(part["text"])
        
        return {"data": utt}, unassigned_slots
        
    def translate_slots(self, utt):
        slots_t = []
        
        for data in utt["data"]:
            if "slot_name" in data or "entity" in data:
                data_t = deepcopy(data)
                data_t["text"] = self.translate(data["text"])
                slots_t.append(data_t)
        
        return slots_t

    def translate_utterance(self, utt):
        slots_t = self.translate_slots(utt)
        
        query = _assemble_query(utt)
        query_t = self.translate(query)

        logger.debug(u"Processing query '{}'".format(query))
        logger.debug(u"Translated query '{}'".format(query_t))
        
        utt_t, unassigned_slots = self._map_slots(query_t, slots_t)
        
        logger.debug(u"Remapped slots: {}".format(utt_t))
        logger.debug(u"Lost slots: {}".format(unassigned_slots))
        
        if len(unassigned_slots) > 0:
            logger.info("Unassigned slots: " + str(unassigned_slots))
        
        return utt_t
    
    def translate_entities(self, assistant):
        pbar = ProgressBar()
        for e_data in pbar(assistant["data"]["entities"].values()):
            for entry in e_data["data"]:
                entry["value"] = self.translate(entry["value"])
                entry["synonyms"] = [ self.translate(phrase) for phrase in entry["synonyms"] ]
            
        return assistant

    def translate_utterances(self, assistant):
        pbar = ProgressBar()
        for u_data in pbar(assistant["data"]["intents"].values()):
            u_data["utterances"] = [ self.translate_utterance(utt) for utt in u_data["utterances"] ]
    
        return assistant

    def translate_assistant(self, assistant):
        
        logger.info("Translating entitites")
        assistant_t = self.translate_entities   (assistant)
        
        logger.info("Translating utterances")
        assistant_t = self.translate_utterances (assistant_t)
        
        assistant_t["language"] = self.target_language
        
        self._save_cache()
        self._save_time_stats()
        
        return assistant_t
    
    def translate_data(self, data):
        
        data_t = {}
        for intent, intent_data in data.items():
            data_t[intent] = [ self.translate_utterance(utt) for utt in intent_data ]

        self._save_cache()
        self._save_time_stats()
        
        return data_t
