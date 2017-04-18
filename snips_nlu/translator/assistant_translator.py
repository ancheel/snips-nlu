# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from progressbar import ProgressBar
from snips_nlu.tokenization import tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import FrenchStemmer
import HTMLParser
import json
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)

def _assemble_query(utterance):
    return "".join(part["text"] for part in utterance["data"])

def _find_in_seq_with_map(sub, seq, mapping):
    for i, _ in enumerate(seq):
        if i + len(sub) <= len(seq) and not any(mapping[i:i + len(sub)]):
            if seq[i:i + len(sub)] == sub:
                return i
    return -1

def _same_slot(el1, el2):
    return el1.get("slot_name") == el2.get("slot_name") and el1.get("entity") == el2.get("entity")

class AssistantTranslator():

    def __init__(self,
                 translator,
                 source_language,
                 target_language,
                 translation_cache_file=None,
                 log_level=None):
        
        self.translator = translator
        self.source_language = source_language
        self.target_language = target_language
        #self.stemmer = SnowballStemmer(target_language)
        self.stemmer = FrenchStemmer()
        self.html_parser = HTMLParser.HTMLParser()
        self.translation_cache_file = translation_cache_file
        self.translation_cache = {}
        
        if translation_cache_file is not None:
            try:
                with open(translation_cache_file, "r") as f:
                    logger.info("Loading translation cache from '{}'".format(translation_cache_file))
                    self.translation_cache = json.load(f)
            except:
                logger.warning("Could not load cache file '{}'".format(translation_cache_file))
        
        if log_level is not None: logger.setLevel(log_level)

    def _save_cache(self):
        if self.translation_cache_file is not None:
            logger.info("Saving translation cache to '{}'".format(self.translation_cache_file))
            try:
                with open(self.translation_cache_file, "w") as f:
                    json.dump(self.translation_cache, f, indent=2)
            except:
                logger.warning("Could not write cache file '{}'".format(self.translation_cache_file))
    
    def translate_text(self, text):
        return self.html_parser.unescape(self.translator.translate(text, self.source_language, self.target_language))
    
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
                                                                    "entity": slot["entity"],
                                                                    "slot_name": slot["slot_name"],
                                                                    "id": slot["id"]
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
            if "slot_name" in data:
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
            logger.info("Unassigned slots")
            logger.info(str(unassigned_slots))
        
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
        
        return assistant_t