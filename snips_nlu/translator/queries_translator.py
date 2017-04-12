from progressbar import ProgressBar
from ..tokenization import tokenize
from ..languages import Language
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import FrenchStemmer
import HTMLParser
import logging


def _assemble_query(utterance):
    return "".join(part["text"] for part in utterance["data"])

def _find_in_seq_with_map(sub, seq, mapping):
    for i, _ in enumerate(seq):
        if i + len(sub) <= len(seq) and not any(mapping[i:i + len(sub)]):
            if seq[i:i + len(sub)]:
                return i
    return -1

def _same_slot(el1, el2):
    return el1.get("slot_name") == el2.get("slot_name") and el1.get("entity") == el2.get("entity")

class AssistantTranslator():

    def __init__(self, TranslatorTask, source_language, target_language, log_level=logging.INFO):
        self.translatorTask = TranslatorTask
        self.source_language = source_language
        self.target_language = target_language
        #self.stemmer = SnowballStemmer(target_language)
        self.stemmer = FrenchStemmer()
        self.html_parser = HTMLParser.HTMLParser()
        self.phrase_translation_map = {}
        self.logger = logging.getLogger("Translator")
        self.logger.setLevel(log_level)


    def translate_text(self, text):
        return self.html_parser.unescape(self.translatorTask.translate(text, self.source_language, self.target_language))
    
    def translate_phrase(self, phrase):
        if phrase in self.phrase_translation_map:
            return self.phrase_translation_map[phrase]
        else:
            phrase_t = self.translate_text(phrase)
            self.phrase_translation_map[phrase] = phrase_t
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
        stems = [ self.stem_text(token["value"]) for token in tokens ]
        slots_map = [ False ] * len(stems)
        
        # map stemmed slots to positions in stemmed tokenized text
        for slot in sorted(slots, key = lambda el: len(el["text"]), reverse=True):
            tokenized_slot = tokenize(slot["text"])
            stemmed_slot = [ self.stem_text(token["value"]) for token in tokenized_slot ]
            position = _find_in_seq_with_map(stemmed_slot, stems, slots_map)
            if position >= 0 and not any(slots_map[position:position + len(stemmed_slot)]):
                slots_map[position:position+len(stemmed_slot)] = [ {"text": token["value"],
                                                                    "entity": slot["entity"],
                                                                    "slot_name": slot["slot_name"]
                                                                    }
                                                                   for token in tokenized_slot ]
            else:
                unassigned_slots.append(slot)
        
        for i, token in enumerate(tokens):
            if slots_map[i] is None:
                slots_map[i] = { "text": token["value"] }
        
        # merge sequences of tokens of same type
        utt = [ slots_map[0] ]
        for i, token in enumerate(slots_map):
            if _same_slot(utt[-1], token):
                utt[-1]["text"] += " " + token["text"]
            else:
                utt.append(token)
                if "entity" in token:
                    utt[-1]["entity"] = token["entity"]
                if "slot_name" in token:
                    utt[-1]["slot_name"] = token["slot_name"]
                
            
        
        return utt, unassigned_slots

    def _map_slots(self, text, slots):


        utt, unassigned_slots = self._map_stemmed_slots(text, slots)
        
        return {"data": utt}, unassigned_slots
        
    def translate_slots(self, utt):
        slots_t = []
        
        for data in utt["data"]:
            if "slot_name" in data:
                data["text"] = self.translate_phrase(data["text"])
                slots_t.append(data)
        
        return slots_t

    def translate_utterance(self, utt):
        language = Language(self.target_language)
        
        slots_t = self.translate_slots(utt)
        
        query = _assemble_query(utt)
        query_t = self.translate_text(query)
        
        utt_t, unassigned_slots = self._map_slots(query_t, slots_t)
        
        if len(unassigned_slots) > 0:
            self.logger.info("Unassigned slots")
            self.logger.info(str(unassigned_slots))
        
        return utt_t
    
    def translate_entities(self, assistant):
        self.logger.info("Translating entities")
        pbar = ProgressBar()
        for e_data in pbar(assistant["entities"].values()):
            e_data["data"]["value"] = self.translate_phrase(e_data["data"]["value"])
            e_data["data"]["synonyms"] = [ self.translate_phrase(phrase) for phrase in e_data["data"]["synonyms"] ]
            
        return assistant

    def translate_utterances(self, assistant):
        self.logger.info("Translating intents")
        pbar = ProgressBar()
        for u_data in pbar(assistant["intents"].values()):
            u_data["utterances"] = [ self.translate_utterance(utt) for utt in u_data["utterances"] ]
    
        return assistant

    def translate_assistant(self, assistant):
        
        assistant_t = self.translate_entities   (assistant)
        assistant_t = self.translate_utterances (assistant_t)
        
        return assistant_t
