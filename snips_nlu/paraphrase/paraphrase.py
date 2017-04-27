# coding=utf-8
from __future__ import unicode_literals

import operator
from random import choice

import Levenshtein
import spacy
from nltk.corpus import wordnet as wn
from pattern.en import singularize, pluralize

from snips_nlu.constants import TEXT, SLOT_NAME
from snips_nlu.dataset import get_text_from_chunks

NOUN = "NOUN"
ADV = "ADV"
PROPN = "PROPN"
ADJ = "ADJ"

WN_ADJ, WN_ADV, WN_NOUN = 'a', 'r', 'n'

WORDNET = "wordnet"
EMBEDDING = "embedding"

TAGS_TO_PARAPHRASE = {ADJ, NOUN, ADV, PROPN}

SPACY_TO_WORDNET_POS = {ADJ: WN_ADJ, NOUN: WN_NOUN, ADV: WN_ADV,
                        PROPN: WN_NOUN}

print "Loading Spacy..."
nlp = spacy.load("en")
print "Loaded Spacy"

WORDNET_CACHE = dict()


def identity(x):
    return x


def title(x):
    return x.title()


def upper(x):
    return x.upper()


def get_similar(token):
    pos = SPACY_TO_WORDNET_POS[token.pos_]
    key = (token.orth, pos)

    if key not in WORDNET_CACHE:
        word_synsets = wn.synsets(token.norm_, pos)
        if len(word_synsets) == 0:
            WORDNET_CACHE[key] = []
            return WORDNET_CACHE[key]
        is_plural = singularize(token.orth_) != token.orth_
        pluralize_fn = pluralize if is_plural else identity
        if token.orth_.istitle():
            upper_fn = title
        elif token.orth_.istitle():
            upper_fn = upper
        else:
            upper_fn = identity
        reshape_fn = lambda x: pluralize_fn(upper_fn(x))
        # TODO: should we consider more synsets
        word_synset = word_synsets[0]
        variants_with_similarity = [
            (reshape_fn(" ".join(unicode(lemma.name()).split("_"))), 1.0)
            for lemma in word_synset.lemmas()]
        variants = word_synset.hyponyms()
        for v in variants:
            sim = word_synset.path_similarity(v)
            variants_with_similarity += [
                (reshape_fn(" ".join(unicode(lemma.name()).split("_"))), sim)
                for lemma in v.lemmas()]
        WORDNET_CACHE[key] = sorted(variants_with_similarity,
                                    key=operator.itemgetter(1),
                                    reverse=True)
    return WORDNET_CACHE[key]


def get_utterance_paraphrase(utterance_data, max_levenshtein_ratio=0.95,
                             min_similarity=0.5,
                             pos_to_paraphrase=TAGS_TO_PARAPHRASE):
    chunk_slot_name_ranges = []
    current_index = 0
    for chunk in utterance_data:
        end = current_index + len(chunk[TEXT])
        if SLOT_NAME in chunk:
            chunk_slot_name_ranges.append(
                ((current_index, end), chunk[SLOT_NAME]))
        current_index = end
    doc = nlp(get_text_from_chunks(utterance_data))
    in_noun_chunk = [range(chunk.start, chunk.end)
                     for chunk in doc.noun_chunks]
    in_noun_chunk = set(i for r in in_noun_chunk for i in r)
    paraphrased_utterance_data = []
    for i, token in enumerate(doc):
        token_start = token.idx
        token_end = token.idx + len(token)
        if i in in_noun_chunk:
            if token.pos_ in pos_to_paraphrase:
                token_lowered = token.orth_.lower()
                most_similars = get_similar(token)
                most_similar_orths = set([w[0] for w in most_similars
                                          if w[1] > min_similarity])
                similar_with_variations = []
                for s in most_similar_orths:
                    s_lowered = s.lower()
                    if Levenshtein.ratio(token_lowered,
                                         s_lowered) < max_levenshtein_ratio:
                        similar_with_variations.append(s)
                if len(similar_with_variations) > 0:
                    similar = choice(similar_with_variations[:3])
                    paraphrased_token = similar
                else:
                    paraphrased_token = token.orth_
            else:
                paraphrased_token = token.orth_
        else:
            paraphrased_token = token.orth_
        is_in_slot = False
        for ((start, end), slot_name) in chunk_slot_name_ranges:
            if token_start >= start and token_end <= end:
                if len(paraphrased_utterance_data) == 0:
                    paraphrased_utterance_data.append(
                        {TEXT: paraphrased_token, SLOT_NAME: slot_name})
                else:
                    previous_chunk = paraphrased_utterance_data[-1]
                    if SLOT_NAME in previous_chunk and \
                                    previous_chunk[SLOT_NAME] == slot_name:
                        previous_chunk[TEXT] += " %s" % paraphrased_token
                    else:
                        paraphrased_utterance_data.append(
                            {TEXT: paraphrased_token, SLOT_NAME: slot_name})
                is_in_slot = True
                break
        if not is_in_slot:
            if len(paraphrased_utterance_data) == 0:
                paraphrased_utterance_data.append({TEXT: paraphrased_token})
            else:
                previous_chunk = paraphrased_utterance_data[-1]
                if SLOT_NAME in previous_chunk:
                    paraphrased_utterance_data.append(
                        {TEXT: paraphrased_token})
                else:
                    previous_chunk[TEXT] += " %s" % paraphrased_token
    return paraphrased_utterance_data


def get_paraphrase(text, max_levenshtein_ratio=0.95,
                   min_similarity=0.5, pos_to_paraphrase=TAGS_TO_PARAPHRASE):
    doc = nlp(text)
    current_index = 0
    paraphrased = ""
    for chunk in doc.noun_chunks:
        if chunk.start_char > current_index:
            paraphrased += text[current_index:chunk.start_char]
        paraphrased_chunk = []
        for token in chunk:
            if token.pos_ in pos_to_paraphrase:
                token_lowered = token.orth_.lower()
                most_similars = get_similar(token)
                most_similar_orths = set([w[0] for w in most_similars
                                          if w[1] > min_similarity])
                similar_with_variations = []
                for s in most_similar_orths:
                    s_lowered = s.lower()
                    if Levenshtein.ratio(token_lowered,
                                         s_lowered) < max_levenshtein_ratio:
                        similar_with_variations.append(s)
                if len(similar_with_variations) > 0:
                    similar = choice(similar_with_variations[:3])
                    paraphrased_chunk.append(similar)
                else:
                    paraphrased_chunk.append(token.orth_)
            else:
                paraphrased_chunk.append(token.orth_)
        paraphrased += " ".join(paraphrased_chunk)
        current_index = chunk.end_char
    paraphrased += text[current_index:]
    return paraphrased
