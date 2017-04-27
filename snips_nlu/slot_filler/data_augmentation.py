import random
from copy import deepcopy
from itertools import cycle

import numpy as np

from snips_nlu.constants import (UTTERANCES, DATA, ENTITY, USE_SYNONYMS,
                                 SYNONYMS, VALUE, TEXT, INTENTS, ENTITIES,
                                 AUTOMATICALLY_EXTENSIBLE)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.intent_classifier.intent_classifier_resources import \
    get_subtitles
from snips_nlu.paraphrase.paraphrase import get_utterance_paraphrase, \
    get_paraphrase
from snips_nlu.tokenization import tokenize


def generate_utterance(contexts_iterator, entities_iterators, noise_iterator,
                       noise_prob):
    context = deepcopy(next(contexts_iterator))
    context_data = []
    for i, chunk in enumerate(context[DATA]):
        if ENTITY in chunk:
            has_entity = True
            new_chunk = dict(chunk)
            new_chunk[TEXT] = deepcopy(
                next(entities_iterators[new_chunk[ENTITY]]))
            context_data.append(new_chunk)
        else:
            has_entity = False
            context_data.append(chunk)

        last_chunk = i == len(context[DATA]) - 1
        space_after = ""
        if not last_chunk and ENTITY in context[DATA][i + 1]:
            space_after = " "

        space_before = " " if has_entity else ""

        if noise_prob > 0 and random.random() < noise_prob:
            noise = deepcopy(next(noise_iterator))
            context_data.append({"text": space_before + noise + space_after})
    context[DATA] = context_data
    return context


def get_contexts_iterator(intent_utterances, language, augmentation_ratio):
    augmented_utterances = []
    for utterance in intent_utterances:
        paraphrased_utterance_data = get_utterance_paraphrase(utterance[DATA])
        utterance_text = get_text_from_chunks(utterance[DATA])
        augmented_utterance_text = get_text_from_chunks(
            paraphrased_utterance_data)
        if augmented_utterance_text != utterance_text:
            augmented_utterances.append({DATA: paraphrased_utterance_data})
    shuffled_utterances = np.random.permutation(
        intent_utterances + augmented_utterances)
    return cycle(shuffled_utterances)


def get_entities_iterators(dataset, language, intent_entities,
                           augmentation_ratio):
    entities_its = dict()
    for entity in intent_entities:
        if dataset[ENTITIES][entity][USE_SYNONYMS]:
            values = [s for d in dataset[ENTITIES][entity][DATA] for s in
                      d[SYNONYMS]]
        else:
            values = [d[VALUE] for d in dataset[ENTITIES][entity][DATA]]
        paraphrases = []
        if dataset[ENTITIES][entity][AUTOMATICALLY_EXTENSIBLE]:
            for value in values:
                paraphrases.append(get_paraphrase(value))
        values += paraphrases
        shuffled_values = np.random.permutation(values)
        entities_its[entity] = cycle(shuffled_values)
    return entities_its


def get_intent_entities(dataset, intent_name):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent_name][UTTERANCES]:
        for chunk in utterance[DATA]:
            if ENTITY in chunk:
                intent_entities.add(chunk[ENTITY])
    return intent_entities


def get_noise_iterator(language, min_size, max_size):
    subtitles = get_subtitles(language)
    subtitles_it = cycle(np.random.permutation(list(subtitles)))
    for subtitle in subtitles_it:
        size = random.choice(range(min_size, max_size + 1))
        tokens = tokenize(subtitle)
        while len(tokens) < size:
            tokens += tokenize(next(subtitles_it))
        start = random.randint(0, len(tokens) - size)
        yield " ".join(t.value.lower() for t in tokens[start:start + size])


def augment_utterances(dataset, intent_name, language, max_utterances,
                       noise_prob, min_noise_size, max_noise_size,
                       paraphrasing_factor):
    utterances = dataset[INTENTS][intent_name][UTTERANCES]
    if max_utterances < len(utterances):
        return utterances

    num_to_generate = max_utterances - len(utterances)
    contexts_it = get_contexts_iterator(utterances, language,
                                        paraphrasing_factor)
    noise_iterator = get_noise_iterator(language, min_noise_size,
                                        max_noise_size)
    intent_entities = get_intent_entities(dataset, intent_name)
    entities_its = get_entities_iterators(dataset, language, intent_entities,
                                          paraphrasing_factor)

    while num_to_generate > 0:
        utterances.append(generate_utterance(contexts_it, entities_its,
                                             noise_iterator, noise_prob))
        num_to_generate -= 1

    return utterances
