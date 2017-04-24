import importlib
from copy import copy
from random import shuffle

from Levenshtein import ratio
from pattern.text import penntreebank2universal, NOUN, ADJ, NUM, INTJ

from snips_nlu.paraphrase.word_embedding import get_most_similar
from snips_nlu.tokenization import tokenize_light

PARAPHRASE_TAGS = {
    ADJ,
    INTJ,
    NOUN,
    NUM
}


def get_paraphrases(text, language, topn_similarity=10,
                    max_levenshtein_ratio=0.8, min_similarity=0.8, limit=None):
    if topn_similarity <= 0:
        return []
    tokens = tokenize_light(text)
    tags = get_pos_tags(tokens, language)
    synonyms_per_word = []
    for token, tag in tags:
        synonyms = [token]
        if tag in PARAPHRASE_TAGS:
            similar_words = get_most_similar(positives=[token],
                                             topn=topn_similarity)
            synonyms += [
                s['word'] for s in similar_words
                if s['similarity'] >= min_similarity
                   and ratio(s['word'], token.lower()) <= max_levenshtein_ratio
                   and len(s['word'].split()) == 1
            ]
        synonyms_per_word.append(synonyms)

    lattice_indices = _generate_lattice_indices(map(len, synonyms_per_word))

    paraphrases = []
    for indices in lattice_indices:
        similar_text_words = []
        for i, index in enumerate(indices):
            synonym = synonyms_per_word[i][index]
            similar_text_words.append(synonym)
        if similar_text_words != tokens:
            similar_text = ' '.join(similar_text_words)
            paraphrases.append(similar_text)
    paraphrases = shuffle(paraphrases)
    if limit is not None:
        return paraphrases[:limit]
    return paraphrases


def get_pos_tags(tokens, language):
    try:
        pattern_module = importlib.import_module(
            'pattern.text.%s' % language.iso_code)
    except ImportError:
        return [(token, None) for token in tokens]
    tag = getattr(pattern_module, 'tag')
    tags = tag(tokens, tokenize=False)
    universal_tags = [penntreebank2universal(token, pos)
                      for token, pos in tags]
    return universal_tags


def _generate_lattice_indices(lengths):
    indices = [[0 for _ in range(len(lengths))]]
    final_indices = [length - 1 for length in lengths]
    while indices[-1] != final_indices:
        last_indices = indices[-1]
        updated_indices = copy(last_indices)
        for i, index in enumerate(last_indices):
            if index + 1 < lengths[i]:
                updated_indices[i] = index + 1
                break
            else:
                updated_indices[i] = 0
        indices.append(updated_indices)
    return indices
