import requests

WORD_EMBEDDING_ROOT = u'http://127.0.0.1:5000/word2vec/most_similar'


def _get_most_similar(positives=None, negatives=None, topn=10):
    if positives is None:
        positives = []
    if negatives is None:
        negatives = []
    params = {
        'positive': positives,
        'negative': negatives,
        'topn': topn
    }

    r = requests.get(WORD_EMBEDDING_ROOT, params=params)
    return [{'word': item[0], 'similarity': item[1]} for item in r.json()]


def get_most_similars(text, topn):
    if topn <= 0:
        return []
    words = text.lower().split()
    original_text = ' '.join(words)
    synonyms_per_word = []
    for word in words:
        synonyms = _get_most_similar(positives=[word], topn=topn)
        synonyms_per_word.append(synonyms)

    similar_texts = []
    for i in range(topn):
        similar_text_words = []
        for word_index, synonyms in enumerate(synonyms_per_word):
            if len(synonyms) > i:
                synonym = synonyms[i]['word']
            else:
                synonym = words[word_index]
            similar_text_words.append(synonym)
        similar_text = ' '.join(similar_text_words)
        if similar_text != original_text:
            similar_texts.append(similar_text)

    return similar_texts
