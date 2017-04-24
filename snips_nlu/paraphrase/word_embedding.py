import requests

WORD_EMBEDDING_ROOT = u'http://127.0.0.1:5000/word2vec/most_similar'

SIMILARITY_CACHE = dict()


def _build_caching_key(positives, negatives, topn):
    pos_key = '|'.join(positives)
    neg_key = '|'.join(negatives)
    return '<+>%s</+><->%s</-><top>%s</top>' % (pos_key, neg_key, topn)


def get_most_similar(positives=None, negatives=None, topn=10):
    if positives is None:
        positives = []
    if negatives is None:
        negatives = []
    params = {
        'positive': positives,
        'negative': negatives,
        'topn': topn
    }

    caching_key = _build_caching_key(positives, negatives, topn)
    if caching_key not in SIMILARITY_CACHE:
        r = requests.get(WORD_EMBEDDING_ROOT, params=params)
        similar_words = [
            {
                'word': ' '.join(item[0].split('_')).lower(),
                'similarity': item[1]
            } for item in r.json()]
        SIMILARITY_CACHE[caching_key] = similar_words
    return SIMILARITY_CACHE[caching_key]


