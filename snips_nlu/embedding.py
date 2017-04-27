# -*- coding: utf-8 -*-

import urllib
from snips_nlu.tokenization import tokenize

class Embedding():
    
    def __init__(self, endpoint):
        self.endpoint = endpoint
    
    def similarity(self, seq1, seq2):
        
        arg1 = tokenize(seq1) if isinstance(seq1, str) else seq1
        arg2 = tokenize(seq2) if isinstance(seq2, str) else seq2
        
        query_part_1 = "&".join( [ urllib.urlencode( { "ws1": token } ) for token in arg1 ] )
        query_part_2 = "&".join( [ urllib.urlencode( { "ws2": token } ) for token in arg2 ] )
        
        url = self.endpoint\
              + "/n_similarity"\
              + "?"\
              + query_part_1\
              + "&"\
              + query_part_2
        
        result = float(urllib.urlopen(url.lower()).read())
        
        return result
