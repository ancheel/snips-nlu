# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import urllib
from snips_nlu.tokenization import tokenize
import json
import re
float_re = re.compile(r"-?\d*\.\d*|NaN")

class Embedding():
    
    def __init__(self, endpoint):
        self.endpoint = endpoint
    
    def similarity(self, seq1, seq2):
        
        arg1 = tokenize(seq1) if isinstance(seq1, str) else seq1
        arg2 = tokenize(seq2) if isinstance(seq2, str) else seq2
        
        query_part_1 = "&".join(
            [
                urllib.urlencode( { "ws1": token.encode("iso-8859-15", errors="replace") } )
                for token in arg1
            ]
        )
        query_part_2 = "&".join(
            [
                urllib.urlencode( { "ws2": token.encode("iso-8859-15", errors="replace") } )
                for token in arg2
            ]
        )
        
        url = self.endpoint\
              + "/n_similarity"\
              + "?"\
              + query_part_1\
              + "&"\
              + query_part_2
        
        url = url.lower()
        
        result = urllib.urlopen(url).read()
        
        if float_re.match(result):
            return float(result)
        
        try:
            status = json.loads(result)["status"]
            if status == 500:
                return 0.
            else:
                raise Exception("Embedding server replied with status {}: {}".format(status, result["message"]))
        except:
            raise Exception("Could not get similarity: "+str(result))
        
        return result
