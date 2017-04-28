# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import urllib
import json
import time

from .abstract_translator import Translator

class SystranTranslator(Translator):
    def __init__(self, auth_file, model="neural", logger=None):
        self.model = model
        self.logger = logger
        
        with open(auth_file) as f:
            self.configuration = json.load(f)
    
    def _debug(self, txt):
        if self.logger is not None:
            self.logger.debug(txt)
    
    def _info(self, txt):
        if self.logger is not None:
            self.logger.info(txt)
    
    def _warn(self, txt):
        if self.logger is not None:
            self.logger.warn(txt)
    
    def _error(self, txt):
        if self.logger is not None:
            self.logger.error(txt)
    
    
    def _mk_query_url(self, text, source_language, target_language):
        
        translation_pair = source_language + target_language
        profile = self.configuration["profiles"][translation_pair][self.model]
        
        query = {
            "key": self.configuration["key"],
            "source": source_language,
            "target": target_language,
            "profile": profile,
            "input": text.encode("iso-8859-15", errors="replace"),
            "backTranslation": "False"
        }
        
        url = self.configuration["endpoint"] + "?" + urllib.urlencode(query)
        
        return url
    
    def _query_service(self, url):
        attempts = 1
        max_attempts = 10
        result = None

        self._debug("Querying {}".format(url))

        while attempts <= max_attempts and result is None:
            try:
                result = urllib.urlopen(url).read()
            except IOError:
                self._debug("Failure #{}".format(attempts))
                if attempts < max_attempts:
                    time.sleep(attempts)
            attempts += 1
        
        if result is None:
            return None
        
        return json.loads(result)
        
    
    def translate(self, text, source_language, target_language):
        url = self._mk_query_url(text, source_language, target_language)
        result = self._query_service(url)
        if result is None: return None
        try:
            return "\n".join( [ res["output"] for res in result["outputs"] ] )
        except:
            raise Exception("Incorrect reply by Systran: {}".format(str(result)))
