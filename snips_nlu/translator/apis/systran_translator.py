# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import urllib
import json

from .abstract_translator import Translator


class SystranTranslator(Translator):
    def __init__(self, auth_file, model="neural"):
        self.model = model
        
        with open(auth_file) as f:
            self.configuration = json.load(f)
    
    def _mk_query_url(self, text, source_language, target_language):
        
        translation_pair = source_language + target_language
        profile = self.configuration["profiles"][translation_pair][self.model]
        
        url =   u"{endpoint}?" \
                u"key={key}" \
                u"&source={source}" \
                u"&target={target}" \
                u"&profile={profile}" \
                u"&input={input}" \
                u"&backTranslation=False" \
                .format(endpoint=self.configuration["endpoint"],
                        key=self.configuration["key"],
                        source=source_language,
                        target=target_language,
                        profile=profile,
                        input=text
                        )
        return url
    
    def translate(self, text, source_language, target_language):
        url = self._mk_query_url(text, source_language, target_language)
        result = json.loads(urllib.urlopen(url).read())
        return "\n".join( [ res["output"] for res in result["outputs"] ] )
