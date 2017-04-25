# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from google.cloud import translate

from .abstract_translator import Translator


class GcloudTranslator(Translator):
    def __init__(self, model="nmt", logger=None):
        self.client = translate.Client()
        self.model = model
        self.logger = logger
    
    def translate(self, text, source_language, target_language):
        self.logger.info("Translating {}->{} \"{}\"".format(source_language, target_language, text))
        t = self.client.translate(text, target_language=target_language, model=self.model)['translatedText']
        self.logger.info("t = \"{}\"".format(t))
        return t
