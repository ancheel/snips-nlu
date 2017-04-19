from google.cloud import translate

from .abstract_translator import Translator


class GcloudTranslator(Translator):

	def __init__(self, model="nmt"):
		self.client = translate.Client()
		self.model = model

	def translate(self, text, source_language , target_language):
		return self.client.translate(text, target_language=target_language, model=self.model)['translatedText']
