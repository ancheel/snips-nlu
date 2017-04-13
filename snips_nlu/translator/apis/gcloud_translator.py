from google.cloud import translate

from .abstract_translator import Translator


class GcloudTranslator(Translator):

	def __init__(self):
		self.client = translate.Client()


	def translate(self, text, source_language , target_language):
		return self.client.translate(text, target_language=target_language)['translatedText']
