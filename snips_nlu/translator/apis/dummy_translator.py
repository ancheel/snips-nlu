from .abstract_translator import Translator


class DummyTranslator(Translator):

	def translate(self, text, source_language , target_language):
		#return "TRANSLATED_{}".format(text)
		return text.upper()
