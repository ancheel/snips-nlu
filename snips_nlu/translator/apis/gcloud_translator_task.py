from google.cloud import translate

from snips_nlu.translator.apis.translator_task import TranslatorTask


class GcloudTranslatorTask(TranslatorTask):

	def __init__(self):
		self.client = translate.Client()


	def translate(self,text, source_language , target_language):
		return self.client.translate(text, target_language=target_language)['translatedText']
