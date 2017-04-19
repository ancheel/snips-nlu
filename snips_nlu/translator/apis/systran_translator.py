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
        
        url =   "{endpoint}?" \
                "key={key}" \
                "&source={source}" \
                "&target={target}" \
                "&profile={profile}" \
                "&input={input}" \
                "&backTranslation=False" \
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
