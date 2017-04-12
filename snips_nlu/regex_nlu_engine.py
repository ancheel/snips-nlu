from dataset import validate_and_format_dataset

from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.nlu_engine import NLUEngine, snips_nlu_entities, _parse


class RegexNLUEngine(NLUEngine):
    def __init__(self, language, custom_parsers=None, entities=None):
        super(RegexNLUEngine, self).__init__(language)
        self.custom_parsers = custom_parsers
        self.entities = entities

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        if self.custom_parsers is None:
            raise ValueError("NLUEngine as no built-in parser nor "
                             "custom parsers")
        parsers = []
        if self.custom_parsers is not None:
            parsers += self.custom_parsers

        return _parse(text, parsers, self.entities).as_dict()

    def fit(self, dataset):
        """
        Fit the engine with a dataset and return it
        :param dataset: A dictionary containing data of the custom and builtin 
        intents.
        See https://github.com/snipsco/snips-nlu/blob/develop/README.md for
        details about the format.
        :return: A fitted SnipsNLUEngine
        """
        dataset = validate_and_format_dataset(dataset)
        custom_parser = RegexIntentParser().fit(dataset)
        self.entities = snips_nlu_entities(dataset)

        self.custom_parsers = [custom_parser]

        return self
