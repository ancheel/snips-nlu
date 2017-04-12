from abc import ABCMeta, abstractmethod

from dataset import validate_and_format_dataset, filter_dataset
from snips_nlu.built_in_entities import BuiltInEntity
from snips_nlu.constants import (
    USE_SYNONYMS, SYNONYMS, DATA, INTENTS, ENTITIES, UTTERANCES,
    LANGUAGE, VALUE, AUTOMATICALLY_EXTENSIBLE, ENTITY, BUILTIN_PARSER,
    CUSTOM_PARSERS, CUSTOM_ENGINE)
from snips_nlu.intent_classifier.snips_intent_classifier import \
    SnipsIntentClassifier
from snips_nlu.intent_parser.builtin_intent_parser import BuiltinIntentParser
from snips_nlu.intent_parser.crf_intent_parser import CRFIntentParser
from snips_nlu.intent_parser.regex_intent_parser import RegexIntentParser
from snips_nlu.languages import Language
from snips_nlu.result import ParsedSlot
from snips_nlu.result import Result
from snips_nlu.slot_filler.crf_tagger import CRFTagger, default_crf_model
from snips_nlu.slot_filler.crf_utils import TaggingScheme
from snips_nlu.slot_filler.feature_functions import crf_features
from snips_nlu.utils import instance_from_dict


class RasaNLUEngine(NLUEngine):
    def __init__(self, backend="spacy_sklearn", language, builtin_parser=None, custom_parsers=None,
                 entities=None):
        super(RasaNLUEngine, self).__init__(language)

        from rasa_nlu.converters import load_data
        from rasa_nlu.config import RasaNLUConfig
        from rasa_nlu.model import Trainer

        self._builtin_parser = None
        self.builtin_parser = builtin_parser
        self.custom_parsers = custom_parsers
        self.entities = entities

    @property
    def builtin_parser(self):
        return self._builtin_parser

    @builtin_parser.setter
    def builtin_parser(self, value):
        if value is not None \
                and value.parser.language != self.language.iso_code:
            raise ValueError(
                "Built in parser language code ('%s') is different from "
                "provided language code ('%s')"
                % (value.parser.language, self.language.iso_code))
        self._builtin_parser = value

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        if self.builtin_parser is None and self.custom_parsers is None:
            raise ValueError("NLUEngine as no built-in parser nor "
                             "custom parsers")
        parsers = []
        if self.custom_parsers is not None:
            parsers += self.custom_parsers
        if self.builtin_parser is not None:
            parsers.append(self.builtin_parser)

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
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE)
        custom_parser = RegexIntentParser().fit(dataset)
        self.entities = snips_nlu_entities(dataset)
        taggers = dict()
        for intent in custom_dataset[INTENTS].keys():
            intent_custom_entities = get_intent_custom_entities(custom_dataset,
                                                                intent)
            features = crf_features(intent_custom_entities,
                                    language=self.language)
            taggers[intent] = CRFTagger(default_crf_model(), features,
                                        TaggingScheme.BIO, self.language)
        intent_classifier = SnipsIntentClassifier(self.language)
        crf_parser = CRFIntentParser(intent_classifier, taggers).fit(dataset)
        self.custom_parsers = [custom_parser, crf_parser]
        return self

    def to_dict(self):
        """
        Serialize the SnipsNLUEngine to a json dict, after having reset the
        builtin intent parser. Thus this serialization, contains only the
        custom intent parsers.
        """
        language_code = None
        if self.language is not None:
            language_code = self.language.iso_code

        return {
            LANGUAGE: language_code,
            CUSTOM_PARSERS: [p.to_dict() for p in self.custom_parsers],
            BUILTIN_PARSER: None,
            ENTITIES: self.entities
        }

    @classmethod
    def load_from(cls, language, customs=None, builtin_path=None,
                  builtin_binary=None):
        """
        Create a `SnipsNLUEngine` from the following attributes

        :param language: ISO 639-1 language code or Language instance
        :param customs: A `dict` containing custom intents data
        :param builtin_path: A directory path containing builtin intents data
        :param builtin_binary: A `bytearray` containing builtin intents data
        """

        if isinstance(language, (str, unicode)):
            language = Language.from_iso_code(language)

        custom_parsers = None
        entities = None
        if customs is not None:
            custom_parsers = [instance_from_dict(d) for d in
                              customs[CUSTOM_PARSERS]]
            entities = customs[ENTITIES]
        builtin_parser = None
        if builtin_path is not None or builtin_binary is not None:
            builtin_parser = BuiltinIntentParser(language=language,
                                                 data_path=builtin_path,
                                                 data_binary=builtin_binary)

        return cls(language, builtin_parser=builtin_parser,
                   custom_parsers=custom_parsers, entities=entities)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.to_dict() == other.to_dict()

    def __ne__(self, other):
        return not self.__eq__(other)
