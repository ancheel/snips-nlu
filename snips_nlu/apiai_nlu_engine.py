import time

from api_bench.lib.intent_tools import create_entity, add_intent, train_model
from api_bench.lib.intent_tools import delete_all_intents, delete_all_entities
from api_bench.lib.parser_tools import parser

from dataset import validate_and_format_dataset
from snips_nlu.nlu_engine import NLUEngine
from snips_nlu.result import IntentClassificationResult
from snips_nlu.result import ParsedSlot
from snips_nlu.result import Result
from snips_nlu.utils import get_intents_and_entities


class ApiaiNLUEngine(NLUEngine):
    def __init__(self, language,
                 developer_token="8979154f948648e0b67c3f5b13a742f4",
                 request_token="10b5dbbd709a4b9196a74d0d500716bb"):
        super(ApiaiNLUEngine, self).__init__(language)

        self.developer_token = developer_token
        self.request_token = request_token

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        res = parser(text, self.developer_token, self.request_token,
                     language='en')

        if len(res['intent']) == 0:
            intent_name = None
            prob = None
            slots = []
        else:
            intent_name = res['intent']['slug']
            prob = res['intent']['confidence']
            slots = res['entities']

        intent_res = IntentClassificationResult(intent_name, prob)

        valid_slot = []
        for slot in slots:
            slot_value = slot['value']
            slot_name = slot['name']
            match_range = [slot['range'][0], slot['range'][1]]

            s = ParsedSlot(match_range, slot_value, 'whathever', slot_name)

            valid_slot.append(s)

        return Result(text, parsed_intent=intent_res,
                      parsed_slots=valid_slot).as_dict()

    def fit(self, dataset):
        """
        Fit the engine with a dataset and return it
        :param dataset: A dictionary containing data of the custom and builtin 
        intents.
        See https://github.com/snipsco/snips-nlu/blob/develop/README.md for
        details about the format.
        :return: A fitted ApiaiNLUEngine
        """
        dataset = validate_and_format_dataset(dataset)

        self.intents, self.entities = get_intents_and_entities(dataset)

        isTraining = True
        count = 0
        while count < 4:
            count += 1
            if isTraining:
                # reinitialize agent
                delete_all_intents(self.developer_token)
                delete_all_entities(self.developer_token)

                # create entities
                mapping_builtin = {
                    'snips/datetime': '@sys.date-time',
                    'city': '@sys.geo-city-us',
                    'country': '@sys.geo-country',
                    'state': '@sys.geo-state-us',
                    'artist': '@sys.music-artist',
                    'genre': '@sys.music-genre',
                }

                for entity in self.entities:
                    if entity not in mapping_builtin:
                        automatedExpansion = dataset['entities'][entity][
                            'automatically_extensible']
                        create_entity(entity, automatedExpansion,
                                      self.developer_token)

                for intent in self.intents:
                    add_intent(intent, mapping_builtin,
                               dataset['intents'][intent]['utterances'],
                               self.developer_token)

                isTraining = train_model(developer_token)

            else:
                print 'training complete!'
                break

        return self
