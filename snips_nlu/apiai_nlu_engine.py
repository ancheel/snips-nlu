from api_bench.lib.intent_tools import create_entity, add_intent
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

        # reinitialize agent
        delete_all_intents(self.developer_token)
        delete_all_entities(self.developer_token)

        # create entities
        mapping_meta = {
            'timeRange': ['@sys.date-time', '@sys.date-time'],
            'location': ['@sys.location', '@sys.location'],
            'action': ['@action', 'action']
        }

        for entity in self.entities:
            if entity not in ['timeRange', 'location']:
                automatedExpansion = dataset['entities'][entity][
                    'automatically_extensible']
                create_entity(entity, automatedExpansion, self.developer_token)

        for intent in self.intents:
            # create UserSays
            userSays = []
            seen_entities = []
            for query in dataset['intents'][intent]['utterances']:
                to_add = []
                for span in query['data']:
                    if 'entity' in span:
                        if span['entity'] not in seen_entities:
                            seen_entities.append(span['entity'])
                        to_add.append(
                            {
                                'text': span['text'],
                                'alias': span['slot_name'],
                                'meta': mapping_meta[span['entity']][0],
                                'userDefined': True
                            }
                        )
                    else:
                        to_add.append({'text': span['text']})
                userSays.append(
                    {
                        'data': to_add,
                        'isTemplate': False,
                        'count': 0
                    }
                )

            # creates intent_parameters
            intent_parameters = [
                {
                    "resetContexts": False,
                    "affectedContexts": [],
                    "parameters": []
                }
            ]
            for entity in seen_entities:
                intent_parameters[0]['parameters'].append(
                    {
                        'name': entity,
                        'dataType': mapping_meta[entity][1],
                        'value': "$" + entity,
                        'auto': True,
                        'isList': True
                    }
                )
            add_intent(intent, userSays, intent_parameters,
                       self.developer_token)

        return self
