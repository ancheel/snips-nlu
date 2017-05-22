import time
from wit_bench.lib.data_tools import map_data, create_training
from wit_bench.lib.intent_tools import delete_all, create_entity, \
    create_intent, add_samples, train_model
from wit_bench.lib.parser_tools import parser

from dataset import validate_and_format_dataset
from snips_nlu.nlu_engine import NLUEngine
from snips_nlu.result import IntentClassificationResult
from snips_nlu.result import ParsedSlot
from snips_nlu.result import Result
from snips_nlu.utils import get_intents_and_entities


class WitNLUEngine(NLUEngine):
    def __init__(self, language,
                 token="WMZDLM5JZVTIJSLF3652PDINJSX2EDP6"):
        super(WitNLUEngine, self).__init__(language)

        self.token = token

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        res = parser(text, self.token)

        if len(res['intent']) == 0:
            intent_name = None
            prob = None
            #slots = []
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
        delete_all(self.token)

        # create entities
        mapping_builtin = {
            'timeRange': 'wit$datetime'
        }

        for entity in self.entities:
            if entity not in mapping_builtin:
                automatedExpansion = dataset['entities'][entity][
                    'automatically_extensible']
                create_entity(entity, automatedExpansion, self.token)

        create_intent(self.token)

        add_samples(self.intents, mapping_builtin, dataset['intents'],
                    self.token)

        isTrained = train_model()

        if not isTrained:
            print 'Could not train agent after 30 tries'

        return self
