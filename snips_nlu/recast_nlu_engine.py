from recast_bench.lib.expression_tools import add_expression
from recast_bench.lib.intent_tools import create_intent, delete_intent, \
    create_entity
from recast_bench.lib.parser_tools import parser

from dataset import validate_and_format_dataset
from snips_nlu.nlu_engine import NLUEngine
from snips_nlu.result import IntentClassificationResult
from snips_nlu.result import ParsedSlot
from snips_nlu.result import Result
from snips_nlu.utils import get_intents_and_entities


class RecastNLUEngine(NLUEngine):
    def __init__(self, language, is_builtin=False, USER_SLUG="choufractal",
                 BOT_SLUG="test",
                 DEVELOPER_TOKEN="03f4cd950c064abf42c79fc22631483e",
                 REQUEST_TOKEN="20e7948d732e36470ba8471f5475d2e1"):
        super(RecastNLUEngine, self).__init__(language)

        self.language = language
        self.USER_SLUG = USER_SLUG
        self.BOT_SLUG = BOT_SLUG
        self.DEVELOPER_TOKEN = DEVELOPER_TOKEN
        self.REQUEST_TOKEN = REQUEST_TOKEN
        self.is_builtin = is_builtin

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        res = parser(text, self.intents, self.entities,
                     self.REQUEST_TOKEN, self.language)

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
            if slot_value not in text:
                raise ValueError("Recast returned unknown entity: %s" % value)
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
        :return: A fitted RasaNLUEngine
        """
        dataset = validate_and_format_dataset(dataset)

        self.intents, self.entities = get_intents_and_entities(dataset)
        for intent in self.intents:
            delete_intent(intent, self.USER_SLUG, self.BOT_SLUG,
                          self.DEVELOPER_TOKEN)
            create_intent(intent, self.USER_SLUG, self.BOT_SLUG,
                          self.DEVELOPER_TOKEN)

        entity_dict = {}
        for entity in self.entities:
            entity_dict[entity] = create_entity(entity, self.USER_SLUG,
                                                self.BOT_SLUG,
                                                self.DEVELOPER_TOKEN)

        for intent in self.intents:
            for query in dataset['intents'][intent]['utterances']:
                if len(query['data']) != 0:
                    add_expression(query, self.is_builtin, intent, entity_dict,
                                   self.USER_SLUG, self.BOT_SLUG,
                                   self.DEVELOPER_TOKEN, self.language)

        return self
