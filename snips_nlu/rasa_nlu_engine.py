import io
import json
import os
import shutil

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.model import Trainer

from dataset import validate_and_format_dataset
from snips_nlu.nlu_engine import NLUEngine
from snips_nlu.result import IntentClassificationResult
from snips_nlu.result import ParsedSlot
from snips_nlu.result import Result
from snips_nlu.utils import transform_to_rasa_format


class RasaNLUEngine(NLUEngine):
    def __init__(self, language, backend):
        super(RasaSpacyNLUEngine, self).__init__(language)
        self.backend = backend
        self.config_file_name = None
        self.interpreter = None
        self.language = language

    def parse(self, text):
        """
        Parse the input text and returns a dictionary containing the most
        likely intent and slots.
        """
        res = self.interpreter.parse(unicode(text))
        intent_res = IntentClassificationResult(res['intent']['name'],
                                                res['intent']['confidence'])
        slots = res['entities']
        valid_slot = []
        for slot in slots:
            slot_value = slot['value']
            slot_name = slot['name']
            match_range = [slot['start'], slot['end']]

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

        # data
        if len(dataset['intents']) < 2:
            raise ValueError(
                "Rasa backend can only be used on a dataset containing multiple"
                " intents")

        dir_name = '__rasa_tmp'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        training_data = transform_to_rasa_format(dataset)
        training_file_name = os.path.join(dir_name, "training_data.json")
        with io.open(training_file_name, 'w', encoding='utf8') as f:
            json.dumps(training_data)

        config_dict = {
            "pipeline": self.backend,
            "mitie_file": "./data/total_word_feature_extractor.dat",
            "path": dir_name,
            "data": training_file_name
        }

        config_file_name = os.path.join(dir_name, "config.json")
        with io.open(config_file_name, 'w', encoding='utf8') as f:
            json.dumps(config_dict)

        trainer = Trainer(RasaNLUConfig(config_file_name))
        trainer.train(training_data)
        trainer.persist(dir_name, persistor=None,
                        create_unique_subfolder=False)

        metadata = Metadata.load(dir_name)

        self.interpreter = Interpreter.load(metadata,
                                            RasaNLUConfig(config_file_name))

        shutil.rmtree(dir_name)

        return self
