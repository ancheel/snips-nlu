from enum import Enum

from utils import classproperty

ISO_CODE, DUCKLING_CODE, NAME = "iso", "duckling_code", "name"


class Language(Enum):
    EN = {ISO_CODE: u"en", DUCKLING_CODE: u"en", NAME: u"english"}
    ES = {ISO_CODE: u"es", DUCKLING_CODE: u"es", NAME: u"spanish"}
    FR = {ISO_CODE: u"fr", DUCKLING_CODE: u"fr", NAME: u"french"}
    DE = {ISO_CODE: u"de", DUCKLING_CODE: u"de", NAME: u"german"}
    KO = {ISO_CODE: u"ko", DUCKLING_CODE: u"ko", NAME: u"korean"}

    @property
    def iso_code(self):
        return self.value[ISO_CODE]

    @property
    def duckling_code(self):
        return self.value[DUCKLING_CODE]

    @classproperty
    @classmethod
    def language_by_iso_code(cls):
        try:
            return cls._language_by_iso_code
        except AttributeError:
            cls._language_by_iso_code = dict()
            for ent in cls:
                cls._language_by_iso_code[ent.iso_code] = ent
        return cls._language_by_iso_code

    @classmethod
    def from_iso_code(cls, iso_code, default=None):
        try:
            ent = cls.language_by_iso_code[iso_code]
        except KeyError:
            if default is None:
                raise KeyError("Unknown iso_code '%s'" % iso_code)
            else:
                return default
        return ent

    @classmethod
    def name_from_iso_code(cls, iso_code, default=None):
        try:
            ent = cls.language_by_iso_code[iso_code]
        except KeyError:
            if default is None:
                raise KeyError("Unknown entity '%s'" % iso_code)
            else:
                return default
        return ent.value[NAME]

    @classproperty
    @classmethod
    def language_by_duckling_code(cls):
        try:
            return cls._language_by_duckling_code
        except AttributeError:
            cls._language_by_duckling_code = dict()
            for ent in cls:
                cls._language_by_duckling_code[ent.duckling_code] = ent
        return cls._language_by_duckling_code

    @classmethod
    def from_duckling_code(cls, duckling_code, default=None):
        try:
            ent = cls.language_by_duckling_code[duckling_code]
        except KeyError:
            if default is None:
                raise KeyError("Unknown duckling_code '%s'" % duckling_code)
            else:
                return default
        return ent
