import os
anw_path = '/Users/3337545/Data/ANW'


def _load_plain(path: str) -> list[str]:
    return [ln.strip() for ln in open(path, 'r').readlines() if '_' not in ln]


def _load_without_det(path: str) -> list[str]:
    return [ln.strip().split()[1] for ln in open(path, 'r').readlines() if '_' not in ln]


def _load_pairs(path: str) -> list[tuple[str, str]]:
    return [tuple(ln.strip().split()) for ln in open(path, 'r').readlines()]


def add_anw_path(fn):
    return os.path.join(anw_path, fn)

_de_persons = _load_without_det(add_anw_path('de_personen.txt'))
_het_persons = _load_without_det(add_anw_path('het_personen.txt'))

_de_abstract = _load_without_det(add_anw_path('de_abstract.txt'))
_het_abstract = _load_without_det(add_anw_path('het_abstract.txt'))

_de_objects = _load_without_det(add_anw_path('de_zaaknaam.txt'))
_het_objects = _load_without_det(add_anw_path('het_zaaknaam.txt'))

_de_massnouns = _load_without_det(add_anw_path('de_verzamelnaam.txt'))
_het_massnouns = _load_without_det(add_anw_path('het_verzamelnaam.txt'))

_de_plants = _load_without_det(add_anw_path('de_plantnaam.txt'))
_het_plants = _load_without_det(add_anw_path('het_plantnaam.txt'))

_de_substances = _load_without_det(add_anw_path('de_stofnaam.txt'))
_het_substances = _load_without_det(add_anw_path('het_stofnaam.txt'))

_de_animals = _load_without_det(add_anw_path('de_diernaam.txt'))
_het_animals = _load_without_det(add_anw_path('het_diernaam.txt'))

_inflection_pairs = _load_pairs(add_anw_path('sg3_inflections.txt'))


def calculate_overlap(set1, set2):
    return len(set1.intersection(set2))/len(set1.union(set2))

class Lexicon:
    @staticmethod
    def persons():
        return set(_de_persons + _het_persons)

    @staticmethod
    def objects():
        return set(_de_objects + _het_objects)

    @staticmethod
    def abstracts():
        return set(_de_abstract + _het_abstract)

    @staticmethod
    def massnouns():
        return set(_de_massnouns + _het_massnouns)

    @staticmethod
    def animals():
        return set(_de_animals + _het_animals)

    @staticmethod
    def plants():
        return set(_de_plants + _het_plants)

    @staticmethod
    def substances():
        return set(_de_substances + _het_substances)

    @staticmethod
    def all_things():
        return set(_de_persons + _het_persons + _de_objects + _het_objects +
                   _de_abstract + _het_abstract + _de_massnouns + _het_massnouns +
                   _de_animals + _het_animals + _de_plants + _het_plants +
                   _de_substances + _het_substances)

    @staticmethod
    def de_words():
        return set(_de_persons + _de_objects + _de_abstract +_de_massnouns +
                   _de_animals + _de_plants + _de_substances)

    @staticmethod
    def het_words():
        return set(_het_persons + _het_objects + _het_abstract + _het_massnouns +
                   _het_animals + _het_plants + _het_substances)

    @staticmethod
    def inflecter_map():
        return dict(_inflection_pairs)


class CategoryClassifier:
    def __init__(self):
        lexicon = Lexicon()
        self.persons = lexicon.persons()
        self.objects = lexicon.objects()
        self.substances = lexicon.substances()
        self.abstracts = lexicon.abstracts()
        self.animals = lexicon.animals()
        self.plants = lexicon.plants()
        self.massnouns = lexicon.massnouns()
        self.de_words = lexicon.de_words()
        self.het_words = lexicon.het_words()
        self.inflecter = lexicon.inflecter_map()

    def classify(self, word):
        if word in self.persons: return "person"
        elif word in self.animals: return "animal"
        elif word in self.plants: return "plant"
        elif word in self.substances: return "substance"
        elif word in self.objects: return "object"
        elif word in self.massnouns: return "massnouns"
        elif word in self.abstracts: return "abstract"
        else:
            print("AAAAH")
            return ""

    def same_class(self, word1, word2):
        return self.classify(word1) == self.classify(word2)

    def get_determiner(self, word):
        if word in self.de_words: return "de"
        elif word in self.het_words: return "het"
        else:
            print("Didn't find a determiner!!")
            return ""

    def inflect_verb(self, verb):
        return self.inflecter[verb]