from dataclasses import dataclass
import pickle
from anw_lexicon import CategoryClassifier


@dataclass(frozen=True)
class PairSample:
    subj: str
    verb: str
    obj: str
    present: str
    postsent: str
    data_tag: str # ["irreversible", "reversible-strong", "reversible-weak"]
    present_tag: str  # ["original", "reversed"]
    postsent_tag: str  # ["original", "reversed"]
    interpretation: str # ["subjrel", "objrel"]


def load_svo_data():
    with open('./data/final_triples.p', 'rb') as inf:
        triples = pickle.load(inf)
    return triples


def fill_svo_template(subj: str, verb: str, obj: str):
    return f"{subj.capitalize()} {verb} {obj}."


def fill_relcl_template(head_noun: str, verb: str, body_noun: str):
    relcl = 'die' if head_noun.split()[0] == 'de' else 'dat'
    return f"{head_noun.capitalize()} {relcl} {body_noun} {verb}."


def triple_to_pair_samples(subj: str, verb: str, obj: str, tag: str, classifier: CategoryClassifier):
    subj_infl = f"{classifier.get_determiner(subj)} {subj}"
    obj_infl = f"{classifier.get_determiner(obj)} {obj}"
    verb_infl = classifier.inflect_verb(verb)
    sample1 = PairSample(subj=subj, verb=verb, obj=obj,
                         present=fill_svo_template(subj=subj_infl, verb=verb_infl, obj=obj_infl),
                         postsent=fill_relcl_template(head_noun=subj_infl, verb=verb_infl, body_noun=obj_infl),
                         data_tag=tag, present_tag="original", postsent_tag="original", interpretation="subjrel")
    sample2 = PairSample(subj=subj, verb=verb, obj=obj,
                         present=fill_svo_template(subj=subj_infl, verb=verb_infl, obj=obj_infl),
                         postsent=fill_relcl_template(head_noun=obj_infl, verb=verb_infl, body_noun=subj_infl),
                         data_tag=tag, present_tag="original", postsent_tag="reversed", interpretation="objrel")
    sample3 = PairSample(subj=subj, verb=verb, obj=obj,
                         present=fill_svo_template(subj=obj_infl, verb=verb_infl, obj=subj_infl),
                         postsent=fill_relcl_template(head_noun=subj_infl, verb=verb_infl, body_noun=obj_infl),
                         data_tag=tag, present_tag="reversed", postsent_tag="original", interpretation="objrel")
    sample4 = PairSample(subj=subj, verb=verb, obj=obj,
                         present=fill_svo_template(subj=obj_infl, verb=verb_infl, obj=subj_infl),
                         postsent=fill_relcl_template(head_noun=obj_infl, verb=verb_infl, body_noun=subj_infl),
                         data_tag=tag, present_tag="reversed", postsent_tag="reversed", interpretation="subjrel")
    return sample1, sample2, sample3, sample4


def triples_to_pair_samples(triples: list[tuple[str, str, str]], tag: str, classifier: CategoryClassifier):
    all_pair_samples = [triple_to_pair_samples(subj=s, verb=v, obj=o, tag=tag, classifier=classifier)
                        for (s, v, o) in triples]
    return sum(list(map(list, zip(*all_pair_samples))), [])


def generate_all_samples() -> list[PairSample]:
    all_triples = load_svo_data()
    classifier = CategoryClassifier()
    return sum([triples_to_pair_samples(all_triples[k], k, classifier) for k in all_triples], [])


# example_sample = PairSample(subj='dokter', verb='genezen', obj='patiënt', present="De patiënt geneest de dokter.",
#                             postsent="De dokter die de patiënt geneest.", data_tag="reversible-strong",
#                             present_tag="reversed", postsent_tag="original", correct_interpretation="objrel")