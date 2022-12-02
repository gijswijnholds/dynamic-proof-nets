from disambiguation import AnalyzedPairSample, load_ap_samples
from inference import Analysis
from collections import Counter
from LassyExtraction.mill.nets import proof_to_links
from LassyExtraction.mill.serialization import serialize_proof, deserialize_proof
from LassyExtraction.mill.proofs import Proof
from LassyExtraction.mill.types import TypeInference, Atom



def get_relcl_type(proof: Proof):
    return [c for c in proof.constants() if c[1].index == 2][0][1].type


def relcl_relativity(relcl_type) -> str:
    deco = relcl_type.argument.content.argument.decoration
    if deco == 'su': return "subjrel"
    elif deco == 'obj1': return "objrel"
    else:
        print(f"No proper relcl type: {relcl_type}")
        return ""


def calculated_interpretation(ap_sample: AnalyzedPairSample):
    return relcl_relativity(get_relcl_type(ap_sample.analysis.proof))


def is_faulty_proof(proof: Proof):
    return isinstance(proof, ValueError) or isinstance(proof, TypeInference.TypeCheckError)


def was_correct_parse(ap_sample: AnalyzedPairSample):
    if is_faulty_proof(ap_sample.analysis.proof):
        return False
    elif isinstance(get_relcl_type(ap_sample.analysis.proof).result, Atom):
        return False
    else:
        return ap_sample.sample.interpretation == calculated_interpretation(ap_sample)


def get_svo_types(ap_samples):
    verb_types = Counter(list(map(lambda d: d.analysis.lexical_phrases[-2].type, ap_samples)))
    head_noun_types = Counter(list(map(lambda d: d.analysis.lexical_phrases[1].type, ap_samples)))
    body_noun_types = Counter(list(map(lambda d: d.analysis.lexical_phrases[4].type, ap_samples)))
    relcl_types = Counter(list(map(lambda d: d.analysis.lexical_phrases[2].type, ap_samples)))
    return verb_types, head_noun_types, body_noun_types, relcl_types


def get_unique_proofs(ap_samples):
    return list(map(deserialize_proof, set(map(lambda ap: serialize_proof(ap.analysis.proof), ap_samples))))




def do_check_on_proofs(ap_samples):
    proofs_serialized = map(lambda ap: serialize_proof(ap.analysis.proof), ap_samples)
    proof_counter = Counter(proofs_serialized)
    for p in proof_counter:
        proof = deserialize_proof(p)
        print(proof, '\t', proof_counter[p], '\t', get_relcl_type(proof))


def calculate_accuracy(corrects: list[bool]):
    return sum(corrects) / len(corrects)


def calculate_parse_accuracy(ap_samples):
    return calculate_accuracy(list(map(was_correct_parse, ap_samples)))


def get_results(ap_samples: list[AnalyzedPairSample]):
    return list(map(lambda s: (s, was_correct_parse(s)), ap_samples))


def calculate_result_accuracy(results: list[tuple[AnalyzedPairSample, bool]]):
    return calculate_accuracy([r[1] for r in results])

def calculate_result_percentage_dat(results: list[tuple[AnalyzedPairSample, bool]]):
    return round(100*len([r for r in results if r[0].sample.postsent.split()[2] == 'dat'])/len(results), 2)


def calculate_result_percentage_de(results: list[tuple[AnalyzedPairSample, bool]]):
    return round(100*len([r for r in results if r[0].sample.postsent.split()[0].lower() == 'de'])/len(results), 2)

def filter_results_by_tags(results: list[tuple[AnalyzedPairSample, bool]], data_tag: str, gen_tag: str):
    return [r for r in results if r[0].sample.data_tag == data_tag and r[0].sample.present_tag == "original"
            and r[0].sample.postsent_tag == gen_tag]


def gather_by_category_no_context(results: list[tuple[AnalyzedPairSample, bool]]):
    data_tags = ['irreversible', 'reversible-strong', 'reversible-weak']
    gen_tags = ['original', 'reversed']
    return {data_tag: {gen_tag: filter_results_by_tags(results, data_tag, gen_tag) for gen_tag in gen_tags}
            for data_tag in data_tags}


def analyze_by_category_no_context(results: list[tuple[AnalyzedPairSample, bool]]):
    data_tags = ['irreversible', 'reversible-strong', 'reversible-weak']
    gen_tags = ['original', 'reversed']
    return {data_tag: {gen_tag: calculate_result_accuracy(filter_results_by_tags(results, data_tag, gen_tag)) for gen_tag in gen_tags}
            for data_tag in data_tags}



def main():
    samples = load_ap_samples('./results/pair_parses_no_context.p')
    results = get_results(samples)

    acc = calculate_parse_accuracy(samples)