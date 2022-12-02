from dataclasses import dataclass
import pickle
from inference import InferenceWrapper, Analysis
from sample_generation import PairSample, generate_all_samples
from tqdm import tqdm
from LassyExtraction.frontend import serialize_phrase, deserialize_phrase
from LassyExtraction.mill.serialization import serialize_proof, deserialize_proof
from LassyExtraction.mill.types import TypeInference
from LassyExtraction.mill.proofs import Proof

def load_original_parser():
    return InferenceWrapper(weight_path='./data/model_weights.pt',
                            atom_map_path='./data/atom_map.tsv',
                            config_path='./data/bert_config.json',
                            device='cpu')


def load_pair_parser():
    return InferenceWrapper(weight_path='./data/model_22.pt',
                            atom_map_path='./data/atom_map.tsv',
                            config_path='./data/bert_config.json',
                            device='cpu')


@dataclass(frozen=True)
class AnalyzedPairSample:
    sample: PairSample
    analysis: Analysis


def analyze_samples(parser, pair_samples: list[PairSample]) -> list[AnalyzedPairSample]:
    sentence_pairs = [(s.present, s.postsent) for s in pair_samples]
    return list(map(lambda sa: AnalyzedPairSample(*sa), zip(pair_samples, parser.analyze_pairs(sentence_pairs))))


def split_in_batches(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def robust_serialize_proof(proof):
    if isinstance(proof, ValueError):
        return proof
    elif isinstance(proof, TypeInference.TypeCheckError):
        return proof
    else:
        return serialize_proof(proof)


def robust_deserialize_proof(proof):
    if isinstance(proof, ValueError):
        return proof
    elif isinstance(proof, TypeInference.TypeCheckError):
        return proof
    else:
        return deserialize_proof(proof)


def serialize_ap_sample(sample: AnalyzedPairSample):
    return (sample.sample,
            tuple(serialize_phrase(phrase) for phrase in sample.analysis.lexical_phrases),
            robust_serialize_proof(sample.analysis.proof))


def deserialize_ap_sample(sample) -> AnalyzedPairSample:
    pair_sample, lexical_phrases, proof = sample
    analysis = Analysis(lexical_phrases=tuple(deserialize_phrase(phrase) for phrase in lexical_phrases),
                        proof=robust_deserialize_proof(proof))
    return AnalyzedPairSample(sample=pair_sample, analysis=analysis)


def save_ap_samples(ap_samples: list[AnalyzedPairSample], out_path: str):
    with open(out_path, 'wb') as outf:
        pickle.dump(list(map(serialize_ap_sample, ap_samples)), outf)

def save_results(results: list[tuple[AnalyzedPairSample, bool]], out_path: str):
    with open(out_path, 'wb') as outf:
        pickle.dump(list(map(lambda sr: serialize_ap_sample(sr[0], sr[1]), results)), outf)

def load_ap_samples(in_path: str):
    with open(in_path, 'rb') as inf:
        data = pickle.load(inf)
    return list(map(deserialize_ap_sample, data))


def main():
    parser = load_pair_parser()
    all_samples = generate_all_samples()
    sample_batches = split_in_batches(all_samples, 2000)
    sample_analyses = sum([analyze_samples(parser, batch) for batch in tqdm(sample_batches)], [])
    save_ap_samples(sample_analyses, './results/pair_parses_no_context.p')
    # sample_results = get_results(sample_analyses)
    # save_results(sample_results, './results/pair_parses_no_context_results.p')