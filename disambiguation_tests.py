from inference import InferenceWrapper
from LassyExtraction.frontend import Sample
from dyngraphpn.neural.batching import batchify_encoder_inputs, pair_batchify_encoder_inputs

def load_model():
    return InferenceWrapper(weight_path='./data/model_weights.pt',
                            atom_map_path='./data/atom_map.tsv',
                            config_path='./data/bert_config.json',
                            device='cpu')

sent1 = "De patiënt geneest de dokter."
sent2 = "De dokter geneest de patiënt."
sent3 = "De patiënt die de dokter geneest, was ziek."
sent4 = "De dokter die de patiënt geneest, was ziek."
sent5 = "Ik ben snel naar huis gegaan en toen nog even gaan douchen."
sent6 = "Ze is moe."

def test(model, sentences):
    analyses = model.analyze([sent1, sent2, sent3, sent4])
    for a in analyses:
        print(Sample(a.lexical_phrases, a.proof, "", "").show_term(show_types=False, show_words=True))


model = load_model()
test(model, [sent1, sent2, sent3, sent4])

present1, postsent1 = sent5, sent1
present2, postsent2 = sent6, sent3

# Implement pair batching so that the output for [(present1, postsent1), (present2, postsent2)]
# contains the results of original batching of [postsent1, postsent2]

def test_case(model, present1, postsent1, present2, postsent2):
    sentences = [postsent1, postsent2]
    tokenized_sents, split_sents = zip(*map(model.tokenizer.encode_sentence, sentences))
    token_ids, cluster_ids = zip(*tokenized_sents)
    encoder_batch, sent_lens = batchify_encoder_inputs(token_ids=token_ids,
                                                       token_clusters=cluster_ids,
                                                       pad_token_id=model.tokenizer.core.pad_token_id)

    paired_sentences = [(present1, postsent1), (present2, postsent2)]
    pair_tokenized_sents, pair_split_sents = ...
    pair_token_ids, pair_cluster_ids = ...

    pair_batchify_encoder_inputs()

attention_mask2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
attention_mask3 = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
token_ids2 = torch.tensor([[1, 312, 313, 2], [1, 412, 413, 2]])
token_clusters2 = torch.tensor([[-1, -1, -1, -1], [-1, -1, -1, -1]])

new_attention_mask = torch.cat([attention_mask2, attention_mask], dim=1)
new_token_ids = torch.cat([token_ids2, token_ids], dim=1)
new_token_clusters = torch.cat([token_clusters2, token_clusters], dim=1)

new_attention_mask_back = torch.cat([attention_mask, attention_mask2], dim=1)
new_token_ids_back = torch.cat([token_ids, token_ids2], dim=1)
new_token_clusters_back = torch.cat([token_clusters, token_clusters2], dim=1)

new_attention_mask_back3 = torch.cat([attention_mask, attention_mask3], dim=1)
new_token_ids_back3 = torch.cat([token_ids, token_ids2], dim=1)
new_token_clusters_back3 = torch.cat([token_clusters, token_clusters2], dim=1)

result = encoder.forward(token_ids, attention_mask, token_clusters)
new_result = encoder.forward(new_token_ids, new_attention_mask, new_token_clusters)
new_result_back = encoder.forward(new_token_ids_back, new_attention_mask_back, new_token_clusters_back)

new_results_back3 = encoder.forward(new_token_ids_back3, new_attention_mask_back3, new_token_clusters_back3)


sent5 = "Ik ben snel naar huis gegaan en toen nog even gaan douchen."
sent6 = "Ze is moe."
tok_ids5, cluster_ids5=tokenizer.encode_words(sent5.replace(".", " .").split())
tok_ids6, cluster_ids6=tokenizer.encode_words(sent6.replace(".", " .").split())

tokenizer = Tokenizer('GroNLP/bert-base-dutch-cased', 'bert')
sent2 = "De dokter geneest de patiënt."
sent3 = "De patiënt die de dokter geneest, was ziek."
tok_ids2, cluster_ids2 = tokenizer.encode_words(sent2.replace(".", " .").split())
tok_ids3, cluster_ids3 = tokenizer.encode_words(sent3.replace(".", " .").split())
token_ids_presents, cluster_ids_presents = [tok_ids2, tok_ids3], [cluster_ids2, cluster_ids3]
orig_batch = batchify_encoder_inputs(token_ids_presents, cluster_ids_presents, 3)[0]
new_batch = pair_batchify_encoder_inputs(2*[10*[33]], token_ids_presents, cluster_ids_presents, 3)[0]
token_ids_postsents, cluster_ids_postsents = [tok_ids5, tok_ids6], [cluster_ids5, cluster_ids6]