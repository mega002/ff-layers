
import argparse
import html
import json
import os

from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from fairseq.modules.adaptive_input import AdaptiveInput
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax


num_layers = 16


def get_model(model_dir):
    model_path = os.path.join(model_dir, "model.pt")
    model = torch.load(model_path)['model']
    embedding_parameters = {k[len('decoder.embed_tokens.'):]: v for k, v in model.items() if 'embed_tokens' in k}
    embedding = AdaptiveInput(267744, 0, 1024, 4, 1024, [20000, 60000])
    embedding.load_state_dict(embedding_parameters)
    softmax = AdaptiveSoftmax(267744, 1024, [20000, 60000], 0.2, adaptive_inputs=embedding)

    return model, softmax


def load_vocab(model_dir):
    vocab = [['<s>', -1], ['<pad>', -1], ['</s>', -1], ['<unk>', -1]]
    vocab_path = os.path.join(model_dir, "dict.txt")
    with open(vocab_path, "r") as fd:
        vocab.extend([line.strip('\n').split(' ') for line in fd.readlines()])

    token_to_id = {token_cnt[0]: i for i, token_cnt in enumerate(vocab)}

    return vocab, token_to_id


def get_target_counts(catalog_dir):
    target_counts = {}
    for filename in tqdm(os.listdir(catalog_dir)):
        dim, layer = filename.strip('.text').split('_')
        with open(os.path.join(catalog_dir, filename), "r") as fd:
            top_records = [line.split('\t')[1] for line in fd.readlines()]

        # fix bug: taking the one-before-last token, because we are ignoring the BOS token.
        dim_targets = []
        for record in top_records:
            record_tokens = record.split(' ')
            if len(record_tokens) > 1:
                target_token = html.unescape(record_tokens[-2])
                if target_token in ['unk', 'pad']:
                    target_token = '<' + target_token + '>'
                dim_targets.append(target_token)
            else:
                dim_targets.append("<s>")
        dim_target_counts = Counter(dim_targets)

        target_counts[f"{layer}_{dim}"] = {target: cnt for target, cnt in dim_target_counts.most_common()}

    return target_counts


def get_softmax_log_probabilities(model, softmax, weight_coefficient):
    log_probs = []
    for layer_i in tqdm(range(num_layers)):
        layer_fc2_vals = model[f"decoder.layers.{layer_i}.fc2.weight"] * weight_coefficient
        layer_log_probs = softmax.get_log_prob(layer_fc2_vals.T.unsqueeze(0), None).squeeze()
        log_probs.append(layer_log_probs)

    return log_probs


def compare_argmax_w1_w2_all_layers(log_probs, vocab, token_to_id, catalog):
    num_dims = len(log_probs[0])
    num_vocab_ids = len(log_probs[0][0])
    max_probs = []
    argmax_probs = []
    emb_tokens = []
    target_counts = []
    target_rankings = []
    for layer_i, layer_log_probs in tqdm(enumerate(log_probs)):
        layer_log_probs_max = layer_log_probs.max(axis=1)
        layer_probs_max_embs = layer_log_probs_max[1].numpy().tolist()
        max_probs.append(layer_log_probs_max[0].exp().detach().numpy().tolist())
        argmax_probs.append(layer_probs_max_embs)
        layer_emb_tokens = [
            vocab[layer_probs_max_emb][0]
            for layer_probs_max_emb in layer_probs_max_embs
        ]
        emb_tokens.append(layer_emb_tokens)
        layer_target_counts = []
        for dim_i, layer_emb_token in enumerate(layer_emb_tokens):
            layer_dim = f"{layer_i}_{dim_i}"
            if layer_dim not in catalog:
                layer_target_counts.append(-1)
            elif layer_emb_token in catalog[layer_dim]:
                layer_target_counts.append(catalog[layer_dim][layer_emb_token])
            else:
                layer_target_counts.append(0)
        target_counts.append(layer_target_counts)

        layer_log_probs_argsort = torch.argsort(layer_log_probs, axis=1, descending=True).numpy()
        layer_target_rankings = []
        for dim_i in range(num_dims):
            layer_dim = f"{layer_i}_{dim_i}"
            dim_layer_target_rankings = []
            for target_token in catalog[layer_dim]:
                if target_token in token_to_id:
                    target_id = token_to_id[target_token]
                else:
                    target_id = token_to_id["<unk>"]
                if target_id >= num_vocab_ids:
                    print(f"{layer_dim}: token {target_token} out-of-vocab with id {target_id}")
                    continue
                target_id_rank = np.where(layer_log_probs_argsort[dim_i] == target_id)[0][0]
                dim_layer_target_rankings.extend([target_id_rank] * catalog[layer_dim][target_token])
            layer_target_rankings.append(dim_layer_target_rankings)
        target_rankings.append(layer_target_rankings)

    all_dims_data = [
        {
            "layer": layer_i,
            "W2_dim": dim_i,
            "max_prob": max_probs[layer_i][dim_i],
            "embedding_index": argmax_probs[layer_i][dim_i],
            "embedding_token": emb_tokens[layer_i][dim_i],
            "embedding_token_target_count": target_counts[layer_i][dim_i],
            "target_rankings": target_rankings[layer_i][dim_i]
        }
        for layer_i in range(len(max_probs))
        for dim_i in range(len(max_probs[layer_i]))
    ]
    df = pd.DataFrame.from_records(all_dims_data)

    return df, max_probs


def main(args):
    model, softmax = get_model(args.model_dir)
    vocab, token_to_id = load_vocab(args.model_dir)
    catalog = get_target_counts(args.data_dir)

    print("get max probability per W2 column...")
    log_probs = get_softmax_log_probabilities(model, softmax, weight_coefficient=1)
    print("map to tokens and get target rankings in W2 induced probs...")
    df_w1_w2_argmax, max_probs = compare_argmax_w1_w2_all_layers(log_probs, vocab, token_to_id, catalog)

    print("saving results...")
    df_w1_w2_argmax.to_csv(f"{args.output_base}.tsv", sep="\t", index=False)
    json.dump(max_probs,
              open(f"{args.output_base}_max_probs.json", "w"),
              indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='path to a per-key trigger examples directory')
    parser.add_argument('--model_dir', type=str, default='checkpoints/adaptive_lm_wiki103.v2',
                        help='path to model checkpoints directory')
    parser.add_argument('--output_base', type=str, default='',
                        help='path to output path (without a file extension)')

    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.model_dir)

    main(args)
