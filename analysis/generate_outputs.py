
import argparse
import json
import logging
import os
import random
import spacy
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from tqdm import tqdm

from fairseq.models.transformer_lm import TransformerLanguageModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en")


def parse_line(line):
        tokens = [
            token for token in line.split(' ')
            if token not in ['', '\n']
        ]
        if len(tokens) == 0:
            return None
        spaces = [True for _ in range(len(tokens)-1)] + [False]
        assert len(tokens) == len(spaces), f"{len(tokens)} != {len(spaces)}"

        doc = spacy.tokens.doc.Doc(
            nlp.vocab, words=tokens, spaces=spaces)
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        return [str(sent) for sent in doc.sents]


def parse_data_file(args, shuffle=True):
    data_file, max_sentences, multiprocess = args.data_file, args.max_sentences, args.multiprocess

    parsed = []
    with open(data_file, "r") as fd:
        lines = fd.readlines()
    if shuffle:
        random.seed(0xdead)
        random.shuffle(lines)
    
    pool = mp.Pool(20)
    if max_sentences > -1:
        line_it = pool.imap_unordered(parse_line, lines)
        sentence_pb = tqdm(total=max_sentences)
    else:
        line_it = tqdm(pool.imap_unordered(parse_line, lines), total=len(lines))

    for curr_sentences in line_it:
        if curr_sentences == None:
            continue
        if -1 < max_sentences:
            sentence_pb.update(len(curr_sentences))
        parsed.extend(curr_sentences)
        if -1 < max_sentences <= len(parsed):
            parsed = parsed[:max_sentences]
            pool.terminate()
            break

    logger.info(f"parsed {len(parsed)} sentences")

    return parsed


def format_ffn_values(hypos, sentences, pos_neg, extract_mode, output_values_shape=False):
    for i, hypo in enumerate(hypos):
        if i == 0 and output_values_shape:
            yield (len(hypo['max_fc1_vals']), len(hypo['max_fc1_vals'][0]))

        if extract_mode == "layer-raw":
            yield {
                'pos_neg': pos_neg,
                'text': str(sentences[i]),
                'max_fc1_vals': hypo['max_fc1_vals'],
                'max_pos_fc1_vals': hypo['max_pos_fc1_vals'],
            }
        elif extract_mode == "dim":
            yield json.dumps({
                'pos_neg': pos_neg,
                'text': str(sentences[i]),
                'output_dist_vals': hypo['output_dist_vals'],
                'output_dist_conf': hypo['output_dist_conf'],
                'residual_ffn_output_rank': hypo['residual_ffn_output_rank'],
                'residual_ffn_output_prob': hypo['residual_ffn_output_prob'],
                'residual_ffn_argmax': hypo['residual_ffn_argmax'],
                'residual_ffn_argmax_prob': hypo['residual_ffn_argmax_prob'],
                'ffn_residual_output_rank': hypo['ffn_residual_output_rank'],
                'ffn_residual_output_prob': hypo['ffn_residual_output_prob'],
                'dim_pattern_preds': hypo['dim_pattern_preds'],
                'dim_pattern_output_rank': hypo['dim_pattern_output_rank'],
                'dim_pattern_ffn_output_rank': hypo['dim_pattern_ffn_output_rank'],
                'dim_pattern_ffn_output_prob': hypo['dim_pattern_ffn_output_prob'],
                'coeffs_vals': hypo['coeffs_vals'],
                'coeffs_l0': hypo['coeffs_l0'],
                'coeffs_residual_rank': hypo['coeffs_residual_rank'],
                'random_pos': hypo['random_pos'],
            }) + '\n'
        else:
            assert extract_mode == "layer"
            yield json.dumps({
                'pos_neg': pos_neg,
                'text': str(sentences[i]),
                'output_dist_vals': hypo['output_dist_vals'],
                'layer_output_argmax': hypo['layer_output_argmax'],
                'layer_output_argmax_prob': hypo['layer_output_argmax_prob'],
                'residual_argmax': hypo['residual_argmax'],
                'residual_argmax_prob': hypo['residual_argmax_prob'],
                'residual_argmax_change': hypo['residual_argmax_change'],
                'residual_output_rank': hypo['residual_output_rank'],
                'residual_output_prob': hypo['residual_output_prob'],
                'ffn_matching_dims_count': hypo['ffn_matching_dims_count'],
                'ffn_output_rank': hypo['ffn_output_rank'],
                'ffn_output_prob': hypo['ffn_output_prob'],
                'ffn_residual_output_rank': hypo['ffn_residual_output_rank'],
                'ffn_residual_output_prob': hypo['ffn_residual_output_prob'],
                'ffn_argmax': hypo['ffn_argmax'],
                'ffn_argmax_prob': hypo['ffn_argmax_prob'],
                'coeffs_l0': hypo['coeffs_l0'],
                'coeffs_residual_rank': hypo['coeffs_residual_rank'],
                'random_pos': hypo['random_pos'],
            }) + '\n'


def extract_ffn_info(all_ffn_values, extract_mode, output_file):
    print("load extracted values...")
    records = []
    skip_count = 0
    for i, ffn_vals in tqdm(enumerate(all_ffn_values)):
        loaded_vals = json.loads(ffn_vals)
        random_pos = loaded_vals['random_pos']
        if random_pos == -1:
            skip_count += 1
            continue
        records.append(loaded_vals)

    print(f"skipped {skip_count} examples.")

    # store as dataframe
    df = pd.DataFrame.from_records(records)
    if extract_mode == "dim":
        df = df[
            [col for col in df.columns
             if col in [
                 'text', 'output_dist_vals', 'coeffs_vals', 'residual_ffn_output_rank', 'residual_ffn_output_prob',
                 'residual_output_prob', 'residual_ffn_argmax', 'residual_ffn_argmax_prob',
                 'ffn_residual_output_rank', 'ffn_residual_output_prob',
                 'dim_pattern_preds', 'dim_pattern_output_rank',
                 'dim_pattern_ffn_output_rank', 'dim_pattern_ffn_output_prob', 'random_pos']
             ]
        ]
    else:
        assert extract_mode == "layer"
        df = df[
            [col for col in df.columns
             if col in [
                 'text', 'output_dist_vals', 'layer_output_argmax', 'layer_output_argmax_prob',
                 'residual_argmax', 'residual_argmax_prob', 'residual_argmax_change',
                 'residual_output_rank', 'residual_output_prob', 'ffn_matching_dims_count',
                 'ffn_output_rank', 'ffn_output_prob', 'ffn_residual_output_rank', 'ffn_residual_output_prob',
                 'ffn_argmax', 'ffn_argmax_prob',
                 'coeffs_l0', 'coeffs_residual_rank', 'random_pos']
             ]
        ]
    df.to_pickle(output_file)


def get_trigger_examples(all_ffn_values, dims_for_analysis, num_sentences, values_shape, output_file,
                         top_k=5, apply_relu=True, num_layers=16):
    values_key = 'max_fc1_vals'
    position_key = 'max_pos_fc1_vals'
    hidden_size = values_shape[1]  # shape: (num_layers, hidden_size)
    if args.dims_for_analysis is not None and len(args.dims_for_analysis) > 0:
        assert len([
            dim for dim in dims_for_analysis
            if dim < 0 or dim >= hidden_size
        ]) == 0
    else:
        dims_for_analysis = list(range(hidden_size))
    layers = list(range(num_layers))
    num_dims = len(dims_for_analysis)

    layer_vals = np.zeros((num_layers, top_k, num_dims))
    min_layer_vals_i = np.zeros((num_layers, num_dims), dtype=int)
    token_indices = np.zeros((num_layers, top_k, num_dims))  # token indices
    sentence_indices = np.zeros((num_layers, top_k, num_dims), dtype=int)
    all_ffn_vals = []
    for i, ffn_vals in enumerate(tqdm(all_ffn_values, total=num_sentences)):
        loaded_vals = ffn_vals
        val = loaded_vals.pop(values_key)
        val_pos = loaded_vals.pop(position_key)
        for layer_index in layers:
            for d_i, d in enumerate(dims_for_analysis):
                loc_ind = min_layer_vals_i[layer_index, d_i]
                if val[layer_index][d] > layer_vals[layer_index, loc_ind, d_i]:
                    layer_vals[layer_index, loc_ind, d_i] = val[layer_index][d]
                    token_indices[layer_index, loc_ind, d_i] = val_pos[layer_index][d]
                    sentence_indices[layer_index, loc_ind, d_i] = i
                    min_layer_vals_i[layer_index, d_i] = np.argmin(
                        layer_vals[layer_index, :, d_i])
        all_ffn_vals.append(loaded_vals)

    top_vals_per_dim = []
    for layer_index in layers:
        if apply_relu:
            layer_vals[layer_index] = np.maximum(layer_vals[layer_index], 0)
        else:
            layer_vals[layer_index] = layer_vals[layer_index]
        top_vals_per_dim.append(np.argsort(layer_vals[layer_index], axis=0)[-top_k::][::-1, :].T)

    # write output
    with open(output_file, "w") as fd:
        for dim_i, dim in enumerate(dims_for_analysis):
            dim_outputs = []
            for layer_index in layers:
                layer_output = []
                for rank, i in enumerate(top_vals_per_dim[layer_index][dim_i]):
                    layer_output.append({
                        "rank": rank, 'layer_index': layer_index,
                        'token_indice': token_indices[layer_index][i][dim_i],
                        "fc1_value": layer_vals[layer_index][i][dim_i],
                        "text": all_ffn_vals[sentence_indices[layer_index][i][dim_i]]['text']
                    })
                dim_outputs.append(layer_output)
            fd.write(json.dumps({"dim": dim, "top_values": dim_outputs}) + '\n')


def main(args):
    start_time = time.time()

    en_lm = TransformerLanguageModel.from_pretrained(args.model_dir, 'model.pt', tokenizer='moses')
    en_lm.eval()  # disable dropout
    en_lm.cuda()  # move model to GPU

    parsed = parse_data_file(args)

    def get_hypos(en_lm_, extract_mode):
        for batch_i in tqdm(list(range(0, len(parsed), 1000))):
            for hypo_parsed in en_lm_.score(
                    parsed[batch_i:min(len(parsed), batch_i + 1000)],
                    extract_mode
            ):
                yield hypo_parsed

    all_ffn_values = format_ffn_values(hypos=get_hypos(en_lm, args.extract_mode),
                                       sentences=parsed,
                                       pos_neg=1,
                                       extract_mode=args.extract_mode,
                                       output_values_shape=True)
    values_shape = next(all_ffn_values)

    if args.get_trigger_examples:
        get_trigger_examples(all_ffn_values,
                             dims_for_analysis=args.dims_for_analysis,
                             num_sentences=len(parsed),
                             values_shape=values_shape,
                             output_file=args.output_file,
                             top_k=args.top_k_trigger_examples)
    elif args.extract_ffn_info:
        extract_ffn_info(all_ffn_values,
                         extract_mode=args.extract_mode,
                         output_file=args.output_file)

    print("\n\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,
                        help='path to WikiText-103 format data file')
    parser.add_argument('--model_dir', type=str, default='checkpoints/adaptive_lm_wiki103.v2',
                        help='path to model checkpoints directory')
    parser.add_argument('--output_file', type=str, default='',
                        help='path to output trigger examples or extracted data '
                             '(.jsonl file for trigger examples, .pkl file for ffn info)')
    parser.add_argument('--max_sentences', type=int, default=-1,
                        help='maximum number of sentences to read from data file, -1 for reading all sentences.')
    parser.add_argument("--multiprocess", default=20, type=int,
                        help="how many multiple processes for input parsing")

    parser.add_argument("--get_trigger_examples", action="store_true",
                        help="get trigger examples for keys in the network")
    parser.add_argument('--top_k_trigger_examples', default=50, type=int,
                        help="top input examples to extract per dimension")
    parser.add_argument('--dims_for_analysis', nargs='+', type=int,
                        help='dimensions to get trigger examples for (from all layers), '
                             'if empty then trigger examples will be extracted for all dimensions in the network.')

    parser.add_argument('--extract_ffn_info', action="store_true",
                        help='extract activations and intermediate predictions')
    parser.add_argument("--extract_mode", choices=["dim", "layer", "layer-raw"], default="layer-raw")

    args = parser.parse_args()

    assert os.path.exists(args.data_file)
    assert os.path.exists(args.model_dir)

    if args.extract_ffn_info is True:
        assert args.extract_mode in ["dim", "layer"]
    if args.get_trigger_examples is True:
        assert args.extract_mode == "layer-raw"

    main(args)

