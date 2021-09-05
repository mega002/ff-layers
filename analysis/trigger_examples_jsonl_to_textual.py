
import argparse
import shutil
import jsonlines as jl
import os

from tqdm import tqdm

from fairseq.models.transformer_lm import TransformerLanguageModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts the monolithic-file output of generate_analysis (produced when running it with "
                    "--get_trigger_examples), to textual files pertaining to individual dimensions. "
                    "These contain the examples with the highest pattern coefficients for each layer and for each "
                    "dimension chosen for the analysis."
                    "Example: python -m analysis.to_textual --input_file trigger_examples.jsonl "
                    "--output_dir trigger_examples_per_layer_dim"
    )
    parser.add_argument('--input_file', type=str,
                        help='path to file output by generate_analysis (when run with --get_trigger_examples)')
    parser.add_argument('--model_dir', type=str, default='checkpoints/adaptive_lm_wiki103.v2',
                        help='path to model checkpoints dir')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path to output directory. By default, it will be the name of the input file with the'
                             ' ".jsonl" extension replaced with ".outp"')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite output dir without asking')
    args = parser.parse_args()

    if not args.output_dir:
        if not args.input_file.endswith("jsonl"):
           print("please provide an output directory")
           quit()
        args.output_dir = args.input_file.replace(".jsonl", ".outp")

    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            overwrite = input(f"dir {args.output_dir} exists. overwrite? (y/N)  ")
            if overwrite.lower() == 'y':
                shutil.rmtree(args.output_dir)
            else:
                quit()

    os.makedirs(args.output_dir, exist_ok=True)
    analysis_input = list(tqdm(jl.open(args.input_file),
                               total=int(os.popen("wc -l %s" % args.input_file).read().split(" ")[0])))
    en_lm = TransformerLanguageModel.from_pretrained(args.model_dir, 'model.pt', tokenizer='moses')

    for dim_inputs in tqdm(analysis_input):
        for layer_index in list(range(0, 16)):
            dim = dim_inputs['dim']
            topvals = dim_inputs['top_values'][layer_index]
            with open(os.path.join(args.output_dir, f'{dim}_{layer_index}.txt'), 'w') as outf:
                for topval in topvals:
                    assert(topval['layer_index'] == layer_index)
                    text = en_lm.tokenizer.tok.tokenize(topval['text'])[:int(topval['token_indice'])+1]
                    text[-1] = f'*{text[-1]}*'
                    text = ' '.join(text)
                    outf.write(f"{topval['fc1_value']}\t{text}\n")
