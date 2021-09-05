# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torch.nn.functional as F
import sys

from fairseq import utils
import numpy as np


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, eos=None, extract_mode="layer-raw"):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment

        assert extract_mode in ["dim", "layer", "layer-raw"]
        self.extract_mode = extract_mode

    def _aggregate_layer_values(self, all_values, batch_index, start_idx, seq_len):
        avg_layer_vals = []
        max_layer_vals = []
        max_layer_pos = []
        rndpos_layer_vals = []
        rndpos_layer_pos = []

        if seq_len > 0:
            rnd_pos = int(torch.randint(start_idx, start_idx + seq_len, (1,))[0])
            for layer_vals in all_values:
                effective_layer_vals = layer_vals[batch_index][start_idx:start_idx + seq_len]

                # aggregate along the input sequence, i.e. take the average/max value over all input positions.
                # avg_layer_vals dim = max_layer_vals dim = hidden_size
                avg_layer_vals.append(effective_layer_vals.mean(dim=0).cpu().numpy().tolist())

                # For maximum values, take every value and its token position.
                max_vals = effective_layer_vals.max(axis=0)
                max_layer_vals.append(max_vals[0].cpu().numpy().tolist())
                max_layer_pos.append(max_vals[1].cpu().numpy().tolist())

                #For the sparsity experiment: we use a randomly chosen position for all layers
                rndpos_layer_vals.append(effective_layer_vals[rnd_pos].cpu().numpy().tolist())
                rndpos_layer_pos.append((np.ones(effective_layer_vals[rnd_pos].shape, dtype=int)*rnd_pos).tolist())

        return avg_layer_vals , max_layer_vals, max_layer_pos, rndpos_layer_vals, rndpos_layer_pos

    @staticmethod
    def _get_output_dist_conf(all_values, start_idx, seq_len):
        confs = []
        if seq_len > 0:
            # effective_values shape: (seq len, total vocab size)
            effective_values = all_values[start_idx:start_idx + seq_len]
            for i in range(len(effective_values)):
                confs.append(torch.distributions.Categorical(probs=effective_values[i]).entropy().cpu().item())

        return confs

    @staticmethod
    def _get_fc2_params(model, num_layers=16):
        param_dict = {
            name: param.data
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        return [
            param_dict[f"decoder.layers.{layer_i}.fc2.weight"]
            for layer_i in range(num_layers)
        ]

    def _get_layer_residual_scores(self, residual, residual_zero_out_logits, model, output_token_idx):
        layer_output = residual + residual_zero_out_logits
        layer_output_dist = model.get_normalized_probs(
            layer_output.resize_((1, 1, 1, layer_output.shape[0])), log_probs=False
        ).data.squeeze()
        layer_output_dist /= layer_output_dist.sum()  # fix buggy fairseq code
        layer_output_argmax = torch.argmax(layer_output_dist).cpu().item()
        layer_output_argmax_prob = layer_output_dist[layer_output_argmax].flatten().tolist()[0]

        # zero-out residual
        residual_zero_out_output_dist = model.get_normalized_probs(
            residual_zero_out_logits.resize_((1, 1, 1, residual_zero_out_logits.shape[0])), log_probs=False
        ).data.squeeze()
        residual_zero_out_output_dist /= residual_zero_out_output_dist.sum()  # fix buggy fairseq code

        # residual induced distribution
        residual_i_dist = model.get_normalized_probs(
            residual.resize_((1, 1, 1, residual.shape[0])), log_probs=False
        ).data.squeeze()
        residual_i_dist /= residual_i_dist.sum()  # fix buggy fairseq code
        residual_argmax = torch.argmax(residual_i_dist).cpu().item()
        residual_argmax_prob = residual_i_dist[residual_argmax].flatten().tolist()[0]

        residual_i_dist_sorted_dims = torch.argsort(residual_i_dist, descending=True)
        residual_output_rank = (residual_i_dist_sorted_dims == output_token_idx).nonzero().flatten().tolist()[0]
        residual_output_prob = residual_i_dist[output_token_idx].flatten().tolist()[0]

        ffn_argmax = torch.argmax(residual_zero_out_output_dist).cpu().item()
        ffn_argmax_prob = residual_zero_out_output_dist[ffn_argmax].flatten().tolist()[0]
        ffn_dist_sorted_dims = torch.argsort(residual_zero_out_output_dist, descending=True)
        ffn_output_rank = (ffn_dist_sorted_dims == output_token_idx).nonzero().flatten().tolist()[0]
        ffn_output_prob = residual_zero_out_output_dist[output_token_idx].flatten().tolist()[0]
        ffn_residual_output_rank = (ffn_dist_sorted_dims == residual_argmax).nonzero().flatten().tolist()[0]
        ffn_residual_output_prob = residual_zero_out_output_dist[residual_argmax].flatten().tolist()[0]

        residual_argmax_change = \
            torch.argmax(layer_output_dist) != torch.argmax(residual_zero_out_output_dist)
        residual_argmax_change_val = int(residual_argmax_change.cpu().item())

        return (layer_output_argmax, layer_output_argmax_prob,
                residual_argmax, residual_argmax_prob, ffn_argmax, ffn_argmax_prob,
                residual_output_rank, residual_output_prob,
                ffn_output_rank, ffn_output_prob,
                ffn_residual_output_rank, ffn_residual_output_prob,
                residual_argmax_change_val)

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        all_ffn_vals = None
        all_fc1_vals = None
        all_residual_vals = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            features = decoder_out[1] if len(decoder_out) > 1 else None
            if type(features) is dict:
                attn = features.get('attn', None)
                ffn_vals = features.get('ffn_vals', None)
                fc1_vals = features.get('fc1_vals', None)
                residual_vals = features.get('residual_vals', None)
                # lists with num_layers elements,
                # each has a shape of (batch size, seq length, hidden dim)
                ffn_vals = [
                    layer_ffn_vals.transpose(0, 1)
                    for layer_ffn_vals in ffn_vals
                ]
                fc1_vals = [
                    layer_fc1_vals.transpose(0, 1)
                    for layer_fc1_vals in fc1_vals
                ]
                residual_vals = [
                    layer_residual_vals.transpose(0, 1)
                    for layer_residual_vals in residual_vals
                ]
            else:
                attn = features
                ffn_vals = None
                fc1_vals = None
                residual_vals = None

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

            if all_ffn_vals is None:
                all_ffn_vals = [ffn_vals]
            else:
                all_ffn_vals.append(ffn_vals)

            if all_fc1_vals is None:
                all_fc1_vals = [fc1_vals]
            else:
                all_fc1_vals.append(fc1_vals)

            if all_residual_vals is None:
                all_residual_vals = [residual_vals]
            else:
                all_residual_vals.append(residual_vals)

        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        # TODO(mega): handle ensemble of models?
        all_ffn_vals = all_ffn_vals[0]
        all_fc1_vals = all_fc1_vals[0]
        all_residual_vals = all_residual_vals[0]

        if len(models) == 1:
            model = models[0]
            all_fc2_params = self._get_fc2_params(model)
            all_fc2_column_norms = [
                torch.norm(fc2_params, dim=0)
                for fc2_params in all_fc2_params
            ]
        act = torch.nn.ReLU()

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None

            avg_layer_fc1_vals_i, max_layer_fc1_vals_i, max_pos_layer_fc1_vals_i, \
            rndpos_layer_fc1_vals_i, rndpos_pos_layer_fc1_vals_i = self._aggregate_layer_values(
                all_fc1_vals, i, start_idxs[i], tgt_len)

            # TODO(mega): move this into the first loop, as the additional call for the model is not efficient.
            assert len(models) == 1

            decoder_out = model(**net_input)
            hidden_size = len(max_pos_layer_fc1_vals_i[0])

            all_output_dists_i = [
                model.get_normalized_probs(
                    pos_logits.resize_((1, 1, 1, pos_logits.shape[0])), log_probs=False
                ).data.squeeze()
                for pos_logits in decoder_out[0][i]
            ]
            # fix buggy fairseq code
            all_output_dists_i = [
                dist_i / dist_i.sum()
                for dist_i in all_output_dists_i
            ]
            all_output_dists_argmax_i = [
                torch.argmax(output_dist_pos)
                for output_dist_pos in all_output_dists_i
            ]
            output_dist_conf_i = self._get_output_dist_conf(all_output_dists_i, start_idxs[i], tgt_len)
            random_pos_i = random.randint(start_idxs[i], start_idxs[i] + tgt_len)
            output_dist_vals_i = all_output_dists_i[random_pos_i].cpu().numpy().tolist()

            num_layers = 16
            coeffs = [
                act(all_fc1_vals[layer_i][i][random_pos_i])
                for layer_i in range(num_layers)
            ]
            coeffs_vals_i = [
                (coeffs[layer_i] * all_fc2_column_norms[layer_i]).cpu().numpy().tolist()
                for layer_i in range(num_layers)
            ]

            residual_ffn_output_rank_i = []
            residual_ffn_output_prob_i = []
            residual_ffn_argmax_i = []
            residual_ffn_argmax_prob_i = []
            dim_pattern_preds_i = []
            dim_pattern_output_rank_i = []
            dim_pattern_ffn_output_rank_i = []
            dim_pattern_ffn_output_prob_i = []

            # Calculate influence scores for all layers
            #######################################
            all_layer_output_argmax = []
            all_layer_output_argmax_prob = []
            all_residual_i = []
            all_residual_output_rank = []
            all_residual_output_prob = []
            all_ffn_output_rank = []
            all_ffn_output_prob = []
            all_ffn_residual_output_rank = []
            all_ffn_residual_output_prob = []
            all_residual_argmax = []
            all_residual_argmax_prob = []
            all_ffn_argmax = []
            all_ffn_argmax_prob = []
            all_residual_argmax_change = []
            for layer_i in range(num_layers):
                all_residual_i.append(all_residual_vals[layer_i][i][random_pos_i])
                residual_i_norm = torch.norm(all_residual_i[layer_i])
                coeffs_vals_i[layer_i] += [1 * residual_i_norm.cpu().item()]

                layer_output_argmax, layer_output_argmax_prob, \
                residual_argmax, residual_argmax_prob, ffn_argmax, ffn_argmax_prob, \
                residual_output_rank, residual_output_prob, \
                ffn_output_rank, ffn_output_prob, \
                ffn_residual_output_rank, ffn_residual_output_prob, \
                residual_argmax_change = self._get_layer_residual_scores(
                    all_residual_i[layer_i],
                    all_ffn_vals[layer_i][i][random_pos_i],
                    model,
                    all_output_dists_argmax_i[random_pos_i]
                )
                all_layer_output_argmax.append(layer_output_argmax)
                all_layer_output_argmax_prob.append(layer_output_argmax_prob)
                all_residual_argmax.append(residual_argmax)
                all_residual_argmax_prob.append(residual_argmax_prob)
                all_ffn_argmax.append(ffn_argmax)
                all_ffn_argmax_prob.append(ffn_argmax_prob)
                all_residual_output_rank.append(residual_output_rank)
                all_residual_output_prob.append(residual_output_prob)
                all_ffn_output_rank.append(ffn_output_rank)
                all_ffn_output_prob.append(ffn_output_prob)
                all_ffn_residual_output_rank.append(ffn_residual_output_rank)
                all_ffn_residual_output_prob.append(ffn_residual_output_prob)
                all_residual_argmax_change.append(residual_argmax_change)

            # Calculate coefficient scores for all layers
            #######################################
            coeffs_l0_i = []
            coeffs_residual_rank_i = []
            for layer_i in range(num_layers):
                coeffs_l0_i.append(len([y for y in coeffs_vals_i[layer_i] if y > 0]))
                sorted_coef_idx = np.array(coeffs_vals_i[layer_i]).argsort()[::-1]
                coeffs_residual_rank_i.append(int(np.where(sorted_coef_idx == 4096)[0][0]))

            # Calculate FFN per-dimension agreement
            #######################################
            all_ffn_matching_dims_count = []
            all_dim_pattern_embedding = torch.stack(all_fc2_params) * torch.stack(coeffs).view(16, 1, 4096)
            for layer_i in range(num_layers):
                layer_output_pred = all_layer_output_argmax[layer_i]
                layer_dim_pattern_embedding = all_dim_pattern_embedding[layer_i].T
                num_parts = 16
                part_size = int(4096 / num_parts)
                parts_res = []
                for part_i in range(num_parts):
                    start_idx = part_size * part_i
                    end_idx = part_size * (part_i + 1)
                    part_dim_dist = model.get_normalized_probs(
                        layer_dim_pattern_embedding[start_idx:end_idx, :].unsqueeze(0).unsqueeze(0),
                        log_probs=False).data.squeeze()
                    part_dim_preds = torch.argmax(
                        part_dim_dist,
                        dim=-1
                    )
                    parts_res.append(part_dim_preds.cpu())
                    del part_dim_dist
                    torch.cuda.empty_cache()

                dim_pattern_pred = torch.cat(parts_res)
                matching_pred_dims = (dim_pattern_pred == layer_output_pred).sum().cpu().item()
                all_ffn_matching_dims_count.append(matching_pred_dims)

            if self.extract_mode == "dim":
                # Go over all the hidden dimensions.
                #######################################
                for dim in range(hidden_size):
                    # get the weighted pattern embedding at dim, zero it out in the decoder output
                    dim_pattern_embedding = all_fc2_params[-1].T[dim] * coeffs[-1][dim]
                    zero_out_logits = decoder_out[0][i][random_pos_i] - dim_pattern_embedding

                    # re-calculate the output distribution over the vocabulary
                    zero_out_output_dist = model.get_normalized_probs(
                        zero_out_logits.resize_((1, 1, 1, zero_out_logits.shape[0])), log_probs=False
                    ).data.squeeze()
                    zero_out_output_dist /= zero_out_output_dist.sum()  # fix buggy fairseq code

                    # calculate the dimension distribution over the vocabulary
                    dim_pattern_embedding_dist = model.get_normalized_probs(
                        dim_pattern_embedding.resize_((1, 1, 1, dim_pattern_embedding.shape[0])), log_probs=False
                    ).data.squeeze()
                    dim_pattern_embedding_dist /= dim_pattern_embedding_dist.sum()  # fix buggy fairseq code
                    dim_pattern_preds_i.append(torch.argmax(dim_pattern_embedding_dist).cpu().item())

                    dim_pattern_embedding_dist_sorted_dims = torch.argsort(dim_pattern_embedding_dist, descending=True)
                    dim_pattern_output_rank = (
                            dim_pattern_embedding_dist_sorted_dims ==
                            all_output_dists_argmax_i[random_pos_i]
                    ).nonzero().flatten().tolist()[0]
                    dim_pattern_output_rank_i.append(dim_pattern_output_rank)
                    dim_pattern_ffn_output_rank = (
                            dim_pattern_embedding_dist_sorted_dims ==
                            all_ffn_argmax[-1]
                    ).nonzero().flatten().tolist()[0]
                    dim_pattern_ffn_output_rank_i.append(dim_pattern_ffn_output_rank)
                    dim_pattern_ffn_output_prob = dim_pattern_embedding_dist[all_ffn_argmax[-1]].flatten().tolist()[0]
                    dim_pattern_ffn_output_prob_i.append(dim_pattern_ffn_output_prob)

                # Add the values for the residual as an additional dimension
                dim_pattern_preds_i.append(all_residual_argmax[-1])
                dim_pattern_output_rank_i.append(all_residual_output_rank[-1])
                residual_ffn_output_rank_i.append(all_residual_output_rank[-1])
                residual_ffn_output_rank_i.append(all_ffn_output_rank[-1])
                residual_ffn_output_prob_i.append(all_residual_output_prob[-1])
                residual_ffn_output_prob_i.append(all_ffn_output_prob[-1])
                residual_ffn_argmax_i.append(all_residual_argmax[-1])
                residual_ffn_argmax_i.append(all_ffn_argmax[-1])
                residual_ffn_argmax_prob_i.append(all_residual_argmax_prob[-1])
                residual_ffn_argmax_prob_i.append(all_ffn_argmax_prob[-1])

                hypos.append([{
                    'tokens': ref,
                    'score': score_i,
                    'attention': avg_attn_i,
                    'alignment': alignment,
                    'positional_scores': avg_probs_i,
                    'max_fc1_vals': max_layer_fc1_vals_i,
                    'max_pos_fc1_vals': max_pos_layer_fc1_vals_i,
                    'rndpos_fc1_vals': rndpos_layer_fc1_vals_i,
                    'rndpos_pos_fc1_vals': rndpos_pos_layer_fc1_vals_i,
                    'output_dist_vals': output_dist_vals_i,
                    'output_dist_conf': output_dist_conf_i,
                    'residual_ffn_output_rank': residual_ffn_output_rank_i,
                    'residual_ffn_output_prob': residual_ffn_output_prob_i,
                    'residual_ffn_argmax': residual_ffn_argmax_i,
                    'residual_ffn_argmax_prob': residual_ffn_argmax_prob_i,
                    'ffn_residual_output_rank': all_ffn_residual_output_rank[-1],
                    'ffn_residual_output_prob': all_ffn_residual_output_prob[-1],
                    'dim_pattern_preds': dim_pattern_preds_i,
                    'dim_pattern_output_rank': dim_pattern_output_rank_i,
                    'dim_pattern_ffn_output_rank': dim_pattern_ffn_output_rank_i,
                    'dim_pattern_ffn_output_prob': dim_pattern_ffn_output_prob_i,
                    'coeffs_vals': coeffs_vals_i[-1],
                    'coeffs_l0': coeffs_l0_i,
                    'coeffs_residual_rank': coeffs_residual_rank_i,
                    'random_pos': random_pos_i,
                }])

            elif self.extract_mode == "layer-raw":
                hypos.append([{
                    'tokens': ref,
                    'score': score_i,
                    'attention': avg_attn_i,
                    'alignment': alignment,
                    'positional_scores': avg_probs_i,
                    'max_fc1_vals': max_layer_fc1_vals_i,
                    'max_pos_fc1_vals': max_pos_layer_fc1_vals_i,
                    'rndpos_fc1_vals': rndpos_layer_fc1_vals_i,
                    'rndpos_pos_fc1_vals': rndpos_pos_layer_fc1_vals_i,
                }])

            else:
                assert self.extract_mode == "layer"
                hypos.append([{
                    'tokens': ref,
                    'score': score_i,
                    'attention': avg_attn_i,
                    'alignment': alignment,
                    'positional_scores': avg_probs_i,
                    'max_fc1_vals': max_layer_fc1_vals_i,
                    'max_pos_fc1_vals': max_pos_layer_fc1_vals_i,
                    'rndpos_fc1_vals': rndpos_layer_fc1_vals_i,
                    'rndpos_pos_fc1_vals': rndpos_pos_layer_fc1_vals_i,
                    'output_dist_vals': output_dist_vals_i,
                    'layer_output_argmax': all_layer_output_argmax,
                    'layer_output_argmax_prob': all_layer_output_argmax_prob,
                    'residual_argmax': all_residual_argmax,
                    'residual_argmax_prob': all_residual_argmax_prob,
                    'residual_argmax_change': all_residual_argmax_change,
                    'residual_output_rank': all_residual_output_rank,
                    'residual_output_prob': all_residual_output_prob,
                    'ffn_matching_dims_count': all_ffn_matching_dims_count,
                    'ffn_output_rank': all_ffn_output_rank,
                    'ffn_output_prob': all_ffn_output_prob,
                    'ffn_residual_output_rank': all_ffn_residual_output_rank,
                    'ffn_residual_output_prob': all_ffn_residual_output_prob,
                    'ffn_argmax': all_ffn_argmax,
                    'ffn_argmax_prob': all_ffn_argmax_prob,
                    'coeffs_l0': coeffs_l0_i,
                    'coeffs_residual_rank': coeffs_residual_rank_i,
                    'random_pos': random_pos_i,
                }])
        return hypos
