#!/usr/bin/env python3
# coding=utf-8

import os
import json
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer, models

from qwen3.configuration_qwen3 import Qwen3Config
from qwen3.modeling_qwen3 import Qwen3ForCausalLM
from den2moee.configuration_den2moee import Den2MoEEConfig
from den2moee.modeling_den2moee import Den2MoEEModel, Den2MoEESparseMoeBlock, Den2MoEESvdMLP


def create_den2moee_config(
    dense_config: Qwen3Config,
    num_experts: int,
    top_k: int,
    top_ke: float = 1.8,
    shared_ratio: float = 0.125,
    rank_ratio: float = 0.40,
) -> Den2MoEEConfig:
    den2moee_config_dict = dict(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
        moe_intermediate_size=dense_config.intermediate_size // num_experts,
        n_routed_experts=int(num_experts * (1 - shared_ratio)),
        n_shared_experts=int(num_experts * shared_ratio),
        n_null_experts=1,
        top_ke=top_ke,
        rank_ratio=rank_ratio,
    )
    return Den2MoEEConfig.from_qwen3_config(dense_config, **den2moee_config_dict)


def load_expert_splits(expert_split_dir: str, num_layers: int, num_experts: int, time_stamp: str = None):
    """load expert-splits json for each layer"""
    splits = {}
    for layer_id in range(1, num_layers + 1):
        # match the corresponding layer json file
        fname = [
            f for f in os.listdir(expert_split_dir)
            if f"layer{layer_id}_" in f and f"k{num_experts}_" in f and time_stamp in f and f.endswith(".json")
        ]
        if len(fname) != 1:
            raise ValueError(f"[ERROR] Layer {layer_id} not found unique expert split json, got {fname}")
        fpath = os.path.join(expert_split_dir, fname[0])
        with open(fpath, "r") as f:
            splits[layer_id] = json.load(f)
    return splits


@torch.no_grad()
def init_den2moeesvdmlp_from_dense(
    expert: Den2MoEESvdMLP,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    eps_energy: float = 0,
    save_name: str = None,
    save_dir: str = "tmp/svd_energy",
    rank_ratio = None,
):
    import os
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)

    def _svd_truncate(W: torch.Tensor, eps_energy: float, name: str, rank_ratio: float = None):
        """calculate SVD and save energy distribution to .npy"""
        # torch.linalg.svd output: U [m,r], S [r], Vh [r,n]
        U, Svals, Vt = torch.linalg.svd(W, full_matrices=False)
        sv2 = Svals ** 2
        ratio = torch.cumsum(sv2, dim=0) / (sv2.sum() + 1e-12)

        # === save singular values and energy ratio ===
        save_path = os.path.join(save_dir, f"{save_name}.npy")
        np.save(save_path, {
            "Svals": Svals.cpu().numpy(),
            "energy_ratio": ratio.cpu().numpy(),
            "eps_energy": eps_energy,
            "shape": W.shape,
        })
        print(f"        [saved] Energy ratio → {save_path}")

        # === rank truncation ===
        r = int((ratio < (1 - eps_energy)).sum().item() + 1)
        r = max(1, min(r, min(W.shape)))

        # === calculate the energy ratio of the current r ===
        rt = r
        if rank_ratio:
            rt = int(min(rt, min(W.shape) * rank_ratio))
        total_energy = (Svals ** 2).sum()
        energy_r = (Svals[:rt] ** 2).sum() / total_energy
        energy_r = energy_r.item()  # convert to float
        print(f"        [rank {rt}] energy = {energy_r * 100:.1f}%")

        return U[:, :r], Svals[:r], Vt[:r, :].T, r

    def _assign_svd_linear(svd_linear, W, eps_energy, name, rank_ratio=0.40):
        """copy the truncated SVD parameters to the module"""
        U, Svals, V, r = _svd_truncate(W, eps_energy, name, rank_ratio)
        rank_limit = svd_linear.rank
        r = min(r, rank_limit)

        # truncate
        U = U[:, :r]
        Svals = Svals[:r]
        V = V[:, :r]

        # zero + copy
        svd_linear.U.data.zero_()
        svd_linear.sigma.data.zero_()
        svd_linear.V.data.zero_()
        svd_linear.U.data[:, :r].copy_(U)
        svd_linear.sigma.data[:r].copy_(Svals)
        svd_linear.V.data[:, :r].copy_(V)

        svd_linear.rank = r
        print(f"        [{name}] rank={r}/{min(W.shape)} ({100 * r / min(W.shape):.1f}%)")

    # initialize gate_proj and up_proj
    _assign_svd_linear(expert.gate_proj, W_gate, eps_energy, "gate_proj", rank_ratio)
    _assign_svd_linear(expert.up_proj, W_up, eps_energy, "up_proj", rank_ratio)

    # down_proj copy full weights
    expert.down_proj.weight.data.copy_(W_down)

    print(f"        ✅ initialization completed: gate_rank={expert.gate_proj.rank}, up_rank={expert.up_proj.rank}, down=full")


@torch.no_grad()
def convert_dense_to_den2moee(
    dense_model: Qwen3ForCausalLM,
    expert_split_dir: str,
    activations_path: str,
    num_experts: int = 8,
    top_k: int = 2,
    time_stamp: str = None,
    shared_ratio: float = 0.25,
    rank_ratio: float = 1.0,
    save_dir: str = None,
) -> Den2MoEEModel:
    device = dense_model.device
    den2moee_config = create_den2moee_config(dense_model.config, num_experts, top_k, shared_ratio=shared_ratio, rank_ratio=rank_ratio)
    den2moee_model = Den2MoEEModel(den2moee_config).to(device)

    # === copy non-MLP parameters ===
    state_dict = dense_model.state_dict()
    missing, unexpected = den2moee_model.load_state_dict(state_dict, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    expert_splits = load_expert_splits(
        expert_split_dir,
        num_layers=len(dense_model.model.layers),
        num_experts=num_experts,
        time_stamp=time_stamp,
    )

    # loop through each layer
    for layer_idx, (dense_layer, den2moee_layer) in enumerate(
        tqdm(zip(dense_model.model.layers, den2moee_model.model.layers),
            total=len(dense_model.model.layers),
            desc="converting layers",
            ncols=100),
        start=1
    ):
        print(f"\n[INFO] converting layer {layer_idx} ...")
        den2moee_block = Den2MoEESparseMoeBlock(den2moee_config).to(device)

        gate_w = dense_layer.mlp.gate_proj.weight.data
        up_w   = dense_layer.mlp.up_proj.weight.data
        down_w = dense_layer.mlp.down_proj.weight.data

        # activation matrix
        act_fname_list = [
            f for f in os.listdir(activations_path)
            if f"_layer{layer_idx}.pt" in f and time_stamp in f
        ]
        if len(act_fname_list) != 1:
            raise ValueError(f"[ERROR] Layer {layer_idx} not found unique activation file: {act_fname_list}")
        act_fpath = os.path.join(activations_path, act_fname_list[0])
        sample_inputs = torch.load(act_fpath, map_location=device)
        all_activations = torch.cat([entry["activation"] for entry in sample_inputs], dim=0)
        all_activations = all_activations.to(den2moee_block.gate.weight.dtype)
        
        split_info = expert_splits[layer_idx]
        shared_offset = 0
        routed_count = 0
        routed_budget_scores = []

        for expert_id, expert_entry in split_info.items():
            expert_type = expert_entry["expert_type"]
            neuron_ids = expert_entry["cluster_neuron_ids"]

            if expert_type == "routed":
                expert = den2moee_block.experts[routed_count]
                print(f"    [Expert {routed_count}] routed, {len(neuron_ids)} neurons")

                # dense weights
                W_gate_e = gate_w[neuron_ids, :]
                W_up_e   = up_w[neuron_ids, :]
                W_down_e = down_w[:, neuron_ids]
                
                # initialize SVD MLP
                init_den2moeesvdmlp_from_dense(
                    expert=expert,
                    W_gate=W_gate_e,
                    W_up=W_up_e,
                    W_down=W_down_e,
                    save_dir=save_dir,
                    save_name=f"layer{layer_idx}_expert{routed_count}",
                    rank_ratio=rank_ratio,
                )

                # Router weight_init
                weight_init_i = up_w[neuron_ids, :].mean(dim=0)
                den2moee_block.gate.weight_init[routed_count].copy_(weight_init_i)

                # Budget score
                if "coverage_vector" in expert_entry:
                    cv = expert_entry["coverage_vector"]
                    score = float(sum(cv)) / max(len(cv), 1)
                else:
                    score = 1.0 / den2moee_config.n_routed_experts
                routed_budget_scores.append(score)
                routed_count += 1

            elif expert_type == "shared":
                n = len(neuron_ids)
                den2moee_block.shared_experts.gate_proj.weight.data[shared_offset:shared_offset+n, :].copy_(gate_w[neuron_ids, :])
                den2moee_block.shared_experts.up_proj.weight.data[shared_offset:shared_offset+n, :].copy_(up_w[neuron_ids, :])
                den2moee_block.shared_experts.down_proj.weight.data[:, shared_offset:shared_offset+n].copy_(down_w[:, neuron_ids])
                shared_offset += n
            else:
                raise ValueError(f"Unknown expert_type {expert_type}")

        # initialize Router
        tgt_logit_std = 1e-2
        std = tgt_logit_std / math.sqrt(den2moee_config.hidden_size)
        nn.init.normal_(den2moee_block.gate.weight, mean=0.0, std=std)
        logits = all_activations @ den2moee_block.gate.weight.T
        den2moee_block.gate.correct_bias.copy_(-logits.mean(dim=0))

        # Budget bias
        if routed_budget_scores:
            routed_budget_scores = torch.tensor(
                routed_budget_scores, device=device, dtype=den2moee_block.gate.budget_bias.dtype
            )
            routed_budget_scores = routed_budget_scores / routed_budget_scores.sum().clamp(min=1e-6)
            den2moee_block.gate.budget_score[: len(routed_budget_scores)] = routed_budget_scores
            den2moee_block.gate.budget_bias.zero_()
        else:
            print(f"[WARN] Layer {layer_idx} no routed experts set budget bias")

        den2moee_layer.mlp = den2moee_block

    print("\n[✅] convert_dense_to_den2moee completed: all routed experts initialized with activation weighted SVD.")
    return den2moee_model

def save_model(
    model: Den2MoEEModel, tokenizer: AutoTokenizer, 
    output_dir: str, dense_model_path: str = None,
):

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Den2MoEE model saved to: {output_dir}")

    if dense_model_path is not None:
        source_st = SentenceTransformer(dense_model_path)
        source_pooling = source_st._modules['1']
        word_embedding_model = models.Transformer(output_dir)
        if word_embedding_model.get_word_embedding_dimension() != source_pooling.word_embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"target {word_embedding_model.get_word_embedding_dimension()} vs "
                f"source {source_pooling.word_embedding_dimension}"
            )
        st_model = SentenceTransformer(modules=[word_embedding_model, source_pooling])
        st_model.save(output_dir)
        print(f"[INFO] SentenceTransformer format saved to: {output_dir}")





GY2_PATH = ""
DENSE_MODEL_PATH = f"{GY2_PATH}/Qwen3-Embedding-0.6B"

EXPERT_SPLITS_DIR = ""
ACTIVATIONS_PATH = ""
RANK_ENERGY_PATH = ""
TIME_STAMP = ""
NUM_EXPERTS = 8
TOP_K = 2
RANK_RATIO = 0.40

OUTPUT_DIR = f"{GY2_PATH}/Den2MoEE-Embedding-0.6B-svd-init-0.0-rank-{RANK_RATIO}"

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_model_path", type=str, default=DENSE_MODEL_PATH,
                        help="path: trained Qwen3ForCausalLM model directory")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="path: converted Den2MoEE model output directory")
    parser.add_argument("--expert_split_dir", type=str, default=EXPERT_SPLITS_DIR,
                        help="path: expert-splits json directory for each layer")
    parser.add_argument("--activations_path", type=str, default=ACTIVATIONS_PATH,
                        help="path: activation file for each layer")
    parser.add_argument("--num_experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--time_stamp", type=str, default=TIME_STAMP)
    parser.add_argument("--rank_ratio", type=float, default=RANK_RATIO)
    args = parser.parse_args()

    # load dense model
    print(f"[INFO] loading dense model: {args.dense_model_path}")
    dense_model = Qwen3ForCausalLM.from_pretrained(
        args.dense_model_path,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model_path)

    # convert to Den2MoEE model
    print("[INFO] converting to Den2MoEE model...")
    den2moee_model = convert_dense_to_den2moee(
        dense_model,
        expert_split_dir=args.expert_split_dir,
        activations_path=args.activations_path,
        num_experts=args.num_experts,
        top_k=args.top_k,
        time_stamp=args.time_stamp,
        rank_ratio=args.rank_ratio,
        save_dir=f"{RANK_ENERGY_PATH}/{args.time_stamp}",
    )

    # save HuggingFace format and SentenceTransformer format
    os.makedirs(args.output_dir, exist_ok=True)
    save_model(den2moee_model, tokenizer, args.output_dir, args.dense_model_path)

    # test AutoModelForCausalLM
    print("[INFO] testing AutoModelForCausalLM")
    den2moee_config = AutoConfig.from_pretrained(args.output_dir)
    den2moee_model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    den2moee_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    print(den2moee_config, den2moee_model, den2moee_tokenizer)


if __name__ == "__main__":
    main()

