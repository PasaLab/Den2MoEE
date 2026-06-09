import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

DATA_DIR = "tmp_out"
TIME_STAMP = ""
GY2_PATH = ""
MODEL_PATH = ""
CLUSTER_FILE = ""
TOKEN_SCORE_FILE = ""
ACTIVATIONS_DIR = f"{DATA_DIR}/activations"
OUTPUT_DIR = f"{DATA_DIR}/importances"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.eval()
    return model, config

model, config = load_model(MODEL_PATH)
num_layers = config.num_hidden_layers

with open(CLUSTER_FILE, 'r', encoding='utf-8') as f:
    cluster_info_json = json.load(f)


def load_layer_activations(layer_id):
    activations_path = os.path.join(
        ACTIVATIONS_DIR, f"_{TIME_STAMP}_activations_layer{layer_id}.pt"
    )
    if not os.path.exists(activations_path):
        return None
    raw_data = torch.load(activations_path, map_location=device)
    return {item["uid"]: item["activation"].to(torch.float32) for item in raw_data}

# Preprocess TokenScores
print(f"Loading and grouping TokenScores from {TOKEN_SCORE_FILE}...")
token_score_data = torch.load(TOKEN_SCORE_FILE)

domain_to_samples = {}
for item in token_score_data:
    domain = item.get('domain', 'default')
    uid = item['uid']
    t_score = item['token_score'].to(device)
    
    if domain not in domain_to_samples:
        domain_to_samples[domain] = {}
    domain_to_samples[domain][uid] = t_score

all_domains = sorted(list(domain_to_samples.keys()))
domain2idx = {name: i for i, name in enumerate(all_domains)}
num_domains = len(all_domains)

print(f"Detected {num_domains} domains: {all_domains}")

with torch.no_grad():
    for layer_id in tqdm(range(num_layers), desc="Processing layers"):
        uid2activation = load_layer_activations(layer_id + 1)
        if uid2activation is None:
            continue

        mlp_block = model.model.layers[layer_id].mlp
        W_gate = mlp_block.gate_proj.weight.to(device)
        W_up = mlp_block.up_proj.weight.to(device)
        num_neurons = W_gate.shape[0]

        s_layer_d = torch.zeros(num_neurons, num_domains, device=device)
        domain_sample_counts = torch.zeros(num_domains, device=device)

        for domain_name, samples in domain_to_samples.items():
            d_idx = domain2idx[domain_name]
            
            for uid, ts in samples.items():
                if uid not in uid2activation:
                    continue
                
                x = uid2activation[uid].to(W_gate.dtype).to(device)
                if x.dim() == 3: x = x.squeeze(0)

                gate_act = F.silu(x @ W_gate.T)
                up_act = x @ W_up.T
                act_score = torch.abs(gate_act * up_act)

                weighted_sum = (ts.unsqueeze(0) @ act_score).squeeze(0)
                domain_score = weighted_sum / x.size(0)

                s_layer_d[:, d_idx] += domain_score
                domain_sample_counts[d_idx] += 1

        valid_mask = domain_sample_counts > 0
        s_layer_d[:, valid_mask] /= domain_sample_counts[valid_mask]

        mu = s_layer_d.mean(dim=0)     # [num_domains]
        sigma = s_layer_d.std(dim=0)    # [num_domains]
        s_hat = (s_layer_d - mu) / (sigma + 1e-9)

        layer_results = []
        s_hat_cpu = s_hat.cpu().tolist()
        for j in range(num_neurons):
            layer_results.append({
                "layer_id": layer_id + 1,
                "neuron_id": j,
                "domain_names": all_domains,
                "domain_vector": s_hat_cpu[j]
            })
        
        output_path = os.path.join(OUTPUT_DIR, f"neuron_importance_layer{layer_id + 1}_{TIME_STAMP}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(layer_results, f, indent=2, ensure_ascii=False)
        
        print(f"Layer {layer_id + 1}/{num_layers} processed: "
              f"{num_neurons} neurons profiled across {num_domains} domains. "
              f"Saved to: {os.path.basename(output_path)}")