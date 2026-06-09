import torch
import argparse
import os
import uuid
from transformers import AutoModelForCausalLM
from tqdm import tqdm


OUTPUT_DIR = "tmp_out"
TIME_STAMP = ""
FORWARD_DATA_FILE = f"{OUTPUT_DIR}/Qwen3-Embedding-0.6B_domain_{TIME_STAMP}_forward.pt"
MODEL_PATH = "Qwen3-0.6B"

def get_args():
    parser = argparse.ArgumentParser(description='Calculate TokenScore (SCS + RSS)')
    parser.add_argument('--forward_data_file', type=str, default=FORWARD_DATA_FILE)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--span_size', type=int, default=64)
    parser.add_argument('--ngram', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    return parser.parse_args()

def calculate_scs(summed_attention, span_size=64):
    seq_len = summed_attention.size(0)
    num_spans = (seq_len + span_size - 1) // span_size
    
    saa_scores = torch.zeros(num_spans)
    for i in range(num_spans):
        start = i * span_size
        end = min((i + 1) * span_size, seq_len)
        saa_scores[i] = summed_attention[start:end].sum()
    
    denom = saa_scores.max() - saa_scores.min()
    scs = (saa_scores - saa_scores.min()) / (denom + 1e-9)
    return scs

@torch.no_grad()
def calculate_rss(model, input_ids, attention_mask, top_span_indices, span_size=64, ngram=8, stride=4):
    model.eval()
    device = model.device
    seq_len = input_ids.size(0)
    rss_token_sums = torch.zeros(seq_len).to(device)
    rss_token_counts = torch.zeros(seq_len).to(device)

    with torch.no_grad():
        orig_out = model(input_ids.unsqueeze(0).to(device), 
                         attention_mask=attention_mask.unsqueeze(0).to(device),
                         output_hidden_states=True)
        phi_X = orig_out.hidden_states[-1][0]
        norm_phi_X = torch.norm(phi_X, p=2)

    allowed_token_starts = set()
    for s_idx in top_span_indices:
        start_bound = s_idx * span_size
        end_bound = (s_idx + 1) * span_size
        for i in range(start_bound, end_bound):
            allowed_token_starts.add(i)

    for start in range(0, seq_len - ngram + 1, stride):
        if start not in allowed_token_starts:
            continue

        end = start + ngram
        g_indices = list(range(start, end))
        perturbed_input_ids = input_ids.clone().to(device)
        perturbed_input_ids[g_indices] = 0 
        
        with torch.no_grad():
            perturbed_out = model(perturbed_input_ids.unsqueeze(0), 
                                 attention_mask=attention_mask.unsqueeze(0).to(device),
                                 output_hidden_states=True)
            phi_X_g = perturbed_out.hidden_states[-1][0]
        
        diff_norm = torch.norm(phi_X - phi_X_g, p=2)
        rss_g = diff_norm / (norm_phi_X + torch.norm(phi_X_g, p=2) + 1e-9)
        
        rss_token_sums[g_indices] += rss_g
        rss_token_counts[g_indices] += 1

    rss_tokens = rss_token_sums / (rss_token_counts + 1e-9)
    return rss_tokens.cpu()


def main():
    args = get_args()
    
    act_dir = os.path.join(args.output_dir, "activations")
    score_dir = os.path.join(args.output_dir, "token_scores")
    os.makedirs(act_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    print(f"📂 Loading: {args.forward_data_file}")
    loaded_outputs = torch.load(args.forward_data_file)
    
    print(f"🤖 Loading Model for RSS: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )

    all_results = []
    num_layers = 29 
    layer_activations = [[] for _ in range(num_layers)]

    for sample in tqdm(loaded_outputs, desc="Processing TokenScores"):
        uid = sample['uid']
        input_ids = sample['input_ids']
        attn_mask = sample['attention_mask']
        domain = sample.get('domain', 'default')
        
        scs_vector = calculate_scs(sample['last_layer_attention'], args.span_size)

        num_spans = scs_vector.size(0)
        k = max(1, num_spans // 2)
        _, top_indices = torch.topk(scs_vector, k)
        top_indices = top_indices.tolist()
    
        rss_vector = calculate_rss(model, input_ids, attn_mask, top_indices, args.span_size, args.ngram, args.stride)
        
        seq_len = input_ids.size(0)
        token_scores = torch.zeros(seq_len)
        for j in range(seq_len):
            span_idx = j // args.span_size
            s_idx = min(span_idx, len(scs_vector) - 1)
            token_scores[j] = 0.5 * (scs_vector[s_idx] + rss_vector[j])
        
        all_results.append({
            "uid": uid,
            "domain": domain,
            "input_text": sample['input_text'],
            "token_score": token_scores,
            "domain": domain,
        })

        if 'activation' in sample:
            for layer, act in sample['activation'].items():
                if layer < num_layers:
                    layer_activations[layer].append({
                        "uid": uid,
                        "activation": act
                    })

    basename = os.path.basename(args.forward_data_file).split('.pt')[0]

    score_save_path = f"{score_dir}/{basename}_token_scores.pt"
    torch.save(all_results, score_save_path)
    print(f"Saved TokenScores to {score_save_path}")

    for layer in range(1, num_layers):
        if layer_activations[layer]:
            act_path = f"{act_dir}/{basename}_activations_layer{layer}.pt"
            torch.save(layer_activations[layer], act_path)
            print(f"✅ Layer {layer}: Saved {len(layer_activations[layer])} activations.")

if __name__ == "__main__":
    main()