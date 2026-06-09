import argparse
import json
import os
import sys
import uuid
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig
from accelerate import Accelerator

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_hf_tokenizer
from qwen3.modeling_qwen3 import Qwen3ForCausalLM


def detach_to_cpu(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return [detach_to_cpu(v) for v in x]
    return x


@torch.no_grad()
def inference_data(
    model: torch.nn.Module,
    tokenizer,
    samples: List[Dict[str, Any]],
    output_file: str,
    accelerator: Accelerator,
    stream_batch_size: int = 10,
    n_sample_tokens: Optional[int] = None,
):
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    done_tokens_local = 0
    total_samples_local = len(samples)

    output_dir = os.path.dirname(output_file)
    if accelerator.is_main_process and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    temp_dir_rank = os.path.join(output_dir if output_dir else ".", f"temp_stream_batches_rank{rank}")
    if not os.path.exists(temp_dir_rank):
        os.makedirs(temp_dir_rank, exist_ok=True)

    if accelerator.is_main_process:
        accelerator.print(f"temporary batch files will be saved to (each rank independently): {os.path.dirname(temp_dir_rank)}")

    pbar = tqdm(
        samples,
        desc=f"Rank {rank} inferring",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )

    current_batch: List[Dict[str, Any]] = []
    temp_files: List[str] = []

    def save_current_batch():
        nonlocal current_batch, temp_files
        if current_batch:
            batch_idx = len(temp_files)
            temp_file = os.path.join(temp_dir_rank, f"batch_{batch_idx:04d}.pt")
            torch.save(current_batch, temp_file)
            print(f"Rank {rank} save {len(current_batch)} batches to {temp_file}")
            temp_files.append(temp_file)
            current_batch = []

    for idx, sample in enumerate(pbar):
        domain = sample["domain"]
        anchor = " ".join(msg["content"] for msg in sample["messages"])
        pos = " ".join(m["content"] for m in sample['positive_messages'][0])
        input_text = f"{anchor}<|endoftext|>\n{pos}<|endoftext|>"

        batch_dict = tokenizer(
            input_text,
            padding=False,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )

        batch_dict = {k: v.to(accelerator.device) for k, v in batch_dict.items()}

        output = model(**batch_dict)

        done_tokens_local += int(batch_dict["input_ids"].shape[-1])

        last_layer_attention = detach_to_cpu(
            output.attentions[-1][0, :, -1, :].sum(dim=0)
        )

        activation = list(
            # each layer: [1, seq_len, hidden_size]
            detach_to_cpu(output["hidden_states"][layer][1][0, :, :])
            for layer in range(len(output["hidden_states"]))
        )

        current_batch.append(
            {
                "uid": sample.get("uid", uuid.uuid4().hex),
                "input_text": input_text,
                "input_ids": detach_to_cpu(batch_dict["input_ids"]).squeeze(0),
                "attention_mask": detach_to_cpu(batch_dict["attention_mask"]).squeeze(0),
                "last_layer_attention": last_layer_attention,
                "activation": activation,
                "domain": domain,
            }
        )

        if (len(current_batch) >= stream_batch_size) or (idx == total_samples_local - 1):
            save_current_batch()

        # Global token-budget early stop
        if n_sample_tokens is not None:
            global_done = accelerator.reduce(
                torch.tensor(done_tokens_local, device=accelerator.device, dtype=torch.long), reduction="sum"
            ).item()
            if accelerator.is_main_process:
                tqdm.write(f"[Global tokens used: {global_done}/{n_sample_tokens}]")
            if global_done >= n_sample_tokens:
                save_current_batch()
                break

    accelerator.wait_for_everyone()

    # Rank 0 collects all rank temp files and merges
    if accelerator.is_main_process:
        all_outputs = []
        rank_dirs = [os.path.join(output_dir if output_dir else ".", f"temp_stream_batches_rank{r}") for r in range(world_size)]

        for r, rd in enumerate(rank_dirs):
            if not os.path.isdir(rd):
                continue
            batch_files = sorted([os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".pt")])
            for bf in tqdm(batch_files, desc=f"Merging rank {r}"):
                batch_data = torch.load(bf, map_location="cpu")
                all_outputs.extend(batch_data)
            for bf in batch_files:
                try:
                    os.remove(bf)
                except FileNotFoundError:
                    pass
            try:
                os.rmdir(rd)
            except OSError:
                pass

        torch.save(all_outputs, output_file)
        accelerator.print(f"merge completed! final file saved to: {output_file}")
        accelerator.print(f"total samples processed: {len(all_outputs)}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use slow tokenizer")
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="Input .jsonl files")
    parser.add_argument("--output_file", type=str, default="tmp_output/model_outputs.pt", help="Final .pt output")
    parser.add_argument("--n_sample_tokens", type=int, default=None, help="Global max tokens across all ranks")
    parser.add_argument("--stream_batch_size", type=int, default=500, help="Samples per temporary batch file")
    parser.add_argument(
        "--shard_with_device_map",
        action="store_true",
        help=(
            "Use Transformers device_map sharding (single-process model parallel). "
            "If enabled, you must run with a single process (world_size=1)."
        ),
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="balanced_low_0",
        help="device_map passed to from_pretrained when --shard_with_device_map is set",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )

    args = parser.parse_args()

    # Accelerator: manages processes, devices, logging
    accelerator = Accelerator(device_placement=True, log_with=None)

    # Load and pre-shard input samples *before* splitting
    samples: List[Dict[str, Any]] = []
    for input_file in args.input_files:
        with open(input_file, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

    # If using device_map sharding, force single-process
    if args.shard_with_device_map and accelerator.num_processes > 1:
        if accelerator.is_main_process:
            raise RuntimeError(
                "When --shard_with_device_map is set, please run with a single process (world_size=1). "
                "Use `accelerate launch --num_processes 1 ...`"
            )
        else:
            return

    # Split samples across processes (DP mode). In device_map mode we still split but world_size==1
    with accelerator.split_between_processes(samples) as split_samples:
        samples = split_samples

    # If this rank has no samples, early exit after a barrier
    if len(samples) == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("No samples assigned to rank 0; nothing to do.")
        return

    if accelerator.is_main_process:
        accelerator.print(
            f"successfully loaded and sharded: total samples approximately {len(samples) * accelerator.num_processes} (each card processes {len(samples)} samples)"
        )

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        output_router_logits=True,
        output_hidden_states=True,
        output_attentions=True,
    )

    tokenizer = load_hf_tokenizer(
        model_name_or_path=args.model_name_or_path,
        use_fast_tokenizer=not args.use_slow_tokenizer,
    )

    # model loading strategy
    if args.shard_with_device_map:
        model = Qwen3ForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            revision=None,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model.eval()
        # Do NOT call accelerator.prepare(model) here to avoid conflicts with device_map
        if accelerator.is_main_process:
            accelerator.print("model loaded (Transformers device_map sharding / single-process model parallel)")
    else:
        # Data-parallel multi-process; each rank has a full copy on its own device
        model = Qwen3ForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            revision=None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model.eval()
        model = accelerator.prepare(model)  # place on the right device per rank
        if accelerator.is_main_process:
            accelerator.print("model loaded (Accelerate data parallel / multi-process multi-card)")
            accelerator.print(f"number of data parallel processes used: {accelerator.num_processes}")

    # Inference
    inference_data(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        output_file=args.output_file,
        accelerator=accelerator,
        stream_batch_size=args.stream_batch_size,
        n_sample_tokens=args.n_sample_tokens,
    )

    if accelerator.is_main_process:
        accelerator.print("\nall processes completed!")


if __name__ == "__main__":
    main()
