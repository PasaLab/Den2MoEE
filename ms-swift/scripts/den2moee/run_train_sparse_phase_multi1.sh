# losses: plugin/loss.py
# data format: docs/source_en/Customization/Custom-dataset.md
# --use_chat_template must be false to use generation template
# --dataloader_drop_last must be true or eval gather will throw error
# --model iic/gte-modernbert-base iic/gte_Qwen2-7B-instruct also supported
# INFONCE_TEMPERATURE default value is 0.01, here we use 0.1 because it makes
# the `sentence-transformers/stsb:positive` dataset result to a zero loss

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE


GY2_PATH=""
CQ2_PATH=""
MASTER_ADDR=29.127.50.145
MASTER_PORT=29500

MODEL_PATH=${GY2_PATH}/models/den2moee-dense/v5-20251231-173016/checkpoint-250


MODEL_TYPE=moe2_emb
TRAIN_TYPE=moe2_sparse

normalize_model_path() {
    local path="$1"
    path="${path%/}"

    local dir base new_path
    dir="$(dirname "$path")"
    base="$(basename "$path")"

    if [[ "$base" =~ ^checkpoint-([0-9]+)$ ]]; then
        local ckpt_id="${BASH_REMATCH[1]}"
        local parent="$(basename "$dir")"
        local parent_parent="$(dirname "$dir")"
        new_path="${parent_parent}-${parent}-ckpt${ckpt_id}"
    else
        new_path="$path"
        if [[ "$new_path" == *opensource* ]]; then
            new_path="${new_path//opensource/ms-swift}"
        fi
    fi

    echo "$new_path"
}

OUTPUT_DIR=$(normalize_model_path "$MODEL_PATH")

echo "✅ OUTPUT_DIR = $OUTPUT_DIR"

DATASET_PATHS=(
    # "${CQ2_PATH}/data/den2moee/finetune_sampling_multi_neg2.jsonl"
    # "${CQ2_PATH}/data/den2moee/finetune_sampling_code_neg2.jsonl"
    # "${CQ2_PATH}/data/den2moee/finetune_sampling_all_neg2.jsonl"
    # "${CQ2_PATH}/data/den2moee/finetune_sampling_med_pos_only.jsonl"
)
DATASET_PATH="${DATASET_PATHS[*]}"
SAVE_STEPS=20
EVAL_STEPS=20
PER_DEVICE_EVAL_BATCH_SIZE=32

GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=5e-6
PER_DEVICE_TRAIN_BATCH_SIZE=32
export INFONCE_TEMPERATURE=0.1

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path=*)
      MODEL_PATH="${1#*=}"
      echo "New MODEL_PATH: $MODEL_PATH"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p $OUTPUT_DIR


NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=8 \
swift sft \
    --model $MODEL_PATH \
    --task_type embedding \
    --model_type $MODEL_TYPE \
    --train_type $TRAIN_TYPE \
    --dataset $DATASET_PATH \
    --load_from_cache_file true \
    --split_dataset_ratio 0.00 \
    --eval_strategy steps \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --output_dir $OUTPUT_DIR \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --loss_type infonce \
    --dataloader_drop_last true \
    --deepspeed zero2


    # --lr_scheduler_type cosine_with_restarts \
    # --lr_scheduler_kwargs '{"num_cycles": 2}' \

