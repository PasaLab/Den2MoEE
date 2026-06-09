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
MODEL_PATH=${GY2_PATH}/models/Den2Moee-Embedding-0.6B-svd-init-0.0-std-1e-2
# MODEL_PATH=${GY2_PATH}/models/Den2Moee-Embedding-4B-svd-init-0.0-std-1e-2

MODEL_TYPE=den2moee_emb
TRAIN_TYPE=den2moee_dense_router


normalize_model_path() {
    local path="$1"
    if [[ "$path" =~ opensource/Moe2-Embedding ]]; then
        path=$(basename "$path")"-dense"
    elif [[ "$path" =~ /v[^/]+/checkpoint-[0-9]+$ ]]; then
        path=$(echo "$path" | sed -E 's|(.*)/v([^/]+)/checkpoint-([0-9]+)$|\1/Moe2-v\2-ckpt\3|')
    elif [[ "$path" =~ /checkpoint-([0-9]+)$ ]]; then
        path=$(echo "$path" | sed -E 's|/checkpoint-([0-9]+)$|-ckpt\1-sparse|')
    else
        [[ "$path" =~ (-dense|-sparse)$ ]] || path="${path}-sparse"
    fi

    echo "$path"
}

OUTPUT_DIR=$(normalize_model_path "$MODEL_PATH")
echo "✅ OUTPUT_DIR = $OUTPUT_DIR"


DATASET_PATHS=(
    "${CQ2_PATH}/data/den2moee/cosine_sampling_code.jsonl"
    "${CQ2_PATH}/data/den2moee/cosine_sampling_multi.jsonl"
)
DATASET_PATH="${DATASET_PATHS[*]}"
SAVE_STEPS=50
EVAL_STEPS=20
PER_DEVICE_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=1e-5
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

swift sft \
    --model $MODEL_PATH \
    --task_type embedding \
    --model_type $MODEL_TYPE \
    --train_type $TRAIN_TYPE \
    --dataset $DATASET_PATH \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --eval_strategy steps \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.10 \
    --output_dir $OUTPUT_DIR \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --loss_type cosine_similarity \
    --dataloader_drop_last true \
    --deepspeed zero2

