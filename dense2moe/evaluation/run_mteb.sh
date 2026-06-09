export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'

GY2_PATH=""
MODEL_PATH=${GY2_PATH}/models/Den2MoEE-Embedding-4B/v9-20260113-213824/checkpoint-2500

# BENCHMARK="MTEB(Multilingual, v2)"
BENCHMARK="MTEB(Code, v1)"
BENCHMARK_CLEAN=$(echo "$BENCHMARK" | sed 's/[ ,]/_/g')
LOG_PATH=./logs
BATCH_SIZE=32
POOLER_TYPE="den2moee"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path=*)
      MODEL_PATH="${1#*=}"
      echo "New MODEL_PATH: $MODEL_PATH"
      shift
      ;;
    --benchmark=*)
      BENCHMARK="${1#*=}"
      echo "New BENCHMARK: $BENCHMARK"
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      echo "New BATCH_SIZE: $BATCH_SIZE"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

BASENAME=$(basename "$MODEL_PATH")

if [[ "$BASENAME" == checkpoint-* ]]; then
    PARENT_DIR=$(basename "$(dirname "$MODEL_PATH")")
    CKPT_NUM=${BASENAME#checkpoint-}
    MODEL_NAME="Den2MoEE/Den2MoEE-Embedding-${PARENT_DIR}-ckpt${CKPT_NUM}"
else
    MODEL_NAME="Den2MoEE/${BASENAME}"
fi

echo "$MODEL_NAME"
mkdir -p $LOG_PATH/${MODEL_NAME}

start_time=$(date +%s)


python run_mteb.py \
  --model ${MODEL_PATH} \
  --model_name ${MODEL_NAME} \
  --precision fp16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"${POOLER_TYPE}\", \"do_norm\": true, \"use_instruction\": true, \"instruction_template\": \"Instruct: {}\nQuery:\", \"instruction_dict_path\": \"task_prompts.json\", \"attn_implementation\":\"flash_attention_2\"}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir results/${MODEL_NAME} \
  --batch_size ${BATCH_SIZE} \
  --benchmark "${BENCHMARK}" $@ 2>&1 | tee -a ${LOG_PATH}/${MODEL_NAME}/${BENCHMARK_CLEAN}_$(date +%Y%m%d_%H%M%S).log


end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"