#!/bin/bash

# Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
NUM_GPUS=8
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

GY2_PATH=""
CQ2_PATH=""
WORK_DIR=${CQ2_PATH}/dense2moe

MODEL_PATH=${GY2_PATH}/models/opensource/Qwen3-Embedding-0.6B
OUTPUT_DIR=${WORK_DIR}/tmp_out
INPUT_FILE=${WORK_DIR}/../data/calibration.jsonl

cd $WORK_DIR
mkdir -p ${OUTPUT_DIR}



while [[ $# -gt 0 ]]; do
  case $1 in
    --input-file=*)
      INPUT_FILE="${1#*=}"
      echo "New INPUT_FILE: $INPUT_FILE"
      shift
      ;;
    --output-file=*)
      OUTPUT_FILE="${1#*=}"
      echo "New OUTPUT_FILE: $OUTPUT_FILE"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

MODEL_NAME=$(basename $MODEL_PATH)

if [ -z "$OUTPUT_FILE" ]; then
  DATE_TIME=$(date +"%Y%m%d_%H%M%S")
  CALIBRATE_DATASET=$(basename $INPUT_FILE .jsonl)
  CALIBRATE_DATASET=domain
  OUTPUT_FILE=${OUTPUT_DIR}/${MODEL_NAME}_${CALIBRATE_DATASET}_${DATE_TIME}_forward.pt
  echo "OUTPUT_FILE was not set. Current OUTPUT_FILE: $OUTPUT_FILE"
fi


accelerate launch --num_processes $NUM_GPUS --multi_gpu forward_calibration.py \
  --model_name_or_path $MODEL_PATH \
  --input_files $INPUT_FILE \
  --output_file $OUTPUT_FILE

chmod -R 777 $OUTPUT_DIR
