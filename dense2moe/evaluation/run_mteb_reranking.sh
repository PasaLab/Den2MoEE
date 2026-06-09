
export OPENBLAS_NUM_THREADS='8'
export VLLM_USE_MODELSCOPE=False

model_path=$1
shift
model_name=$1
shift
previous_save_path=$1
shift
benchmark=$1
shift

python run_mteb_reranking.py \
  --model ${model_path} \
  --batch_size 16 --precision fp16 \
  --model_kwargs "{\"batch_size\": 8}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --previous_results ${previous_save_path} \
  --output_dir  results/${model_name}  \
  --benchmark "${benchmark}" $@

  #--tasks "WinoGrande"
