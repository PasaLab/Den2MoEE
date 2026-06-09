# Qwen3 Embedding / Reranking model MTEB Evaluation Scripts

## Evaluate Embedding Models

```bash
bash run_mteb.sh ${model_path} ${model_name} ${benchmark_name}
```

- **model_path**: Path or name of the model weights file (e.g., "Qwen/Qwen3-Embedding-0.6B").
- **model_name**: Name of the model, used for naming the result directory.
- **benchmark_name**: Name of the benchmark. Supported values: "MTEB(eng, v2)", "MTEB(cmn, v1)", "MTEB(Code, v1)", "MTEB(Multilingual, v2)".
- Evaluation results will be saved in the directory: `results/${model_name}/${model_name}/no_revision_available`. Each task's results will be stored in a separate JSON file.

### Evaluate Reranking Models
```bash
bash run_mteb_reranking.sh ${model_path} ${model_name} ${retrieval_path} ${benchmark}
```
- **model_path**: Path to the reranking model weights file (e.g., "Qwen/Qwen3-Reranker-0.6B").
- **model_name**: Name of the model, used for naming the result directory.
- **retrieval_path**: Path to the retrieval results generated during the embedding evaluation phase. (Reranking evaluation depends on the recall results produced by the embedding model). This should generally be:
```
results/${embedding_model_name}
```
- Results will be saved in: `results/${model_name}/no_model_name_available/no_revision_available/`.

### Summarizing Experimental Results
```bash
python3 summary.py results/${embedding_model_name}/${embedding_model_name}/no_version_available benchmark_name
python3 summary.py results/${reranking_model_name}/no_model_name/no_version_available benchmark_name
```