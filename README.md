# Den2MoEE

Open-source anonymous code for paper Den2MoEE: Reconstructing Dense LLMs into Expert-Specialized
Mixture-of-Experts for Efficient Embedding Models

## Training
Training code is available in `ms-swift/`, and the scripts are located in `ms-swift/scripts/den2moee`. 

## Den-to-Moe Conversion
Coversion code is availabel in `dense2moe/`, and see `dense2moe/README.md` for details.

## Evaluation
Evaluation code is available in `dense2moe/evaluation/`, and the script is `dense2moe/evaluation/run_mteb.sh`. 
```bash
bash run_mteb.sh --benchmark=[XXX] --model-path=[XXX] --batch-size=[XX]
```

