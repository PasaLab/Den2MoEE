import json
import sys
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Any

import torch
from transformers import HfArgumentParser
import mteb
from pathlib import Path
from mteb import AbsTaskRetrieval,RetrievalEvaluator
from time import time
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
import csv
from collections import defaultdict
import statistics
from mteb.benchmarks.benchmarks import Benchmark


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen3_embedding_model import Qwen3Embedding
from den2moee_embedding_model import Den2MoEEEmbedding
from dense2moe.evaluation.utils import load_hagrid_data, load_belebel_data, load_rarb_data, load_mlqa_data


logging.basicConfig(
    format="%(levelname)s|%(asctime)s|%(name)s#%(lineno)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger('run_mteb.py')


RARB_tasks = [ "ARCChallenge",
            "AlphaNLI",
            "HellaSwag",
            "WinoGrande",
            "PIQA",
            "SIQA",
            "Quail",
            "SpartQA",
            "TempReasonL1",
            "TempReasonL2Pure",
            "TempReasonL2Fact",
            "TempReasonL2Context",
            "TempReasonL3Pure",
            "TempReasonL3Fact",
            "TempReasonL3Context",
            "RARbCode",
            "RARbMath",
        ]

def evaluate(
    self,
    model,
    split: str = "test",
    subsets_to_run: list | None = None,
    *,
    encode_kwargs: dict[str, Any] = {},
    **kwargs,
):
    retriever = RetrievalEvaluator(
        retriever=model,
        task_name=self.metadata.name,
        encode_kwargs=encode_kwargs,
        **kwargs,
    )
    scores = {}
    hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
    if subsets_to_run is not None:
        hf_subsets = [s for s in hf_subsets if s in subsets_to_run]
    for hf_subset in hf_subsets:
        logger.info(f"Subset: {hf_subset}")
        if hf_subset == "default":
            corpus, queries, relevant_docs = (
                self.corpus[split],
                self.queries[split],
                self.relevant_docs[split],
            )
        else:
            corpus, queries, relevant_docs = (
                self.corpus[hf_subset][split],
                self.queries[hf_subset][split],
                self.relevant_docs[hf_subset][split],
            )
        scores[hf_subset] = self._evaluate_subset(
            retriever, corpus, queries, relevant_docs, hf_subset, split=split, **kwargs
        )
    return scores

def _evaluate_subset(
    self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
):
    start_time = time()
    results = retriever(corpus, queries)
    end_time = time()

    save_predictions = kwargs.get("save_predictions", False)
    export_errors = kwargs.get("export_errors", False)
    if save_predictions or export_errors:
        output_folder = Path(kwargs.get("output_folder", "results"))
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    if save_predictions:
        top_k = kwargs.get("top_k", None)
        if top_k is not None:
            for qid in list(results.keys()):
                doc_ids = set(
                    sorted(
                        results[qid], key=lambda x: results[qid][x], reverse=True
                    )[:top_k]
                )
                results[qid] = {
                    k: v for k, v in results[qid].items() if k in doc_ids
                }
        split = kwargs.get('split', 'test')
        if split != 'test':
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_{split}_predictions.json"
            )
        else:
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )

        with open(qrels_save_path, "w") as f:
            json.dump(results, f)

    ndcg, _map, recall, precision, naucs = retriever.evaluate(
        relevant_docs,
        results,
        retriever.k_values,
        ignore_identical_ids=self.ignore_identical_ids,
    )
    mrr, naucs_mrr = retriever.evaluate_custom(
        relevant_docs, results, retriever.k_values, "mrr"
    )
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        **{
            k.replace("@", "_at_").replace("_P", "_precision").lower(): v
            for k, v in naucs.items()
        },
        **{
            k.replace("@", "_at_").replace("_P", "_precision").lower(): v
            for k, v in naucs_mrr.items()
        },
    }
    self._add_main_score(scores)

    if export_errors:
        errors = {}

        top_k = kwargs.get("top_k", 1)
        if not save_predictions and top_k == 1:
            for qid in results.keys():
                doc_scores = results[qid]
                sorted_docs = sorted(
                    doc_scores.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
                results[qid] = dict(sorted_docs)
        for qid, retrieved_docs in results.items():
            expected_docs = relevant_docs[qid]
            false_positives = [
                doc for doc in retrieved_docs if doc not in expected_docs
            ]
            false_negatives = [
                doc for doc in expected_docs if doc not in retrieved_docs
            ]
            if false_positives or false_negatives:
                errors[qid] = {
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                }
        split = kwargs.get('split', 'test')
        if split != 'test':
            errors_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_{split}_errors.json"
            )
        else:
            errors_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_errors.json"
            )
        with open(errors_save_path, "w") as f:
            json.dump(errors, f)

    return scores

AbsTaskRetrieval.evaluate = evaluate
AbsTaskRetrieval._evaluate_subset = _evaluate_subset

@dataclass
class EvalArguments:
    """
    Arguments.
    """
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model name for the save path"}
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model kwargs, json string."},
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific encode kwargs, json string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific kwargs for `MTEB.run()`, json string."},
    )

    output_dir: Optional[str] = field(default='results', metadata={"help": "output dir of results"})
    benchmark: Optional[str] = field(default=None, metadata={"help": "Benchmark name"})
    tasks: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    langs: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    only_load: bool = field(default=False, metadata={"help": ""})
    load_model: bool = field(default=False, metadata={"help": "when only_load"})

    batch_size: int = field(default=128, metadata={"help": "Will be set to `encode_kwargs`"})
    precision: str = field(default='fp16', metadata={"help": "amp_fp16,amp_bf16,fp16,bf16,fp32"})

    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(',')
        if isinstance(self.langs, str):
            self.langs = self.langs.split(',')
        for name in ('model', 'encode', 'run'):
            name = name + '_kwargs'
            attr = getattr(self, name)
            if attr is None:
                setattr(self, name, dict())
            elif isinstance(attr, str):
                setattr(self, name, json.loads(attr))


def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks


def get_model(model_path: str, model_name: str, precision: str = 'fp16', **kwargs):
    if "Den2MoEE-Embedding" in model_name:
        if "pooler_type" not in kwargs:
            kwargs["pooler_type"] = "den2moee"
        model = Den2MoEEEmbedding(model_path, model_name=model_name, precision=precision, **kwargs)
    elif "Qwen3-Embedding" in model_name:
        model = Qwen3Embedding(model_path, model_name=model_name, precision=precision, **kwargs)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model

def run_bright(t, model, args, **kwargs):
    # task_instructions = {}
    Instructions = {
        "aops" : "Given a Math problem, retrieve relevant examples that help answer the problem.",
        "biology": "Given a post, retrieve relevant passages that help answer the post.",
        "earth_science": "Given a post, retrieve relevant passages that help answer the post.",
        "economics": "Given a economics post, retrieve relevant passages that help answer the post.",
        "leetcode": "Given a coding problem, retrieve relevant examples that help answer the problem.",
        "pony": "Given a question about pony program language, retrieve relevant passages that help answer the question.",
        "psychology": "Given a psychology post, retrieve relevant passages that help answer the post.",
        "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
        "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
        "robotics": "Given a robotics post, retrieve relevant passages that help answer the post.",
        "stackoverflow": "Given a stackoverflow post, retrieve relevant passages that help answer the post.",
        "sustainable_living": "Given a sustainable_living post, retrieve relevant passages that help answer the post."
    }
    encode_kwargs = args.encode_kwargs or dict()

    for task in Instructions.keys():
        instruct = Instructions[task]
        t.metadata.prompt = {'query': instruct}
        evaluation = mteb.MTEB(tasks=[t], task_cache_dir="./mteb_cache")
        eval_splits = evaluation.tasks[0].metadata.eval_splits
        results = evaluation.run(
            model,
            output_folder=args.output_dir,
            encode_kwargs=encode_kwargs,
            eval_splits=eval_splits,
            eval_subsets=[task],
            **kwargs
        )
        break


def run_eval(model, tasks: list, args: EvalArguments, **kwargs):
    if not tasks:
        raise RuntimeError("No task selected")

    encode_kwargs = args.encode_kwargs or dict()

    _num_gpus, _started = torch.cuda.device_count(), False
    if _num_gpus > 1 and not _started and hasattr(model, 'start'):
        model.start()
        _started = True

    tasks_results = {}
    for t in tasks:
        if t.metadata.name == 'BrightRetrieval':
            run_bright(t, model, args, **kwargs)
            continue
        if t.metadata.name == 'MLQARetrieval':
            load_mlqa_data(t)
        if t.metadata.name == 'HagridRetrieval':
            load_hagrid_data(t)

        if t.metadata.name == 'BelebeleRetrieval':
            load_belebel_data(t)
        if t.metadata.name in RARB_tasks:
            load_rarb_data(t)
        evaluation = mteb.MTEB(tasks=[t])
        
        try:
            os.environ['HF_DATASETS_OFFLINE'] = "1"
            results = evaluation.run(
                model,
                output_folder=args.output_dir,
                encode_kwargs=encode_kwargs,
                **kwargs
            )
        except Exception as e:
            try:
                os.environ['HF_DATASETS_OFFLINE'] = "0"
                results = evaluation.run(
                    model,
                    output_folder=args.output_dir,
                    encode_kwargs=encode_kwargs,
                    **kwargs
                )
            except Exception as e:
                print(f'meet error when running task: {t.metadata.name}. {str(e)}')
                continue
        tasks_results[t] = results   
            

    if model is not None and _started and hasattr(model, 'stop'):
        model.stop()
    return tasks_results



def analyze_results(tasks_results: dict):
    task_scores = defaultdict(list)
    all_scores = []

    try:
        for task, task_result in tasks_results.items():
            scores_dict = task_result[0].scores
            score_key = next(iter(scores_dict))
            subset_scores = scores_dict[score_key]
            
            if isinstance(subset_scores, list):
                score = sum([ele['main_score'] for ele in subset_scores]) / len(subset_scores)
            elif isinstance(subset_scores, dict):
                score = subset_scores.get('main_score', 0)
            else:
                score = subset_scores[0]['main_score'] if len(subset_scores) > 0 else 0
            
            task_scores[task.metadata.type].append(score)
            all_scores.append(score)
    except Exception as e:
        print(f'meet error when analyzing results: {str(e)}')
        import pdb; pdb.set_trace()

    # 1) 每个任务(type)的平均分
    task_means = {task_type: statistics.mean(scores) 
                  for task_type, scores in task_scores.items()}

    # 2) Type-Mean：先按 task type 平均，再对类型平均（任务类型平权）
    type_mean = statistics.mean(task_means.values())

    # 3) Task-Mean：所有数据集直接平均（数据集平权）
    task_mean = statistics.mean(all_scores)

    # 4) 格式化输出
    print("=== Task Means ===")
    for task_type, mean_score in task_means.items():
        print(f"{task_type:20s} {mean_score:.4f}")
    print("==================")
    print(f"Task-Mean : {task_mean:.4f}")
    print(f"Type-Mean : {type_mean:.4f}")
    print(f"all-scores len: {len(all_scores)}")

    return task_means, task_mean, type_mean



def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1])) as f:
            config = json.load(f)
        logger.warning(f"Json config {f.name} : \n{json.dumps(config, indent=2)}")
        args, *_ = parser.parse_dict(config)
        del config, f
    else:
        args, *_ = parser.parse_args_into_dataclasses()
        logger.warning(f"Args {args}")
    del parser

    tasks = get_tasks(args.tasks, args.langs, args.benchmark)
    print('args.model_kwargs', args.model_kwargs)
    logger.warning(f"Selected {len(tasks)} tasks:\n" + '\n'.join(str(t) for t in tasks))
    if args.only_load:
        for t in tasks:
            logger.warning(f"Loading {t}")
            try:
                t.load_data()
            except Exception as e:
                t.load_data(force_download=True)
            else:
                continue
            
        if not args.load_model:
            return
    model = get_model(args.model, args.model_name, precision=args.precision, **args.model_kwargs)
    if args.only_load:
        return

    start_time = time()
    args.encode_kwargs.update(batch_size=args.batch_size)
    tasks_results = run_eval(model, tasks, args, **args.run_kwargs)
    analyze_results(tasks_results)
    end_time = time()

    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    logger.warning(f"Done {len(tasks)} tasks. Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")
    return


if __name__ == '__main__':
    main()
