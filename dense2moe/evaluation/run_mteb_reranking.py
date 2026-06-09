import json
import csv
import sys
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional
import torch
from transformers import HfArgumentParser
import mteb
from utils import *
from qwen3_reranker_model import Qwen3RerankerInferenceModel

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

@dataclass
class EvalArguments:
    """
    Arguments.
    """
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    previous_results: Optional[str] = field(default='results', metadata={"help": "output dir of results"})

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

    running_tasks = []
    for t in tasks:
        task_type = t.metadata.type
        if task_type not in ['Retrieval']:
            continue
        running_tasks.append(t)
    return running_tasks


def get_model(model_name: str,  precision: str = 'fp16', **kwargs):
    model = Qwen3RerankerInferenceModel(model_name, **kwargs)
    return model



def run_eval(model, tasks: list, args: EvalArguments, **kwargs):
    Bright_Instructions = {
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
    if not tasks:
        raise RuntimeError("No task selected")
    task_prompts_path = "task_prompts.json"
    with open(task_prompts_path) as f:
        task_prompts = json.load(f)

    encode_kwargs = args.encode_kwargs or dict()

    _num_gpus, _started = torch.cuda.device_count(), False
    if _num_gpus > 1 and not _started and hasattr(model, 'start'):
        model.start() 
        _started = True

    for t in tasks:
        if t.metadata.name == 'BrightRetrieval':
            run_bright(t, model, args, **kwargs)
            continue
        if t.metadata.name in RARB_tasks:
            load_rarb_data(t)
        if t.metadata.name == 'MLQARetrieval':
            load_mlqa_data(t)
        if t.metadata.name == 'HagridRetrieval':
            load_hagrid_data(t)
        if t.metadata.name == 'BelebeleRetrieval':
            load_belebel_data(t)
        evaluation = mteb.MTEB(tasks=[t])
        eval_splits = evaluation.tasks[0].metadata.eval_splits

        task_name = evaluation.tasks[0].metadata.name
        previous_results = args.previous_results
        if task_name in task_prompts:
            model.instruction = task_prompts[task_name]
        subsets = t.hf_subsets
        for split in eval_splits:
            for sub_set in subsets:
                if sub_set in Bright_Instructions:
                    t.metadata.prompt = Bright_Instructions[sub_set]
                if split == 'test':
                    retrieval_save_path = os.path.join(previous_results, f"{task_name}_{sub_set}_predictions.json")
                else:
                    retrieval_save_path = os.path.join(previous_results, f"{task_name}_{sub_set}_{split}_predictions.json")
                try:
                    os.environ['HF_DATASETS_OFFLINE'] = "1"
                    result = evaluation.run(
                        model,
                        eval_splits=[split],
                        eval_subsets=[sub_set],
                        top_k=100,
                        save_predictions=True,
                        output_folder=args.output_dir,
                        previous_results=retrieval_save_path
                    )
                except Exception as e:
                    try:
                        os.environ['HF_DATASETS_OFFLINE'] = "0"
                        result = evaluation.run(
                            model,
                            eval_splits=[split],
                            eval_subsets=[sub_set],
                            top_k=100,
                            save_predictions=True,
                            output_folder=args.output_dir,
                            previous_results=retrieval_save_path
                        )
                    except Exception as e:
                        print(f'failed run {task_name} subset {sub_set}', e)
                        continue
    if model is not None and _started and hasattr(model, 'stop'):
        model.stop()
    return


def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
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
    logger.warning(f"Selected {len(tasks)} tasks:\n" + '\n'.join(str(t) for t in tasks))
    if args.only_load:
        for t in tasks:
            logger.warning(f"Loading {t}")
            t.load_data()
        if not args.load_model:
            return
    model = get_model(args.model, **args.model_kwargs)
    if args.only_load:
        return

    args.encode_kwargs.update(batch_size=args.batch_size)
    run_eval(model, tasks, args, **args.run_kwargs)
    logger.warning(f"Done {len(tasks)} tasks.")
    return


if __name__ == '__main__':
    main()
