# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.llm.template import TemplateType
from ..constant import LLMModelType
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model)


register_model(
    ModelMeta(
        LLMModelType.den2moee_emb, [
            ModelGroup([
                Model('Den2moee/Den2moee-Embedding-0.6B', 'Den2moee/Den2moee-Embedding-0.6B'),
            ]),
        ],
        TemplateType.qwen3_emb,
        get_model_tokenizer_with_flash_attn,
        additional_saved_files=['config_sentence_transformers.json', '1_Pooling', 'modules.json'],
        architectures=['Den2moeeForCausalLM']))

