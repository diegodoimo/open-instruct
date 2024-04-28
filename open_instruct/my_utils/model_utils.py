# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
from pathlib import Path
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers import LlamaConfig
from peft import LoraConfig, TaskType, get_peft_model
import warnings

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def get_model_hf(
    accelerator,
    model_name_or_path,
    config_name,
    low_cpu_mem_usage,
    torch_dtype=torch.bfloat16,
    use_lora=False,
    lora_rank=64,
    lora_alpha=16,
    lora_dropout=0,
    use_flash_attention=False,
):
    # Load pretrained model and tokenizer
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        warnings.warn("Using a fake llama for debugging\n", stacklevel=2)
        config = LlamaConfig()
        config.intermediate_size = 1000
        config.num_hidden_layers = 3
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.hidden_size = 500
        # raise ValueError(
        #     "You are instantiating a new config instance from scratch. This is not supported by this script."
        # )

    if model_name_or_path:
        # here option for qlora in case
        accelerator.print("model_loading started. \n\n")
        sys.stdout.flush()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            use_flash_attention_2=True if use_flash_attention else False,
        )
        accelerator.print("model loading finished. \n\n")
        sys.stdout.flush()
    else:
        accelerator.print("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if use_lora:

        accelerator.print("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model


def get_model_no_lora(model_name_or_path, precision, low_cpu_mem_usage, accelerator):

    if model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
        accelerator.print("model_loading started. \n\n")
        sys.stdout.flush()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=precision,
            use_flash_attention_2=False,
        )

    else:
        warnings.warn("Using a fake llama for debugging\n", stacklevel=2)
        config = LlamaConfig()
        config.intermediate_size = 1000
        config.num_hidden_layers = 3
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.hidden_size = 500
        model = AutoModelForCausalLM.from_config(config)

    accelerator.print("model loading finished. \n\n")
    sys.stdout.flush()

    return model
