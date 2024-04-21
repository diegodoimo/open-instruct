# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
from pathlib import Path
import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


WORLD_SIZE = int(os.environ["WORLD_SIZE"])


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
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

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
        print("model loading finished. \n\n")
        sys.stdout.flush()
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if use_lora:

        logger.info("Initializing LORA model...")
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
