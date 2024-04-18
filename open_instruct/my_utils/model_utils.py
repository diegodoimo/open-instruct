# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
from pathlib import Path
import os

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from lit_gpt.lora import GPT, Config, mark_only_lora_as_trainable

# from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    load_checkpoint,
    num_parameters,
)

WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def get_model(train_config, fabric):

    if not any(
        (
            train_config.lora_query,
            train_config.lora_key,
            train_config.lora_value,
            train_config.lora_projection,
            train_config.lora_mlp,
            train_config.lora_head,
        )
    ):
        print("Warning: all LoRA layers are disabled!")

    mine_to_lit = {
        "laptop_llama": "laptop_llama",
        "llama-1-7b": "Llama-1-7b-hf",
        "llama-1-13b": "Llama-1-13b-hf",
        "llama-1-30b": "Llama-1-30b-hf",
        "llama-1-65b": "Llama-1-65b-hf",
        "llama-2-7b": "Llama-2-7b-hf",
        "llama-2-13b": "Llama-2-13b-hf",
        "llama-2-70b": "Llama-2-70b-hf",
        "llama-2-7b-chat": "Llama-2-7b-chat-hf",
        "llama-2-13b-chat": "Llama-2-13b-chat-hf",
        "llama-2-70b-chat": "Llama-2-70b-chat-hf",
    }

    if train_config.model_name is not None:
        model_name = train_config.model_name
    else:
        model_name = mine_to_lit[train_config.checkpoint_dir.name]

    print("model_name", model_name)

    config = Config.from_name(
        name=model_name,
        r=train_config.lora_r,
        alpha=train_config.lora_alpha,
        dropout=train_config.lora_dropout,
        to_query=train_config.lora_query,
        to_key=train_config.lora_key,
        to_value=train_config.lora_value,
        to_projection=train_config.lora_projection,
        to_mlp=train_config.lora_mlp,
        to_head=train_config.lora_head,
    )

    with fabric.init_module(empty_init=(WORLD_SIZE > 1)):
        model = GPT(config)

    mark_only_lora_as_trainable(model)
    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )
    fabric.print(
        f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}"
    )

    model = fabric.setup_module(model)

    if train_config.checkpoint_dir is not None and model_name != "laptop_llama":
        checkpoint_path = train_config.checkpoint_dir / "lit_model.pth"
        fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
        # strict=False because missing keys due to LoRA weights not contained in state dict
        load_checkpoint(fabric, model, checkpoint_path, strict=False)

    return model, model_name
