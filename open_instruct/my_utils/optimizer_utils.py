# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
from pathlib import Path
import torch
import math
from transformers import get_scheduler as get_scheduler_hf

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def get_optimizer(model, learning_rate, weight_decay):
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    trainable_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
    #     import bitsandbytes as bnb

    #     optimizer = bnb.optim.PagedAdamW(
    #         trainable_params,
    #         lr=train_config.learning_rate,
    #         weight_decay=train_config.weight_decay,
    #     )
    # else:
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
    )

    return optimizer


def get_scheduler(
    lr_scheduler_type,
    optimizer,
    epochs,
    num_iters,
    warmup_steps=None,
    warmup_ratio=None,
    gradient_accumulation_iters=1,
):
    assert warmup_ratio is None or warmup_steps is None

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(num_iters / gradient_accumulation_iters)
    num_training_steps_for_scheduler = epochs * num_update_steps_per_epoch

    if warmup_steps is None and warmup_ratio is None:
        warmup_steps = 0
    elif warmup_steps is None:
        warmup_steps = int(warmup_ratio * num_training_steps_for_scheduler)

    scheduler = get_scheduler_hf(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=warmup_steps,
    )
    return scheduler
