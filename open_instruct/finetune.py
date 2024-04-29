#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import (
    SchedulerType,
    get_scheduler,
)


import numpy as np
import sys
import time


from collections import defaultdict


# *******************************************************************************

from my_utils.dataloader_utils import get_dataloader
from my_utils.helpers import print_memory_consumed, save_with_accelerate
from my_utils.dataset_utils import (
    get_dataset_open_instruct_new,
    get_mmlu_open_instruct,  # old version
    MMLU_Dataset,
)
from my_utils.dataloader_utils import get_dataloader
from my_utils.optimizer_utils import get_optimizer, get_scheduler
from my_utils.tokenizer_utils import get_tokenizer
from my_utils.model_utils import get_model_hf, get_model_no_lora

from overlap_utils.overlap_functions import compute_overlap
from overlap_utils.helpers import get_embdims, get_target_layers_llama
from overlap_utils.extract_repr import extract_activations

# with fully sharded daat parallel if we can make this working
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
)
from functools import partial
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from collections import defaultdict
from dadapy.data import Data
import pickle
from overlap_utils.pairwise_distances import compute_distances

# *******************************************************************


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )

    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Where to store the final model."
    )
    parser.add_argument(
        "--out_filename", type=str, default="", help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=("Turn on gradient checkpointing. Saves memory but slows training."),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=-1,
        help="Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).",
    )
    parser.add_argument(
        "--use_8bit_optimizer",
        action="store_true",
        help="Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).",
    )
    parser.add_argument(
        "--measure_baselines",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--measure_overlap",
        action="store_true",
        help="",
    )
    parser.add_argument("--overlap_base_dir", type=str, default=None, help="")

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json",
                "jsonl",
            ], "`train_file` should be a json/jsonl file."
    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # os.environ["ACCELERATE_MIXED_PRECISION"] = args.precision

    # # # we use fsdp also when world size ==1. accelerate issue in casting
    # if int(os.environ["WORLD_SIZE"]) > 1:
    #     os.environ["ACCELERATE_USE_FSDP"] = "true"

    # #     os.environ["FSDP_SHRDING_STRATEGY"] = "FULL_SHARD"
    # #     os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    # #     os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "LlamaDecoderLayer"

    # #     os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    # #     os.environ["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"
    # #     os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    # def lambda_fn(module: torch.nn.Module):
    #     if isinstance(module, LlamaDecoderLayer):
    #         return True  # like transformer_auto_wrap_policy
    #     if isinstance(module, torch.nn.Linear) and all(
    #         p.requires_grad for p in module.parameters()
    #     ):
    #         return True  # wrap each trainable linear separately
    #     return False

    # auto_wrap_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

    # fsdp_plugin = FullyShardedDataParallelPlugin(
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    #     mixed_precision_policy=MixedPrecision(
    #         param_dtype=torch.bfloat16,
    #         reduce_dtype=torch.bfloat16,
    #         buffer_dtype=torch.bfloat16,
    #     ),
    #     auto_wrap_policy=auto_wrap_policy,
    #     cpu_offload=False,
    #     ignored_modules=None,
    #     limit_all_gathers=True,
    #     use_orig_params=False,
    #     param_init_fn=None,
    #     sync_module_states=True,
    #     forward_prefetch=False,
    #     activation_checkpointing=False,
    # )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        # fsdp_plugin=fsdp_plugin,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    world_size = accelerator.num_processes

    # *******************************************************
    # # Load pretrained model and tokenizer

    # model = get_model_no_lora(
    #     model_name_or_path=args.model_name_or_path,
    #     precision=torch.bfloat16,
    #     low_cpu_mem_usage=args.low_cpu_mem_usage,
    #     accelerator=accelerator,
    # )

    model = get_model_hf(
        accelerator=accelerator,
        model_name_or_path=args.model_name_or_path,
        config_name=args.config_name,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_flash_attention=False,
    )

    tokenizer = get_tokenizer(
        tokenizer_path=args.tokenizer_name, model_path=args.model_name_or_path
    )

    max_seq_len = model.config.max_position_embeddings
    if args.max_seq_length is not None:
        max_seq_len = args.max_seq_length
    accelerator.print(max_seq_len)

    # ****************************************************************************
    # # Preprocessing the datasets.

    # train_dataset = get_dataset_open_instruct_new(
    #     accelerator=accelerator,
    #     filepath=args.train_file,
    #     tokenizer=tokenizer,
    #     max_seq_length=2048,
    #     num_processes=1,
    # )

    # val_dataset = get_mmlu_open_instruct(
    #     filepath=args.test_file,
    #     tokenizer=tokenizer,
    #     data_fold="val",
    #     max_seq_length=args.max_seq_length,
    #     num_processes=6,
    #     num_samples=None,
    #     subjects=None,
    # )

    # test_dataset = get_mmlu_open_instruct(
    #     filepath=args.test_file,
    #     tokenizer=tokenizer,
    #     data_fold="test",
    #     max_seq_length=args.max_seq_length,
    #     num_processes=6,
    #     num_samples=None,
    #     subjects=None,
    # )

    train_dataset, longest_seq = MMLU_Dataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        num_few_shots=0,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        split="train",
    ).construct_dataset()

    val_dataset, longest_seq = MMLU_Dataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        num_few_shots=0,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        split="validation",
    ).construct_dataset()

    test_dataset, longest_seq = MMLU_Dataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        num_few_shots=0,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        split="test",
    ).construct_dataset()

    # ******************************************************************************************
    # ignore the issue of pad token in Llamas
    assert args.per_device_train_batch_size == 1
    assert args.per_device_eval_batch_size == 1

    # # DataLoaders creation:
    train_loader, train_sampler = get_dataloader(
        train_dataset,
        args.per_device_train_batch_size,
        tokenizer.pad_token_id,
        world_size=world_size,
        shuffle=True,
        num_processes=6,
        return_sampler=True,
    )

    val_loader = get_dataloader(
        val_dataset,
        args.per_device_eval_batch_size,
        tokenizer.pad_token_id,
        world_size=world_size,
        shuffle=False,
        num_processes=6,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = get_dataloader(
            test_dataset,
            args.per_device_eval_batch_size,
            tokenizer.pad_token_id,
            world_size=world_size,
            shuffle=False,
            num_processes=6,
        )

    # *******************************************************************************

    gradient_accumulation_iters = max(
        1, int(args.batch_size / args.per_device_train_batch_size / world_size)
    )
    optimizer = get_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        epochs=args.num_train_epochs,
        num_iters=len(train_loader),
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_iters=gradient_accumulation_iters,
    )

    # ************************************************************************

    # Prepare everything with `accelerator`.
    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_loader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(
    #    range(args.max_train_steps), disable=not accelerator.is_local_main_process
    # )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    # progress_bar.update(completed_steps)

    filename = ""
    if args.out_filename != "":
        filename = "_" + args.out_filename

    stats = defaultdict()
    stats["num_epochs"] = args.num_train_epochs
    stats["lr"] = args.learning_rate
    stats["scheduler"] = args.lr_scheduler_type
    stats["batch_size"] = args.batch_size

    meter = measure_statistics(
        stats,
        model,
        val_loader,
        test_loader,
        tokenizer,
        accelerator,
        ckpt_dir=args.overlap_base_dir,
        results_dir=args.output_dir,
        prepare_for_overlap=args.measure_overlap,
        filename=f"{filename}epoch{args.num_train_epochs}",
    )

    if args.measure_baselines:
        meter.update(
            accelerator=accelerator,
            model=model,
            completed_steps=0,
            epoch=0,
            do_overlap=args.measure_overlap,
        )

    accelerator.print("start training")
    print_memory_consumed()
    accelerator.print("before train run")
    sys.stdout.flush()

    for epoch in range(starting_epoch, args.num_train_epochs):
        meter.update(
            accelerator=accelerator,
            model=model,
            completed_steps=0,
            epoch=0,
            do_overlap=args.measure_overlap,
            do_val=True,
        )

        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        start = time.time()
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):

                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    t_tot = time.time() - start

                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    logger.info(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Time: {t_tot/3600: .2f} hours"
                    )
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if completed_steps % args.eval_steps == 0:

                    meter.update(
                        accelerator=accelerator,
                        model=model,
                        completed_steps=completed_steps,
                        epoch=epoch,
                        do_val=True,
                        do_overlap=args.measure_overlap,
                    )

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        meter.update(
            accelerator=accelerator,
            model=model,
            completed_steps=completed_steps,
            epoch=epoch,
            do_test=True,
            do_overlap=args.measure_overlap,
        )
        print_memory_consumed()

        # save model
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        save_with_accelerate(accelerator, model, output_dir, args)

    if args.with_tracking:
        accelerator.end_training()

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #     save_with_accelerate(accelerator, model, args.output_dir, args)


# FSDP has issues with `inference_mode`
# @torch.inference_mode()
@torch.no_grad()
def evaluate(model, dataloader, tokenizer, restrict_targets):
    model.eval()

    predictions, ground_truths = [], []
    choices = ["A", "B", "C", "D"]

    candidate_token_ids = None
    if restrict_targets:
        candidate_token_ids = [
            tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1]
            for answer_choice in choices
        ]

    for iter_num, batch in enumerate(dataloader):
        if (iter_num + 1) % int(2000 / dataloader.batch_size) == 0:
            print(
                f"{iter_num * dataloader.batch_size+1}/ {len(dataloader.dataset)} inputs processed"
            )
            sys.stdout.flush()

        input_ids, targets, mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        input_ids = input_ids.to("cuda")
        outputs = model(input_ids)
        logits = outputs.logits

        seq_len = torch.sum(mask, dim=1)
        batch_probs = torch.softmax(
            logits[torch.arange(input_ids.shape[0]), seq_len - 1, :], dim=-1
        )
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        predictions += batch_prediction_indices.tolist()
        ground_truths += tokenizer.batch_decode(targets, skip_special_tokens=True)

    assert len(predictions) == len(
        dataloader.dataset
    ), "number of predictions should be equal to number of prompts"

    # get the metrics
    cors = []
    for i in range(len(predictions)):
        prediction = choices[predictions[i]]
        ground_truth = ground_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    model.train()

    return acc


class measure_statistics:
    def __init__(
        self,
        stats,
        model,
        val_loader,
        test_loader,
        tokenizer,
        accelerator,
        ckpt_dir=None,
        results_dir=None,
        prepare_for_overlap=False,
        filename="",
    ):

        self.stats = stats
        self.train_stats = defaultdict(dict)
        self.results_dir = results_dir
        self.ckpt_dir = ckpt_dir
        self.tokenizer = tokenizer
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.filename = filename

        if prepare_for_overlap:
            assert ckpt_dir is not None
            target_layers = get_target_layers_llama(
                model=model,
                n_layer=model.config.num_hidden_layers,
                option="norm1",
                every=8,
                world_size=accelerator.num_processes,
            )

            self.target_layers = target_layers
            self.base_indices = defaultdict(dict)

            with open(f"{ckpt_dir}/0shot/statistics_target.pkl", "rb") as f:
                stats = pickle.load(f)
                self.subjects = np.array(stats["subjects"])

            accelerator.print("preparing for overlap")
            sys.stdout.flush()
            for shots in ["0shot", "5shot"]:
                for norm in ["unnorm"]:  # , "norm"]:
                    layer_indices = defaultdict()
                    for index, name in target_layers.items():
                        if index < 1:
                            continue
                        else:
                            act = torch.load(f"{ckpt_dir}/{shots}/l{index}_target.pt")
                            act = act.to(torch.float64).numpy()

                            if norm == "norm":
                                assert len(act.shape()) == 2, act.shape()

                                act = act / np.linalg.norm(act, axis=1, keepdims=True)
                                assert np.all(
                                    np.linalg.norm(act, axis=1) == np.ones(act.shape[0])
                                ), np.linalg.norm(act, axis=1)

                            _, dist_index, _, _ = compute_distances(
                                X=act,
                                n_neighbors=40 + 1,
                                n_jobs=1,
                                working_memory=2048,
                                range_scaling=40 + 1,
                                argsort=False,
                            )
                            layer_indices[index] = dist_index

                    self.base_indices[shots][norm] = layer_indices

            accelerator.print("preparation finished")
            sys.stdout.flush()

            self.embdims, self.dtypes = get_embdims(
                model, val_loader, list(target_layers.values())
            )

    def update(
        self,
        accelerator,
        model,
        completed_steps,
        epoch,
        loss=None,
        do_overlap=False,
        do_val=False,
        do_test=False,
    ):

        self.train_stats["epoch"][completed_steps] = epoch
        self.train_stats["iter"][completed_steps] = completed_steps
        if loss is not None:
            self.train_stats["loss"][completed_steps] = loss

        if do_val:
            acc = evaluate(
                model=model,
                dataloader=self.val_loader,
                tokenizer=self.tokenizer,
                restrict_targets=True,
            )
            logger.info(f"iter {completed_steps}. mmlu val accuracy: {acc:.4f}")
            self.train_stats["mmlu_val"][completed_steps] = acc

        if do_test:
            acc = evaluate(
                model=model,
                dataloader=self.test_loader,
                tokenizer=self.tokenizer,
                restrict_targets=True,
            )
            logger.info(f"mmlu test accuracy after epoch {epoch}: {acc:.4f}")
            self.train_stats["mmlu_val"][completed_steps] = acc

        if do_overlap:
            accelerator.print("overlap computation started")
            overlaps = compute_overlap(
                accelerator=accelerator,
                model=model,
                val_loader=self.val_loader,
                tokenizer=self.tokenizer,
                target_layers=self.target_layers,
                embdims=self.embdims,
                dtypes=self.dtypes,
                base_indices=self.base_indices,
                subjects=self.subjects,
                results_dir=self.results_dir,
                filename=self.filename,
                ckpt_dir=self.ckpt_dir,
            )
            self.train_stats["overlaps"][completed_steps] = overlaps
            for shot, shot_val in overlaps.items():
                for norm, norm_val in shot_val.items():
                    for k, k_val in norm_val.items():
                        #logger.info(
                        #    f"iter {completed_steps}. overlap with subjects outputs {shot}, {norm}, {k}: {list(overlaps[shot][norm][k].values())[-1]}\n"
                        #)
                        logger.info(
                            f"iter {completed_steps}. overlap outputs {shot}, {norm}, {k}: {np.mean(list(list(overlaps[shot][norm][k].values())[-1].values())):.4f}\n"
                        )

        self.stats["train_stats"] = self.train_stats
        with open(
            f"{self.results_dir}/train_statistics_{self.filename}.pkl", "wb"
        ) as f:
            pickle.dump(self.stats, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
