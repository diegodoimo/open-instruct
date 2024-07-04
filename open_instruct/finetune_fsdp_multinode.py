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
from torch.optim.lr_scheduler import LambdaLR

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

from my_utils.dataloader_utils import get_dataloader, DataCollatorForCausalLMEval
from my_utils.helpers import print_memory_consumed, save_with_accelerate
from my_utils.dataset_utils import (
    get_dataset_open_instruct_new,
    get_mmlu_open_instruct,  # old version
    MMLU_Dataset,
)
from my_utils.dataloader_utils import get_dataloader
from my_utils.optimizer_utils import get_optimizer, get_scheduler
from my_utils.tokenizer_utils import get_tokenizer
from my_utils.model_utils import get_model_hf

from overlap_utils.overlap_functions import compute_overlap
from overlap_utils.helpers import get_embdims, get_target_layers_llama

# with fully sharded daat parallel if we can make this working
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
)
from functools import partial
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from collections import defaultdict
import pickle
from overlap_utils.pairwise_distances import compute_distances
from peft import PeftModel
import torch.distributed as dist

# *******************************************************************

import numpy as np


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
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
        default=None,
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
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--use_rslora", action="store_true")
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
        type=int,
        default=10,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
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
    parser.add_argument(
        "--train_on_dev",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--train_on_test",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--samples_per_subject",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
    )
    parser.add_argument("--clip_grad_thresh", type=float, default=1.0)
    parser.add_argument("--lr_min_fact", type=float, default=0.01)
    parser.add_argument("--overlap_base_dir", type=str, default=None, help="")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--weight_samples", action="store_true")
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


def lambda_fn(module: torch.nn.Module):
    if isinstance(module, LlamaDecoderLayer):
        return True  # like transformer_auto_wrap_policy
    if isinstance(module, torch.nn.Linear) and all(
        p.requires_grad for p in module.parameters()
    ):
        return True  # wrap each trainable linear separately
    return False


def find_grad_accumulation_steps(args):
    gradient_accumulation_steps = max(
        1, int(args.batch_size / WORLD_SIZE / args.per_device_train_batch_size)
    )

    if args.batch_size % (WORLD_SIZE * args.per_device_train_batch_size) != 0:
        args.batch_size = (
            gradient_accumulation_steps * WORLD_SIZE * args.per_device_train_batch_size
        )
    return gradient_accumulation_steps, args.batch_size


def compute_weighted_ce(logits, labels, weights, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    weights = weights.to(shift_logits.device)
    # here we prefer to leave the magnitude of the average batch loss relatively
    # constant rather than enforcinf the exact global sample weights
    # weighted average instead of plain averge:
    weighted_loss = (loss * weights / weights.sum()).sum()
    return weighted_loss


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # # # we use fsdp also when world size ==1. accelerate issue in casting
    if WORLD_SIZE > 1:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    auto_wrap_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=False,
        ignored_modules=None,
        limit_all_gathers=True,
        use_orig_params=True,
        param_init_fn=None,
        sync_module_states=True,
        forward_prefetch=False,
        activation_checkpointing=False,
    )

    gradient_accumulation_steps, args.batch_size = find_grad_accumulation_steps(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_log_kwargs,
        fsdp_plugin=fsdp_plugin,
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

    if accelerator.is_main_process:
        if args.train_on_dev:
            args.output_dir += "/dev"
        elif args.train_on_test:
            args.output_dir += "/test_balanced"
        else:
            args.output_dir += "/dev_val_balanced"

        if args.samples_per_subject is not None:
            args.output_dir += f"_{args.samples_per_subject}samples"

        args.output_dir += f"/{args.num_train_epochs}epochs"
        if args.seed is not None:
            args.output_dir += f"+=_seed{args.seed}"
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    world_size = accelerator.num_processes

    # # *******************************************************
    # # Load pretrained model and tokenizer

    model = get_model_hf(
        accelerator=accelerator,
        model_name_or_path=args.model_name_or_path,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        precision=torch.bfloat16,
        use_flash_attention_2=args.use_flash_attn,
        activation_checkpointing=args.activation_checkpointing,
    )

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        if args.resume_from_checkpoint:
            accelerator.print("loading pretrained peft models")
            model = PeftModel.from_pretrained(model, args.resume_from_checkpoint)
        else:
            accelerator.print("Initializing LORA model...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=[
                    "q_proj",
                    "o_proj",
                    "v_proj",
                    "k_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                use_dora=args.use_dora,
                use_rslora=args.use_rslora,
            )
            model = get_peft_model(model, peft_config)

        if RANK == 0:
            model.print_trainable_parameters()

    # ****************************************************************************
    tokenizer = get_tokenizer(
        tokenizer_path=args.tokenizer_name, model_path=args.model_name_or_path
    )

    # max_seq_len = model.config.max_position_embeddings
    max_seq_len = 2048
    if args.max_seq_length is not None and args.model_name_or_path is not None:
        max_seq_len = args.max_seq_length
    accelerator.print("max_seq_len: ", max_seq_len)

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
        max_seq_len=max_seq_len,
        num_few_shots=0,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        split="train",
        train_on_dev=args.train_on_dev,
        train_on_test=args.train_on_test,
        mask_path=args.mask_path,
        samples_per_subject=args.samples_per_subject,
    ).construct_dataset()
    accelerator.print(f"num_samples = {len(train_dataset)}")

    val_dataset, longest_seq = MMLU_Dataset(
        tokenizer=tokenizer,
        max_seq_len=1024,
        num_few_shots=0,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        split="validation",
    ).construct_dataset()

    test_dataset, longest_seq = MMLU_Dataset(
        tokenizer=tokenizer,
        max_seq_len=1024,
        num_few_shots=0,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        split="test",
    ).construct_dataset()

    global int_to_subject
    global subject_to_int

    int_to_subject = {
        i: subject for i, subject in enumerate(np.unique(test_dataset["subjects"]))
    }
    subject_to_int = {subj: i for i, subj in int_to_subject.items()}

    # ******************************************************************************************
    # ignore the issue of pad token in Llamas
    assert args.per_device_train_batch_size == 1
    assert args.per_device_eval_batch_size == 1

    # # DataLoaders creation:
    train_loader, sampler = get_dataloader(
        dataset=train_dataset,
        batch_size=args.per_device_train_batch_size,
        pad_token_id=tokenizer.pad_token_id,
        world_size=world_size,
        shuffle=True,
        num_processes=6,
        weight_samples=args.weight_samples,
        return_sampler=True,
    )

    val_loader = get_dataloader(
        val_dataset,
        args.per_device_eval_batch_size,
        tokenizer.pad_token_id,
        world_size=world_size,
        shuffle=False,
        num_processes=6,
        collate_fn=DataCollatorForCausalLMEval(tokenizer.pad_token_id),
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
            collate_fn=DataCollatorForCausalLMEval(tokenizer.pad_token_id),
        )

    # *******************************************************************************

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )

    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Prepare everything with `accelerator` model must be prepared before givin it to the optimizer.

    accelerator.print("memory consumed before loading model")
    print_memory_consumed(rank=RANK)
    sys.stdout.flush()
    model = accelerator.prepare(model)
    accelerator.print("memory consumed after loading model")
    print_memory_consumed(rank=RANK)
    sys.stdout.flush()

    # should be done after wrapping the model in FSDP
    if args.activation_checkpointing:
        accelerator.print("preparing activation checkpointing..")
        sys.stdout.flush()
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        # non_reentrant_wrapper = partial(
        # checkpoint_wrapper,
        #   offload_to_cpu=False,
        #   checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        # )
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    # ********************************************************************************

    # model must be alredy prepared here!
    accelerator.print("setup scheduler and optimizer..")
    sys.stdout.flush()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # "use_orig_parameters" needed for this split of parameters!
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
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

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.warmup_steps is None and args.warmup_ratio is None:
        warmup_steps = 0
    elif args.warmup_steps is None:
        warmup_steps = int(args.warmup_ratio * args.max_train_steps)

    scheduler = lambda x: min(
        args.lr_min_fact + (1 - args.lr_min_fact) * min(x, warmup_steps) / warmup_steps,
        args.lr_min_fact
        + 0.5
        * (1 - args.lr_min_fact)
        * (
            1
            + math.cos(
                max(0, x - warmup_steps)
                / (args.max_train_steps - warmup_steps)
                * math.pi
            )
        ),
    )
    lr_scheduler = LambdaLR(optimizer, lambda x: scheduler(x))

    # model must be prepared before initializing th optimizer
    # for the optimizer and lr the accelerator. prepare is needed to skip the update inside accclerator.accumulate, otherwise is not needed

    # optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # accelerator.print("final setup steps..")
    # sys.stdout.flush()
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_loader) / gradient_accumulation_steps
    # )

    # args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # ***************************************************************************************
    filename = ""
    if args.out_filename != "":
        filename = "_" + args.out_filename

    eval_steps, _ = get_cpt_steps(args.eval_steps, args.max_train_steps, logspace=False)
    checkpointing_steps, _ = get_cpt_steps(
        args.checkpointing_steps, args.max_train_steps, logspace=True
    )
    log_steps, log_interval = get_cpt_steps(
        args.logging_steps, args.max_train_steps, logspace=False
    )

    stats = defaultdict()

    stats["num_epochs"] = args.num_train_epochs
    stats["lr"] = args.learning_rate
    stats["scheduler"] = args.lr_scheduler_type
    stats["batch_size"] = args.batch_size
    stats["weight_decay"] = args.weight_decay
    stats["lora_rank"] = args.lora_rank
    stats["lora_alpha"] = args.lora_rank
    stats["lora_dropout"] = args.lora_dropout

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

    if args.save_checkpoint:
        output_dir = f"epoch_0"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        # save pretrained model
        accelerator.print("saving pretrained model at initialization..")
        sys.stdout.flush()
        save_with_accelerate(accelerator, model, output_dir, args)

    if args.measure_baselines:
        accelerator.print("measuring baselines..")
        sys.stdout.flush()

        meter.update(
            accelerator=accelerator,
            model=model,
            completed_steps=0,
            epoch=0,
            do_overlap=args.measure_overlap,
            do_val=True,
            do_test=args.eval_only,
        )

    if args.eval_only:
        assert False, "Evaluation mode selected. Progam ended."

    accelerator.print("start training")
    accelerator.print("memory before train run")
    sys.stdout.flush()
    print_memory_consumed(rank=RANK)

    # *******************************************************************************

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Weight Decay = {args.weight_decay}")
    logger.info(f"  Lora Rank = {args.lora_rank}")
    logger.info(f"  Lora Alpha = {args.lora_alpha}")
    logger.info(f"  Lora Dropout = {args.lora_dropout}")
    logger.info(f"  Use Dora = {args.use_dora}")
    logger.info(f"  Use rslora = {args.use_rslora}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total batch size (w. parallel, distributed & accumulation) = {args.batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  len_dataloader = {len(train_loader)}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Warmup steps = {warmup_steps}")
    logger.info(f"  Log steps number = {len(log_steps)}")

    completed_steps = 0
    total_loss = 0
    total_time = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        start = time.time()
        num_tokens = 0
        # gradient accumulation step may not finish with a proper update at the end of the epoch so we call zero grad here
        optimizer.zero_grad()
        if WORLD_SIZE > 1:
            sampler.set_epoch(epoch)
        for index, batch in enumerate(train_loader):

            if WORLD_SIZE == 1:
                # NO FSDP
                batch = {key: val.to("cuda") for key, val in batch.items()}
                # outputs = model(**batch, use_cache=False)
                # loss = outputs.loss
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                loss = compute_weighted_ce(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    weights=batch["weight"],
                    vocab_size=model.config.vocab_size,
                )

                loss = loss / gradient_accumulation_steps
                loss.backward()
                total_loss += loss.detach().float()
                num_tokens += batch["input_ids"].numel()

                if (index + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_thresh
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:

                if (index + 1) % gradient_accumulation_steps == 0:
                    # # fsdp wrapper automatically puts the tensor to gpu device
                    num_tokens += batch["input_ids"].numel()

                    # outputs = model(**batch, use_cache=False)
                    # loss = outputs.loss

                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = compute_weighted_ce(
                        logits=outputs.logits,
                        labels=batch["labels"],
                        weights=batch["weight"],
                        vocab_size=model.config.vocab_size,
                    )

                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    total_loss += loss.detach().float()

                    # gradient clipping must take into acocunt the fsdp wrapping
                    model.clip_grad_norm_(args.clip_grad_thresh)
                    # parameter update
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                else:
                    # # FSDP NO SYNC FUNCTION we do not need to compute gradients
                    with model.no_sync():
                        # 1 forward and 1 backward must be inside the context manager
                        num_tokens += batch["input_ids"].numel()

                        # outputs = model(**batch, use_cache=False)
                        # loss = outputs.loss

                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                        loss = compute_weighted_ce(
                            logits=outputs.logits,
                            labels=batch["labels"],
                            weights=batch["weight"],
                            vocab_size=model.config.vocab_size,
                        )

                        loss = loss / gradient_accumulation_steps
                        # no need for grad scaler
                        loss.backward()
                        total_loss += loss.detach().float()

            # with accelerator.accumulate(model):
            #     outputs = model(**batch, use_cache=False)
            #     loss = outputs.loss
            #     # We keep track of the loss at each logged step
            #     total_loss += loss.detach().float()
            #     accelerator.backward(loss)
            #     # clip gradient norm. don't do this with deepspeed
            #     if accelerator.sync_gradients and args.clip_grad_norm > 0:
            #         accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            #     optimizer.step()
            #     lr_scheduler.step()
            #     optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            if (index + 1) % gradient_accumulation_steps == 0:
                completed_steps += 1
                total_time += time.time() - start

                if completed_steps in log_steps:
                    accelerator.print(f"log step: {completed_steps}/{log_steps[-1]}")
                    sys.stdout.flush()

                    if WORLD_SIZE > 1:
                        total_loss = total_loss.reshape(1)
                        dist.all_reduce(total_loss)

                    avg_loss = (
                        total_loss.item()
                        / WORLD_SIZE
                        / gradient_accumulation_steps
                        / log_interval
                    )

                    accelerator.print(
                        f"LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, \
                            Time: {int(total_time//3600)} h {(total_time%3600)/60: .2f} min"
                    )
                    total_loss = 0
                    meter.update(
                        accelerator=accelerator,
                        model=model,
                        loss=avg_loss,
                        completed_steps=completed_steps,
                        epoch=epoch,
                    )

                if completed_steps in eval_steps:
                    meter.update(
                        accelerator=accelerator,
                        model=model,
                        completed_steps=completed_steps,
                        epoch=epoch,
                        do_val=True,
                        do_overlap=args.measure_overlap,
                    )
                    print_memory_consumed(rank=RANK)
                    sys.stdout.flush()

                if completed_steps in checkpointing_steps and args.save_checkpoint:
                    accelerator.print("saving checkpoint")
                    sys.stdout.flush()
                    output_dir = (
                        f"{len(checkpointing_steps)}ckpts/step_{completed_steps}"
                    )
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    save_with_accelerate(accelerator, model, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break
                start = time.time()

        if WORLD_SIZE > 1:
            num_tokens = torch.tensor([num_tokens]).to("cuda")
            dist.all_reduce(num_tokens)
            num_tokens = num_tokens.item()
        throughput = num_tokens / total_time / WORLD_SIZE
        accelerator.print(f"processed {throughput: .2f} token/sec/gpu")

        meter.update(
            accelerator=accelerator,
            model=model,
            completed_steps=completed_steps,
            epoch=epoch,
            do_test=True,
            do_overlap=args.measure_overlap,
            throughput=throughput,
        )
        print_memory_consumed(rank=RANK)

        # save model
        output_dir = f"epoch_{epoch+1}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        save_with_accelerate(accelerator, model, output_dir, args)

    if args.with_tracking:
        accelerator.end_training()


# FSDP has issues with `inference_mode`
# @torch.inference_mode()
@torch.no_grad()
def evaluate(model, dataloader, tokenizer):
    model.eval()

    predictions, ground_truths, subjects = [], [], []
    num_tokens = 0
    total_time = 0
    start = time.time()
    for iter_num, batch in enumerate(dataloader):

        if (iter_num + 1) % int(
            1000 / (dataloader.batch_size * WORLD_SIZE)
        ) == 0 and RANK == 0:
            print(
                f"{iter_num * dataloader.batch_size*WORLD_SIZE+1}/ {len(dataloader.dataset)} inputs processed"
            )
            sys.stdout.flush()

        input_ids, targets, mask = (
            batch["input_ids"].to("cuda"),
            batch["labels"].to("cuda"),
            batch["attention_mask"].to("cuda"),
        )
        outputs = model(input_ids)
        logits = outputs.logits
        seq_len = torch.sum(mask, dim=1)

        last_logits = logits[torch.arange(logits.shape[0]), seq_len - 1]
        predictions.extend(torch.argmax(last_logits, dim=-1, keepdims=True))
        ground_truths.extend(targets)
        subjects.extend(
            [
                torch.tensor([subject_to_int[subj]]).to("cuda")
                for subj in batch["subjects"]
            ]
        )
        num_tokens += batch["input_ids"].numel()

    total_time = time.time() - start

    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)
    subjects = torch.cat(subjects)

    if WORLD_SIZE > 1:
        pred_list = [torch.zeros_like(predictions) for _ in range(WORLD_SIZE)]
        gt_list = [torch.zeros_like(ground_truths) for _ in range(WORLD_SIZE)]
        subject_list = [torch.zeros_like(subjects) for _ in range(WORLD_SIZE)]

        dist.all_gather(pred_list, predictions)
        dist.all_gather(gt_list, ground_truths)
        dist.all_gather(subject_list, subjects)

        predictions = torch.cat(pred_list, dim=0).cpu()
        ground_truths = torch.cat(gt_list, dim=0).cpu()
        subjects = torch.cat(subject_list, dim=0)

        num_tokens = torch.tensor([num_tokens]).to("cuda")
        dist.all_reduce(num_tokens)
        num_tokens = num_tokens.item()

    throughput = num_tokens / total_time / WORLD_SIZE
    if RANK == 0:
        print(f"inference throughput: {throughput: .2f} token/sec/gpu")
        sys.stdout.flush()

    ground_truths = np.array([tokenizer.decode(tg).strip() for tg in ground_truths])
    predictions = np.array([tokenizer.decode(pred).strip() for pred in predictions])
    subjects = subjects.cpu().numpy()

    acc_pred = compute_accuracy(predictions, ground_truths, subjects)

    return acc_pred


def compute_accuracy(predictions, answers, subjects=None):

    accuracy = defaultdict(lambda: "not computed")
    tot_ans = len(predictions)
    num_correct = 0
    for pred, ans in zip(predictions, answers):
        if pred.strip() == ans.strip():
            num_correct += 1
    accuracy["micro"] = num_correct / tot_ans

    if subjects is not None:
        acc_subj = {}
        for subject in np.unique(subjects):
            mask = subject == subjects

            pred_tmp = predictions[mask]
            ans_tmp = answers[mask]

            tot_ans = len(ans_tmp)
            num_correct = 0
            for pred, ans in zip(pred_tmp, ans_tmp):
                if pred.strip() == ans.strip():
                    num_correct += 1
            acc_tmp = num_correct / tot_ans

            acc_subj[int_to_subject[subject]] = acc_tmp

        accuracy["subjects"] = acc_subj
        accuracy["macro"] = np.mean(list(acc_subj.values()))

    return accuracy


def get_cpt_steps(nsteps, max_train_steps, logspace=True):

    if logspace:
        steps = np.unique(
            np.around(np.geomspace(1, max_train_steps, nsteps, endpoint=False)).astype(
                int
            )
        )
        step = None
    else:
        step = max(1, int(np.around(max_train_steps / nsteps)))

        steps = np.arange(0, max_train_steps, step).astype(int)

    return steps, step


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
        self.accelerator = accelerator

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
                layer_indices = defaultdict()
                for index, name in target_layers.items():
                    if index < 1:
                        continue
                    else:
                        act = torch.load(f"{ckpt_dir}/{shots}/l{index}_target.pt")
                        act = act.to(torch.float64).numpy()

                        _, dist_index, _, _ = compute_distances(
                            X=act,
                            n_neighbors=40 + 1,
                            n_jobs=1,
                            working_memory=2048,
                            range_scaling=40 + 1,
                            argsort=False,
                        )
                        layer_indices[index] = dist_index

                self.base_indices[shots] = layer_indices

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
        throughput=None,
    ):

        self.train_stats["epoch"][completed_steps] = epoch
        self.train_stats["iter"][completed_steps] = completed_steps
        if loss is not None:
            self.train_stats["loss"][completed_steps] = loss
        if throughput is not None:
            self.train_stats["throughput"][completed_steps] = throughput

        if do_val:
            accelerator.print("measuring validation accuracy")
            sys.stdout.flush()
            acc = evaluate(
                model=model,
                dataloader=self.val_loader,
                tokenizer=self.tokenizer,
            )
            logger.info(
                f"iter {completed_steps}. mmlu val accuracy: macro {acc['macro']:.4f}, micro {acc['micro']:.4f}"
            )

            self.train_stats["mmlu_val_macro"][completed_steps] = acc["macro"]
            self.train_stats["mmlu_val_micro"][completed_steps] = acc["micro"]

        if do_test:
            accelerator.print("measuring test accuracy")
            sys.stdout.flush()
            acc = evaluate(
                model=model,
                dataloader=self.test_loader,
                tokenizer=self.tokenizer,
            )

            logger.info(
                f"mmlu test accuracy after epoch {epoch}: macro {acc['macro']:.4f}, micro {acc['micro']:.4f}"
            )

            self.train_stats["mmlu_test_macro"][completed_steps] = acc["macro"]
            self.train_stats["mmlu_test_micro"][completed_steps] = acc["micro"]

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
                logger.info(
                    f"iter {completed_steps}. overlap with subjects outputs {shot}: {list(overlaps[shot].values())[-1]}\n"
                )

        self.stats["train_stats"] = self.train_stats
        with open(
            f"{self.results_dir}/train_statistics_{self.filename}.pkl", "wb"
        ) as f:
            pickle.dump(self.stats, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    main()
