#!/usr/bin/env python
# coding=utf-8
import numpy as np
import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    # get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    LlamaConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from data_utils import get_mmlu_open_instruct, DataCollatorForCausalLM
import numpy as np
import sys
import time
import warnings


sys.path.append("/orfeo/cephfs/scratch/area/ddoimo/finetuning_llm/final/lit_gpt")
try:
    from my_utils.dataloader_utils import get_dataloader
except:
    sys.path.append("/home/diego/area_science/ricerca/finetuning_llm/final/lit_gpt")
    from my_utils.dataloader_utils import get_dataloader

from my_utils.helpers import print_memory_consumed
from my_utils.dataset_utils import get_dataset, get_mmlu_open_instruct
from my_utils.dataloader_utils import get_dataloader
from my_utils.optimizer_utils import get_optimizer, get_scheduler
from my_utils.tokenizer_utils import get_tokenizer

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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the training dataloader.",
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
        "--warmup_ratio",
        type=float,
        default=0,
        help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
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

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
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
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    world_size = accelerator.num_processes
    # **************************************************************************************
    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
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

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
        else:
            print("model_loading started. \n\n")
            print(config)
            sys.stdout.flush()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
            print("model loading finished. \n\n")
            sys.stdout.flush()
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
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
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("model loaded. \n\n")
    sys.stdout.flush()

    tokenizer = get_tokenizer(
        tokenizer_path=args.tokenizer_name, model_path=args.model_name_or_path
    )

    # ****************************************************************************

    train_dataset = get_dataset(
        filepath=args.train_file,
        data_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        num_processes=6,
        num_train_examples=None,
        dataset_info=None,
    )
    # longest_seq_length_train, _ = get_longest_seq_length(train_dataset)
    # model.max_seq_length = longest_seq_length_train
    # print(
    #     f"The longest sequence length in the train data is {longest_seq_length_train}, the model's maximum sequence length is"
    #     f" {model.max_seq_length} and context length is {model.config.block_size}"
    # )
    sys.stdout.flush()
    val_dataset = get_mmlu_open_instruct(
        filepath=args.val_file,
        tokenizer=tokenizer,
        data_fold="val",
        max_seq_length=2048,
        num_processes=6,
        num_samples=None,
        subjects=None,
    )
    test_dataset = None
    if args.test_file is not None:
        test_dataset = get_mmlu_open_instruct(
            filepath=args.test_file,
            tokenizer=tokenizer,
            data_fold="test",
            max_seq_length=2048,
            num_processes=6,
            num_samples=None,
            subjects=None,
        )

    # ******************************************************************************************

    train_loader, train_sampler = get_dataloader(
        train_dataset,
        args.per_device_train_batch_size,
        tokenizer.pad_token_id,
        max_seq_len=args.max_seq_length,
        world_size=world_size,
        shuffle=True,
        num_processes=6,
        return_sampler=True,
    )

    val_loader = get_dataloader(
        val_dataset,
        args.per_device_eval_batch_size,
        tokenizer.pad_token_id,
        max_seq_len=2048,
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
            max_seq_len=2048,
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
    # optimizer = fabric.setup_optimizers(optimizer)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        epochs=args.num_train_epochs,
        num_iters=len(train_loader),
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_iters=gradient_accumulation_iters,
    )

    # ***********************************************************************

    # # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps
    # )
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

    # # Create the learning rate scheduler.
    # # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # # number of updates in the end matches the num_training_steps here.
    # # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    # num_training_steps_for_scheduler = (
    #     args.max_train_steps
    #     if overrode_max_train_steps
    #     else args.max_train_steps * accelerator.num_processes
    # )
    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_training_steps=num_training_steps_for_scheduler,
    #     num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    # )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
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

    print_memory_consumed()
    print("before train run")
    sys.stdout.flush()
    acc = evaluate(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer,
            restrict_targets=True,
        )
    print(f"baseline average mmlu test accuracy: {acc:.4f}")
    print_memory_consumed()
    print("before after evaluate")
    
    for epoch in range(starting_epoch, args.num_train_epochs):
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
        # print("tot iter:", len(active_dataloader))
        start = time.time()
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()

                accelerator.backward(loss)
                # print("grad: ", model.model.model.embed_tokens.weight.grad)
                # torch.save(model.model.lm_head.weight.grad, "./results/lm_head_grad_hf_first.pt")
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
                    print(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Time: {t_tot/3600: .2f} hours"
                    )
                    print_memory_consumed()
                    sys.stdout.flush()

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
                    t0 = time.time()
                    acc = evaluate(
                        model=model,
                        dataloader=val_loader,
                        tokenizer=tokenizer,
                        restrict_targets=True,
                    )
                    t1 = time.time() - t0
                    t_tot -= t1
                    print(
                        f"baseline average mmlu val accuracy: {acc:.4f}, time {t1/60: .2f} min"
                    )
                    sys.stdout.flush()

                # if isinstance(checkpointing_steps, int):
                #    if completed_steps % checkpointing_steps == 0:
                #        output_dir = f"step_{completed_steps}"
                #        if args.output_dir is not None:
                #            output_dir = os.path.join(args.output_dir, output_dir)
                #        save_with_accelerate(
                #            accelerator, model, tokenizer, output_dir, args
                #        )

                if completed_steps >= args.max_train_steps:
                    break

        # if args.checkpointing_steps == "epoch":
        #    output_dir = f"epoch_{epoch}"
        #    if args.output_dir is not None:
        #        output_dir = os.path.join(args.output_dir, output_dir)
        #    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    acc = evaluate(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer,
            restrict_targets=True,
    )
    print(f"baseline average mmlu test accuracy: {acc:.4f}")
    print_memory_consumed()
    print("before after evaluate")
    

    # if args.with_tracking:
    #     accelerator.end_training()

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #     save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)


# *************************************************************************************************


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

    print("evaluating mmlu")
    sys.stdout.flush()

    for iter_num, batch in enumerate(dataloader):
        if (iter_num + 1) % int(1000 / dataloader.batch_size) == 0:
            print(
                f"{iter_num * dataloader.batch_size}/ {len(dataloader.dataset)} inputs processed"
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


if __name__ == "__main__":
    main()
