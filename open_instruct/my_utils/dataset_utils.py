import os
import pandas as pd
from datasets import load_dataset, Dataset
import torch
import copy
from functools import partial
from .encoding_helpers import (
    encode_plain,
    encode_with_prompt_completion_format,
    encode_with_messages_format,
    encode_open_instruct_mmlu,
)
from datasets.utils.logging import disable_progress_bar
import sys


disable_progress_bar()


IGNORE_INDEX = -100


# ************************************************************************************************


def get_dataset_hf(
    filepath=None,
    data_name=None,
    tokenizer=None,
    max_seq_length=2048,
    num_processes=1,
    text_field="text",
    num_train_examples=None,
    dataset_info=None,
):
    assert data_name is not None or filepath is not None
    assert data_name is None or filepath is None

    split = "train"
    if num_train_examples is not None:
        split = f"train[:{num_train_examples}]"

    if filepath is not None:
        # file style is "text" modify the files if necessary
        raw_dataset = load_dataset("text", data_files=filepath, split=split)
    elif data_name is not None and dataset_info is None:
        raw_dataset = load_dataset(data_name, split=split)
    else:
        raw_dataset = load_dataset(data_name, dataset_info, split=split)

    if data_name is not None and data_name in ["sst"]:
        text_field = "sentence"
    encode_function = partial(
        encode_plain,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        text_field=text_field,
    )

    tokenized_dataset = raw_dataset.map(
        encode_function,
        batched=False,
        num_proc=num_processes,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in raw_dataset.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
    )
    # the output is always list of lists
    tokenized_dataset.set_format(type="pt")

    return tokenized_dataset


# def get_dataset_baroni(
#     filepath,
#     data_name,
#     tokenizer,
#     max_seq_length=2048,
#     num_processes=1,
#     text_field="text",
#     num_train_examples=None,
#     dataset_info=None,
# ):

#     raw_dataset = load_dataset(
#         "text", data_files=filepath, split=f"train[:{num_train_examples}]"
#     )
#     encode_function = partial(
#         encode_plain,
#         tokenizer=tokenizer,
#         max_seq_length=max_seq_length,
#         text_field=text_field,
#     )

#     tokenized_dataset = raw_dataset.map(
#         encode_function,
#         batched=False,
#         num_proc=num_processes,
#         load_from_cache_file=False,
#         remove_columns=[
#             name
#             for name in raw_dataset.column_names
#             if name not in ["input_ids", "labels", "attention_mask"]
#         ],
#     )

#     # the output is always list of lists
#     tokenized_dataset.set_format(type="pt")

#     return tokenized_dataset


# ******************************************************************************
def get_dataset_open_instruct_new(
    accelerator,
    filepath,
    tokenizer,
    max_seq_length=2048,
    num_processes=1,
):
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #     )
    # else:
    data_files = {}
    # dataset_args = {}
    if filepath is not None:
        data_files["train"] = filepath
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        # **dataset_args,
    )

    print("start preprocessing the data. \n\n")
    sys.stdout.flush()
    if (
        "prompt" in raw_datasets["train"].column_names
        and "completion" in raw_datasets["train"].column_names
    ):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names."
        )

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=num_processes,
            load_from_cache_file=False,
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )

        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(
            lambda example: (example["labels"] != -100).any()
        )

    train_dataset = lm_datasets["train"]
    print("finished preprocessing. \n\n")
    sys.stdout.flush()
    return train_dataset


def get_dataset_open_intruct(
    filepath,
    data_name,
    tokenizer,
    max_seq_length=2048,
    num_processes=1,
    text_field="text",
    num_train_examples=None,
    dataset_info=None,
):
    print("loading dataset")
    data_files = {}
    if filepath is not None:
        data_files["train"] = filepath
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    # Preprocessing the datasets.
    if (
        "prompt" in raw_datasets["train"].column_names
        and "completion" in raw_datasets["train"].column_names
    ):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        sys.stdout.flush()
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names."
        )
    print("tokenizing the datasets")
    # with accelerator.main_process_first():
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,  # True,  # False,
        num_proc=num_processes,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in raw_datasets["train"].column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
    )
    lm_datasets.set_format(type="pt")
    tot_examples = lm_datasets["train"].num_rows
    lm_datasets = lm_datasets.filter(lambda example: (example["labels"] != -100).any())

    lm_datasets = lm_datasets.filter(
        lambda example: len(example["input_ids"]) < max_seq_length
    )
    tot_filtered_examples = lm_datasets["train"].num_rows
    print(f"total examples: {tot_examples}")

    if tot_filtered_examples < tot_examples:
        diff = tot_examples - tot_filtered_examples
        print(
            f"you filter out {diff} examples, {diff/tot_examples*100: .2f}% of the total"
        )
        sys.stdout.flush()

    train_dataset = lm_datasets["train"]

    return train_dataset


# **************************************************************************************


def get_mmlu_open_instruct(
    filepath,
    tokenizer,
    data_fold="val",
    max_seq_length=2048,
    num_processes=1,
    num_samples=None,
    subjects=None,
):
    all_subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(filepath, "test"))
            if "_test.csv" in f
        ]
    )

    if subjects is not None:
        assert all(
            subj in subjects for subj in subjects
        ), f"Some of the subjects you specified are not valid: {subjects}"
        all_subjects = subjects
        print(f"mmlu evaluation restricted to {all_subjects}")

    print("loading dataframes")
    test_list = []
    dev_list = []
    for subject in all_subjects:
        dev_df = pd.read_csv(
            os.path.join(filepath, "dev", subject + "_dev.csv"), header=None
        )
        test_df = pd.read_csv(
            os.path.join(filepath, f"{data_fold}", subject + f"_{data_fold}.csv"),
            header=None,
        )
        if num_samples and num_samples < test_df.shape[0]:
            test_df = test_df.sample(num_samples, random_state=42)

        test_df.rename(
            columns={0: "question", 1: "A", 2: "B", 3: "C", 4: "D", 5: "answers"},
            inplace=True,
        )
        dev_df.rename(
            columns={0: "question", 1: "A", 2: "B", 3: "C", 4: "D", 5: "answers"},
            inplace=True,
        )
        test_df["subject"] = subject
        dev_df["subject"] = subject
        test_list.append(test_df)
        dev_list.append(dev_df)

    test_df = Dataset.from_pandas(pd.concat(test_list))
    dev_df = Dataset.from_pandas(pd.concat(dev_list))

    encode_function = partial(
        encode_open_instruct_mmlu,
        tokenizer=tokenizer,
        dev_df=dev_df,
        max_seq_length=max_seq_length,
        num_few_shots=0,
    )

    lm_datasets = test_df.map(
        encode_function,
        batched=False,  # True,  # False,
        num_proc=num_processes,
        load_from_cache_file=False,
        # remove_columns=[
        #     name
        #     for name in test_df.column_names
        #     if name not in ["prompt", "input_ids", "labels", "attention_mask"]
        # ],
    )

    lm_datasets.set_format(type="pt")
    tot_examples = lm_datasets.num_rows
    lm_datasets = lm_datasets.filter(
        lambda example: len(example["input_ids"]) < max_seq_length
    )
    tot_filtered_examples = lm_datasets.num_rows

    if tot_filtered_examples < tot_examples:
        diff = tot_examples - tot_filtered_examples
        print(
            f"you filter out {diff} examples, {diff/tot_examples*100: .2f}% of the total"
        )
        sys.stdout.flush()

    return lm_datasets


# ******************************************************************************************


def get_mmlu_qlora(mmlu_dir, tokenizer, protocol="fs"):
    if protocol == "zs":
        mmlu_dataset = load_dataset(
            "json",
            data_files={
                "eval": f"{mmlu_dir}/zero_shot_mmlu_val.json",
                "test": f"{mmlu_dir}/zero_shot_mmlu_test.json",
            },
        )
        # mmlu_dataset = mmlu_dataset.remove_columns("subject")
    # MMLU Five-shot (Eval/Test only)
    elif protocol == "fs":
        mmlu_dataset = load_dataset(
            "json",
            data_files={
                "eval": f"{mmlu_dir}/five_shot_mmlu_val.json",
                "test": f"{mmlu_dir}/five_shot_mmlu_test.json",
            },
        )
        # mmlu_dataset = mmlu_dataset.remove_columns('subject')
    dataset = mmlu_dataset["test"]

    def tokenize_example(tokenizer, batch):
        encoded_input = [
            tokenizer.encode(f"{example}".strip(), bos=False, eos=False)
            for example in batch["input"]
        ]
        encoded_input_and_output = [
            tokenizer.encode(s + " " + t, bos=False, eos=False)
            for s, t in zip(batch["input"], batch["output"])
        ]

        labels = copy.deepcopy(encoded_input_and_output)

        for label, source in zip(labels, encoded_input):
            label[: len(source)] = torch.tensor(
                [IGNORE_INDEX for _ in range(len(source))], dtype=torch.int
            )

        return {
            "input_ids": encoded_input_and_output,
            "labels": labels,
            "subject": batch["subject"],
        }

    dataset = dataset.map(
        partial(tokenize_example, tokenizer),
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=6,
        load_from_cache_file=False,
    )

    return dataset
