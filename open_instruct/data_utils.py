# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Sequence
import pandas as pd
import transformers
from datasets import Dataset
import torch
from dataclasses import dataclass
from functools import partial
import sys
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

IGNORE_INDEX = -100
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(example, num_answers=4, include_answer=True):
    choices = ["A", "B", "C", "D"]
    prompt = example["question"]
    for j in range(num_answers):
        prompt += "\n{}. {}".format(choices[j], example[choices[j]])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(example["labels"])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def encode_open_instruct_mmlu(
    example, tokenizer, dev_df, max_seq_length, num_few_shots=0
):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    subject = example["subject"]
    prompt_end = format_example(example, include_answer=False)
    train_prompt = gen_prompt(dev_df, subject, num_few_shots)
    prompt = train_prompt + prompt_end

    tokenized_example = tokenizer(
        prompt, return_tensors="pt", max_length=max_seq_length, truncation=False
    )

    tokenized_labels = tokenizer(example["answers"], return_tensors="pt")

    input_ids = tokenized_example.input_ids
    labels = tokenized_labels.input_ids
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def get_mmlu_open_instruct(
    filepath,
    tokenizer,
    data_fold="val",
    max_seq_length=4096,
    num_processes=6,
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
        remove_columns=[
            name
            for name in test_df.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
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


@dataclass
class DataCollatorForCausalLM:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    max_seq_len: int

    # check if we can set padding value in labels == eos_token_id_directly (as the attention mask should take into account the gradient masking)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # in the structure of open-instruct the instances are already tensors, and already take into account max_seq_len

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
