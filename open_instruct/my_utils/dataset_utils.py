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

import numpy as np
from datasets import load_dataset, concatenate_datasets
from collections import Counter


rng = np.random.default_rng(42)

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


# same as plain
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


def get_dataset_open_instruct(
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

# NEW VERSION (MAYBE IDENTICAL TO OLD)
# def get_mmlu_open_instruct(
#     filepath,
#     tokenizer,
#     data_fold="val",
#     max_seq_length=2048,
#     num_processes=1,
#     num_samples=None,
#     subjects=None,
# ):
#     all_subjects = sorted(
#         [
#             f.split("_test.csv")[0]
#             for f in os.listdir(os.path.join(filepath, "test"))
#             if "_test.csv" in f
#         ]
#     )

#     if subjects is not None:
#         assert all(
#             subj in subjects for subj in subjects
#         ), f"Some of the subjects you specified are not valid: {subjects}"
#         all_subjects = subjects
#         print(f"mmlu evaluation restricted to {all_subjects}")

#     print("loading dataframes")
#     test_list = []
#     dev_list = []
#     for subject in all_subjects:
#         dev_df = pd.read_csv(
#             os.path.join(filepath, "dev", subject + "_dev.csv"), header=None
#         )
#         test_df = pd.read_csv(
#             os.path.join(filepath, f"{data_fold}", subject + f"_{data_fold}.csv"),
#             header=None,
#         )
#         if num_samples and num_samples < test_df.shape[0]:
#             test_df = test_df.sample(num_samples, random_state=42)

#         test_df.rename(
#             columns={0: "question", 1: "A", 2: "B", 3: "C", 4: "D", 5: "answers"},
#             inplace=True,
#         )
#         dev_df.rename(
#             columns={0: "question", 1: "A", 2: "B", 3: "C", 4: "D", 5: "answers"},
#             inplace=True,
#         )
#         test_df["subject"] = subject
#         dev_df["subject"] = subject
#         test_list.append(test_df)
#         dev_list.append(dev_df)

#     test_df = Dataset.from_pandas(pd.concat(test_list))
#     dev_df = Dataset.from_pandas(pd.concat(dev_list))

#     encode_function = partial(
#         encode_open_instruct_mmlu,
#         tokenizer=tokenizer,
#         dev_df=dev_df,
#         max_seq_length=max_seq_length,
#         num_few_shots=0,
#     )

#     lm_datasets = test_df.map(
#         encode_function,
#         batched=False,  # True,  # False,
#         num_proc=num_processes,
#         load_from_cache_file=False,
#         # remove_columns=[
#         #     name
#         #     for name in test_df.column_names
#         #     if name not in ["prompt", "input_ids", "labels", "attention_mask"]
#         # ],
#     )

#     lm_datasets.set_format(type="pt")
#     tot_examples = lm_datasets.num_rows
#     lm_datasets = lm_datasets.filter(
#         lambda example: len(example["input_ids"]) < max_seq_length
#     )
#     tot_filtered_examples = lm_datasets.num_rows

#     if tot_filtered_examples < tot_examples:
#         diff = tot_examples - tot_filtered_examples
#         print(
#             f"you filter out {diff} examples, {diff/tot_examples*100: .2f}% of the total"
#         )
#         sys.stdout.flush()

#     return lm_datasets


# ******************************************************************************


# old_version
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


# ******************************************************************************************************


def filter_out_long_sequences(tokenized_dataset, max_seq_len):

    tot_examples = tokenized_dataset.num_rows
    tokenized_datasets = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) < max_seq_len
    )
    tot_filtered_examples = tokenized_datasets.num_rows

    if tot_filtered_examples < tot_examples:
        diff = tot_examples - tot_filtered_examples
        print(
            f"you filter out {diff} examples, {diff/tot_examples*100: .2f}% of the total"
        )
        sys.stdout.flush()
    return tokenized_dataset


# prompt builder
class MMLU_Dataset:
    # num_few_shots = # shots
    # model_name number_istences to remove
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        accelerator,
        num_few_shots=0,
        subject=None,
        num_processes=1,
        num_samples=None,
        split="test",
        train_on_dev=False,
        train_on_test=False,
        mask_path=None,
        samples_per_subject=None,
    ):

        self.dataset = "mmlu"
        self.subject = subject
        if subject is not None:
            self.dataset = f"mmlu:{self.subject}"
        self.answers = np.array(["A", "B", "C", "D"])
        self.num_few_shots = num_few_shots
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_processes = num_processes
        self.num_samples = num_samples
        self.accelerator = accelerator
        self.split = split
        self.train_on_dev = train_on_dev
        self.train_on_test = train_on_test
        self.mask_path = mask_path
        self.samples_per_subject = samples_per_subject

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def construct_question(self, question, choices, answer, include_answer=False):
        # added strip
        prompt = f"{question.strip()}\n"
        for i, choice in enumerate(choices):
            # added strip
            prompt += f"{self.answers[i]}. {choice.strip()}\n"
        # added space to final answers
        prompt += "Answer:"
        if include_answer:
            prompt += f" {self.answers[answer]}\n\n"
        return prompt

    # prompt contruction.buils to operate on list of inputs.
    def construct_prompt(self, batch, tokenizer, dev_set, max_seq_len, num_few_shots):
        prompts = []

        questions = batch["question"]  # list of strings
        subjects = batch["subject"]  # list of strings
        choices = batch["choices"]  # list of list of strings
        answer_indices = np.array(batch["answer"])  # array of integers

        # build a dict of subsets of the dev set with the subject of the batch
        if num_few_shots > 0:
            local_dev_set = {}
            for subject in set(subjects):
                local_dev_set[subject] = dev_set.filter(
                    lambda dev_example: dev_example["subject"] == subject,
                )

        for i in range(len(questions)):
            prompt = f"The following are multiple choice questions (with answers) about{self.format_subject(subjects[i])}.\n\n"
            current_subject = subjects[i]
            for j in range(num_few_shots):
                shot = local_dev_set[current_subject][j]
                prompt += self.construct_question(
                    shot["question"],
                    shot["choices"],
                    shot["answer"],
                    include_answer=True,
                )
            question = self.construct_question(
                questions[i], choices[i], answer_indices[i]
            )
            prompt += question
            prompts.append(prompt)

        # tokenization part
        tokenized_examples = [
            tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for prompt in prompts
        ]

        # targets are tokenized with space included
        tokenized_labels = [
            tokenizer(
                self.answers[index], return_tensors="pt", add_special_tokens=False
            ).input_ids.flatten()
            for index in answer_indices
        ]

        attention_mask = [
            torch.ones_like(input_ids) for input_ids in tokenized_examples
        ]

        return {
            "prompt": prompts,
            "answers": [self.answers[index] for index in answer_indices],
            "subjects": subjects,
            "input_ids": tokenized_examples,
            "labels": tokenized_labels,
            "attention_mask": attention_mask,
        }

    def construct_prompt_train(
        self, batch, tokenizer, dev_set, max_seq_len, num_few_shots=None
    ):
        # dev_set and few_shots are not used here
        prompts = []
        premises = []

        questions = batch["question"]  # list of strings
        subjects = batch["subject"]  # list of strings
        choices = batch["choices"]  # list of list of strings
        answer_indices = np.array(batch["answer"])  # array of integers

        for i in range(len(questions)):
            # prompt = f"The following are multiple choice questions (with answers) about{self.format_subject(subjects[i])}.\n\n"
            prompt = f"The following is a multiple choice question (with answers) about{self.format_subject(subjects[i])}.\n\n"
            question = self.construct_question(
                questions[i],
                choices[i],
                answer_indices[i],
                include_answer=False,
            )
            answer = f" {self.answers[answer_indices[i]]}"
            prompt = question + answer
            prompts.append(prompt)
            premises.append(question)

        # tokenization part
        tokenized_examples = [
            tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for prompt in prompts
        ]

        # tokenized questions
        tokenized_questions = [
            tokenizer(
                question,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for question in premises
        ]

        # mask out question part
        tokenized_labels = [example.clone() for example in tokenized_examples]

        for i, label_i in enumerate(tokenized_labels):
            label_i[: len(tokenized_questions[i])] = IGNORE_INDEX
            tokenized_labels[i] = label_i

        # double check
        for label in tokenized_labels:
            assert label[-1] != IGNORE_INDEX

        attention_mask = [
            torch.ones_like(input_ids) for input_ids in tokenized_examples
        ]

        return {
            "prompt": prompts,
            "answers": [self.answers[index] for index in answer_indices],
            "subjects": subjects,
            "input_ids": tokenized_examples,
            "labels": tokenized_labels,
            "attention_mask": attention_mask,
        }

    def construct_balanced(self, samples_per_subject, split, mask_path):
        assert samples_per_subject is not None or mask_path is not None

        dataset = load_dataset("cais/mmlu", "all", split=split)
        subjects = np.unique(dataset["subject"])

        if self.mask_path is not None:
            mask = np.load(self.mask_path)
            final = dataset.select(mask)
            frequences = Counter(final["subject"]).values()

            if self.split == "test":
                assert len(np.unique(list(frequences))) == 1
                assert np.unique(list(frequences))[0] == 100
                assert mask.shape[0] == 5700

        else:
            mask = []
            for sub in np.unique(subjects):
                ind = np.nonzero(sub == subjects)[0]
                nsamples = min(samples_per_subject, len(ind))
                chosen = rng.choice(ind, nsamples, replace=False)
                mask.extend(list(np.sort(chosen)))

            mask = np.array(mask)
            final = dataset.select(mask)

        return final

    def construct_dataset(self):
        """
        Construct the request instances for the scenario
        """
        # removed trust remote code

        split = self.split
        if self.split == "train":
            # training on the dev + validation datasets
            if self.train_on_dev:
                split = "dev"
            elif self.train_on_test:
                split = "test"
            else:
                split = "dev+validation"
            assert self.num_few_shots == 0

        self.accelerator.print(f"loading dataset\nsplit: {self.split}\nmode: {split}")

        if self.num_samples is not None:
            split = f"test[:{self.num_samples}]"

        if self.subject is not None:
            dataset = load_dataset("cais/mmlu", self.subject, split=split)
        elif self.split == "train" and split != "dev":
            dataset = self.construct_balanced(
                mask_path=self.mask_path,
                samples_per_subject=self.samples_per_subject,
                split=split,
            )
        else:
            dataset = load_dataset("cais/mmlu", "all", split=split)

        few_shot_dataset = None
        if self.num_few_shots > 0 and self.num_few_shots <= 5:
            few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev")
        elif self.num_few_shots > 5:
            few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev+validation")

        prompt_func = self.construct_prompt
        if self.split == "train":
            prompt_func = self.construct_prompt_train

        encode_function = partial(
            prompt_func,
            tokenizer=self.tokenizer,
            dev_set=few_shot_dataset,
            max_seq_len=self.max_seq_len,
            num_few_shots=self.num_few_shots,
        )
        self.accelerator.print("tokenization started")
        sys.stdout.flush()
        tokenized_dataset = dataset.map(
            encode_function,
            batched=True,
            batch_size=self.num_processes,
            num_proc=self.num_processes,
            load_from_cache_file=False,
        )

        self.accelerator.print("tokenization finished")
        sys.stdout.flush()

        def sort_by_token_length(example):
            return len(example["input_ids"])

        sorted_indices = sorted(
            range(len(tokenized_dataset)),
            key=lambda i: sort_by_token_length(tokenized_dataset[i]),
            reverse=True,
        )
        longest_sequences = tokenized_dataset.select(sorted_indices[:10])
        longest_sequences.set_format(type="pt")

        # remove examples loger than max seq len maybe not necessary at all
        # list of list is made list of tensors
        tokenized_dataset.set_format(type="pt")
        tokenized_dataset = filter_out_long_sequences(
            tokenized_dataset, self.max_seq_len
        )

        return tokenized_dataset, longest_sequences


# ******************************************************************************************


# def get_mmlu_qlora(mmlu_dir, tokenizer, protocol="fs"):
#     if protocol == "zs":
#         mmlu_dataset = load_dataset(
#             "json",
#             data_files={
#                 "eval": f"{mmlu_dir}/zero_shot_mmlu_val.json",
#                 "test": f"{mmlu_dir}/zero_shot_mmlu_test.json",
#             },
#         )
#         # mmlu_dataset = mmlu_dataset.remove_columns("subject")
#     # MMLU Five-shot (Eval/Test only)
#     elif protocol == "fs":
#         mmlu_dataset = load_dataset(
#             "json",
#             data_files={
#                 "eval": f"{mmlu_dir}/five_shot_mmlu_val.json",
#                 "test": f"{mmlu_dir}/five_shot_mmlu_test.json",
#             },
#         )
#         # mmlu_dataset = mmlu_dataset.remove_columns('subject')
#     dataset = mmlu_dataset["test"]

#     def tokenize_example(tokenizer, batch):
#         encoded_input = [
#             tokenizer.encode(f"{example}".strip(), bos=False, eos=False)
#             for example in batch["input"]
#         ]
#         encoded_input_and_output = [
#             tokenizer.encode(s + " " + t, bos=False, eos=False)
#             for s, t in zip(batch["input"], batch["output"])
#         ]

#         labels = copy.deepcopy(encoded_input_and_output)

#         for label, source in zip(labels, encoded_input):
#             label[: len(source)] = torch.tensor(
#                 [IGNORE_INDEX for _ in range(len(source))], dtype=torch.int
#             )

#         return {
#             "input_ids": encoded_input_and_output,
#             "labels": labels,
#             "subject": batch["subject"],
#         }

#     dataset = dataset.map(
#         partial(tokenize_example, tokenizer),
#         remove_columns=list(dataset.features),
#         batched=True,
#         num_proc=6,
#         load_from_cache_file=False,
#     )

#     return dataset


# **********************************************
