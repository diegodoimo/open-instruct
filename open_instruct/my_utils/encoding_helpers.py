import torch

IGNORE_INDEX = -100


def encode_plain(example, tokenizer, max_seq_length, text_field):
    example_text = example[text_field]
    tokenized_example = tokenizer(
        example_text.strip(),
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    # in the output they will be converted to lists but at least we can apply flatten
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example[
        "completion"
    ].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors="pt")
    # , max_length=max_seq_length, truncation=True
    # )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example["prompt"], return_tensors="pt")
    # max_length=max_seq_length,
    # truncation=True,
    # )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example[
        "completion"
    ].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"],
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


# def encode_with_messages_format(example, tokenizer, max_seq_length):
#     """
#     Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
#     We concatenate all messages with the roles as delimiters and tokenize them together.
#     """
#     messages = example["messages"]
#     if len(messages) == 0:
#         raise ValueError("messages field is empty.")

#     def _concat_messages(messages):
#         message_text = ""
#         for message in messages:
#             if message["role"] == "system":
#                 message_text += "<|system|>\n" + message["content"].strip() + "\n"
#             elif message["role"] == "user":
#                 message_text += "<|user|>\n" + message["content"].strip() + "\n"
#             elif message["role"] == "assistant":
#                 message_text += (
#                     "<|assistant|>\n"
#                     + message["content"].strip()
#                     + tokenizer.eos_token
#                     + "\n"
#                 )
#             else:
#                 raise ValueError("Invalid role: {}".format(message["role"]))
#         return message_text

#     example_text = _concat_messages(messages).strip()

#     tokenized_example = tokenizer(
#         example_text, return_tensors="pt"
#     )  #  max_length=max_seq_length, truncation=True)
#     input_ids = tokenized_example.input_ids
#     labels = input_ids.clone()

#     # mask the non-assistant part for avoiding loss
#     for message_idx, message in enumerate(messages):
#         if message["role"] != "assistant":
#             if message_idx == 0:
#                 message_start_idx = 0
#             else:
#                 # message_start_idx = tokenizer(
#                 #     _concat_messages(messages[:message_idx]),
#                 #     return_tensors="pt",
#                 #     max_length=max_seq_length,
#                 #     truncation=True,
#                 # ).input_ids.shape[1]
#                 message_start_idx = tokenizer(
#                     _concat_messages(messages[:message_idx]), return_tensors="pt"
#                 ).input_ids.shape[1]
#             if (
#                 message_idx < len(messages) - 1
#                 and messages[message_idx + 1]["role"] == "assistant"
#             ):
#                 # here we also ignore the role of the assistant
#                 messages_so_far = (
#                     _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
#                 )
#             else:
#                 messages_so_far = _concat_messages(messages[: message_idx + 1])

#             # message_end_idx = tokenizer(
#             #     messages_so_far,
#             #     return_tensors="pt",
#             #     max_length=max_seq_length,
#             #     truncation=True,
#             # ).input_ids.shape[1]
#             message_end_idx = tokenizer(
#                 messages_so_far,
#                 return_tensors="pt",
#             ).input_ids.shape[1]
#             labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

#             # if message_end_idx >= max_seq_length:
#             #     break

#     attention_mask = torch.ones_like(input_ids)
#     return {
#         "input_ids": input_ids.flatten(),
#         "labels": labels.flatten(),
#         "attention_mask": attention_mask.flatten(),
#     }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


# ****************************************************************
# mmlu #


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
        prompt,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=False,
        add_special_tokens=True,
    )

    tokenized_labels = tokenizer(example["answers"], return_tensors="pt")

    input_ids = tokenized_example.input_ids
    labels = tokenized_labels.input_ids
    attention_mask = torch.ones_like(input_ids)

    return {
        "prompt": prompt,
        "answers": example["answers"],
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }
