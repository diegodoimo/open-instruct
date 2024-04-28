from transformers import AutoTokenizer, LlamaTokenizer


def get_tokenizer(tokenizer_path=None, model_path=None):

    assert tokenizer_path is not None or model_path is not None
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False
    )  # check what happens when true

    # num_added_tokens = tokenizer.add_special_tokens(
    #     {
    #         "bos_token": "<s>",
    #         "eos_token": "</s>",
    #         "unk_token": "<unk>",
    #         "pad_token": "<pad>",
    #     }
    # )
    # assert num_added_tokens in [
    #     0,
    #     1,
    # ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    # FOR THE MOMENT WE USE BATCH SIZE 1 FOR EVERYTHING
    num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # resize_model_embeddings(model, tokenizer)

    return tokenizer


# openinstruct

# if args.tokenizer_name:
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_name, use_fast=not args.use_slow_tokenizer
#     )
# elif args.model_name_or_path:
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name_or_path, use_fast=not args.use_slow_tokenizer
#     )
# else:
#     raise ValueError(
#         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
#         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
#     )

# # no default pad token for llama!
# # here we add all special tokens again, because the default ones are not in the special_tokens_map
# if isinstance(tokenizer, LlamaTokenizer) or isinstance(
#     tokenizer, LlamaTokenizerFast
# ):
#     num_added_tokens = tokenizer.add_special_tokens(
#         {
#             "bos_token": "<s>",
#             "eos_token": "</s>",
#             "unk_token": "<unk>",
#             "pad_token": "<pad>",
#         }
#     )
#     assert num_added_tokens in [
#         0,
#         1,
#     ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
# elif isinstance(tokenizer, GPTNeoXTokenizerFast):
#     num_added_tokens = tokenizer.add_special_tokens(
#         {
#             "pad_token": "<pad>",
#         }
#     )
#     assert (
#         num_added_tokens == 1
#     ), "GPTNeoXTokenizer should only add one special token - the pad_token."
# elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
#     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})

# # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# # on a small vocab and want a smaller embedding size, remove this test.
# # embedding_size = model.get_input_embeddings().weight.shape[0]
# # if len(tokenizer) > embedding_size:
# #     model.resize_token_embeddings(len(tokenizer))

# # print("model embedding resized. \n\n")
# # sys.stdout.flush()

# tokenizer.pad_token = "<pad>"
# tokenizer.pad_token_id = tokenizer.eos_token_id

# print("tokenizer loaded. \n\n")
# sys.stdout.flush()
