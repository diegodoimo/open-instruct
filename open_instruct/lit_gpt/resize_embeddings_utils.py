from torch import nn
from typing import Optional
from accelerate.hooks import add_hook_to_module
import torch


def resize_embeddings(self, new_num_tokens, pad_to_multiple_of=None, padding_idx=None):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = get_resized_embeddings(
        self, old_embeddings, new_num_tokens, pad_to_multiple_of, padding_idx
    )
    if hasattr(old_embeddings, "_hf_hook"):
        hook = old_embeddings._hf_hook
        add_hook_to_module(new_embeddings, hook)

    old_embeddings_requires_grad = old_embeddings.weight.requires_grad
    new_embeddings.requires_grad_(old_embeddings_requires_grad)
    self.set_input_embeddings(new_embeddings)

    # Update new_num_tokens with the actual size of new_embeddings
    if pad_to_multiple_of is not None:
        new_num_tokens = new_embeddings.weight.shape[0]

    # if word embeddings are not tied, make sure that lm head is resized as well
    if self.get_output_embeddings() is not None:
        old_lm_head = self.get_output_embeddings()
        new_lm_head = get_resized_lm_head(self, old_lm_head, new_num_tokens)
        if hasattr(old_lm_head, "_hf_hook"):
            hook = old_lm_head._hf_hook
            add_hook_to_module(new_lm_head, hook)
        old_lm_head_requires_grad = old_lm_head.weight.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        self.set_output_embeddings(new_lm_head)

    if hasattr(self.lm_head, "lora_B"):
        old_lm_head = self.get_output_lora_B()
        new_lm_head = get_resized_lora_b(old_lm_head, new_num_tokens)
        old_lm_head_requires_grad = old_lm_head.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        self.set_output_lora_B(new_lm_head)

    return self.get_input_embeddings()


def get_resized_embeddings(
    self,
    old_embeddings: nn.Embedding,
    new_num_tokens: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    padding_idx: Optional[int] = None,
) -> nn.Embedding:
    """
    Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
    initialized vectors at the end. Reducing the size will remove vectors from the end

    Args:
        old_embeddings (`torch.nn.Embedding`):
            Old embeddings to be resized.
        new_num_tokens (`int`, *optional*):
            New number of tokens in the embedding matrix.

            Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
            vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
            `torch.nn.Embedding` module of the model without doing anything.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
            `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
            `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
            details about this, or help on choosing the correct value for resizing, refer to this guide:
            https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


    Return:
        `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
        `new_num_tokens` is `None`
    """

    if pad_to_multiple_of is not None:
        if not isinstance(pad_to_multiple_of, int):
            raise ValueError(
                f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
            )
        if new_num_tokens is None:
            new_num_tokens = old_embeddings.weight.shape[0]
        new_num_tokens = (
            (new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of
    else:
        print(
            "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
            f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
            " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
            " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
        )

    if new_num_tokens is None:
        return old_embeddings

    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if old_num_tokens == new_num_tokens:
        return old_embeddings

    if not isinstance(old_embeddings, nn.Embedding):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
            " should either use a different resize function or make sure that `old_embeddings` are an instance of"
            f" {nn.Embedding}."
        )

    # Build new embeddings

    # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
    # because the shape of the new embedding layer is used across various modeling files
    # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
    # to errors when training.
    new_embeddings = nn.Embedding(
        new_num_tokens,
        old_embedding_dim,
        padding_idx=padding_idx,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )

    # initialize all new embeddings (in particular added tokens)
    self._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights

    # numbers of tokens to copy
    n = min(old_num_tokens, new_num_tokens)

    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

    return new_embeddings


def get_resized_lm_head(
    self,
    old_lm_head: nn.Linear,
    new_num_tokens: Optional[int] = None,
    transposed: Optional[bool] = False,
) -> nn.Linear:
    """
    Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
    vectors at the end. Reducing the size will remove vectors from the end

    Args:
        old_lm_head (`torch.nn.Linear`):
            Old lm head liner layer to be resized.
        new_num_tokens (`int`, *optional*):
            New number of tokens in the linear matrix.

            Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
            vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
            `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
            to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
            vocab_size` else `vocab_size, lm_head_dim`.

    Return:
        `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
        `None`
    """
    if new_num_tokens is None:
        return old_lm_head

    old_num_tokens, old_lm_head_dim = (
        old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
    )

    if old_num_tokens == new_num_tokens:
        return old_lm_head

    if not isinstance(old_lm_head, nn.Linear):
        raise TypeError(
            f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
            " should either use a different resize function or make sure that `old_lm_head` are an instance of"
            f" {nn.Linear}."
        )

    # Build new lm head
    new_lm_head_shape = (
        (old_lm_head_dim, new_num_tokens)
        if not transposed
        else (new_num_tokens, old_lm_head_dim)
    )
    has_new_lm_head_bias = old_lm_head.bias is not None

    # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
    # because the shape of the new embedding layer is used across various modeling files
    # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
    # to errors when training.
    new_lm_head = nn.Linear(
        *new_lm_head_shape,
        bias=has_new_lm_head_bias,
        device=old_lm_head.weight.device,
        dtype=old_lm_head.weight.dtype,
    )

    # initialize new lm head (in particular added tokens)
    self._init_weights(new_lm_head)

    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

    _copy_lm_head_original_to_resized(
        new_lm_head,
        old_lm_head,
        num_tokens_to_copy,
        transposed,
        has_new_lm_head_bias,
    )

    return new_lm_head


def _copy_lm_head_original_to_resized(
    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
):
    # Copy old lm head weights to new lm head
    if not transposed:
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
            :num_tokens_to_copy, :
        ]
    else:
        new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[
            :, :num_tokens_to_copy
        ]

    # Copy bias weights to new lm head
    if has_new_lm_head_bias:
        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
            :num_tokens_to_copy
        ]


def get_resized_lora_b(
    old_lm_head: nn.Parameter,
    new_num_tokens: Optional[int] = None,
    transposed: Optional[bool] = False,
) -> nn.Parameter:
    """
    Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
    vectors at the end. Reducing the size will remove vectors from the end

    Args:
        old_lm_head (`torch.nn.Linear`):
            Old lm head liner layer to be resized.
        new_num_tokens (`int`, *optional*):
            New number of tokens in the linear matrix.

            Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
            vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
            `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
            to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
            vocab_size` else `vocab_size, lm_head_dim`.

    Return:
        `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
        `None`
    """
    if new_num_tokens is None:
        return old_lm_head

    old_num_tokens, old_lm_head_dim = (
        old_lm_head.size() if not transposed else old_lm_head.t().size()
    )

    if old_num_tokens == new_num_tokens:
        return old_lm_head

    device = old_lm_head.device
    dtype = old_lm_head.dtype
    new_lm_head = nn.Parameter(
        torch.zeros((new_num_tokens, old_lm_head_dim), device=device, dtype=dtype)
    )
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_lm_head.data[:num_tokens_to_copy, :] = old_lm_head.data[:num_tokens_to_copy, :]
    return new_lm_head
