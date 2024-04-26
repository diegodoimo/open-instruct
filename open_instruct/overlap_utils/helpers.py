from collections import defaultdict
import sys
import torch


def get_target_layers_llama(model, n_layer, option="norm1", every=1, world_size=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    prefix = "base_model.model."
    middle = ""

    if world_size > 1:
        prefix = "_fsdp_wrapped_module."
        if map_names[option] != "":
            middle = "._fsdp_wrapped_module"

    target_layers = {
        i: f"{prefix}model.layers.{i}{middle}{suffix}" for i in range(0, n_layer, every)
    }

    target_layers[-1] = f"{prefix}model.embed_tokens"
    target_layers[n_layer] = f"{prefix}model.norm"
    target_layers[n_layer + 1] = f"{prefix}lm_head"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


@torch.no_grad()
def get_embdims(model, dataloader, target_layers):
    model = model.eval()
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name, embdims):
        def hook_fn(module, input, output):
            try:
                embdims[name] = input.shape[-1]
                dtypes[name] = input.dtype
            except:
                embdims[name] = input[0].shape[-1]
                dtypes[name] = input[0].dtype

        return hook_fn

    handles = {}
    for name, module in model.named_modules():
        if name in target_layers:
            handles[name] = module.register_forward_hook(get_hook(name, embdims))

    batch = next(iter(dataloader))
    sys.stdout.flush()
    _ = model(batch["input_ids"].to("cuda"))

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()

    assert len(embdims) == len(target_layers), (
        f"num embdims: {len(embdims)}",
        f"num target layers: {len(target_layers)}",
    )
    return embdims, dtypes
