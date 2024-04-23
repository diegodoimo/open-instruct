from collections import defaultdict
import torch
from dadapy.data import Data
from .extract_repr import extract_activations
import sys


@torch.no_grad()
def compute_overlap(
    accelerator,
    model,
    val_loader,
    tokenizer,
    target_layers,
    embdims,
    dtypes,
    base_dir,
):
    model.eval()
    extr_act = extract_activations(
        accelerator,
        model,
        val_loader,
        target_layers,
        embdims,
        dtypes,
        use_last_token=True,
    )
    extr_act.extract(val_loader, tokenizer)
    extr_act.remove_hooks()

    act_dict = extr_act.hidden_states

    ov_0shot = {}
    ov_5shot = {}
    for name, act in act_dict.items():
        d = Data(coordinates=act.to(torch.float32).numpy())

        repr_0shot = torch.load(f"{base_dir}/0shot/l{name}_target.pt")
        ov_0shot[name] = d.compute_data_overlap(
            corrdinates=repr_0shot.to(torch.float32).numpy()
        )

        repr_5shot = torch.load(f"{base_dir}/5shot/l{name}_target.pt")
        ov_5shot[name] = d.compute_data_overlap(
            corrdinates=repr_5shot.to(torch.float32).numpy()
        )

    model.train()
    return ov_0shot, ov_5shot


def get_target_layers_llama(model, n_layer, option="norm1", every=1, world_size=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    prefix = "module."
    middle = ""
    # accelerate does not cast to bf16 a DDP model yet
    if world_size > 0:
        prefix = "_fsdp_wrapped_module."
        if map_names[option] != "":
            middle = "._fsdp_wrapped_module"

    target_layers = {
        i: f"{prefix}model.layers.{i}{middle}{suffix}" for i in range(0, n_layer, every)
    }

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
            embdims[name] = output.shape[-1]
            dtypes[name] = output.dtype

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

    assert len(embdims) == len(target_layers)
    return embdims, dtypes
