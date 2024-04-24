from collections import defaultdict
import torch
from dadapy.data import Data
from .extract_repr import extract_activations
from .pairwise_distances import compute_distances
import sys
import numpy as np


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
    target_layer_names = list(target_layers.values())
    target_layer_indices = list(target_layers.keys())

    model.eval()
    extr_act = extract_activations(
        accelerator,
        model,
        val_loader,
        target_layer_names,
        embdims,
        dtypes,
        use_last_token=True,
    )
    extr_act.extract(val_loader, tokenizer)
    extr_act.remove_hooks()

    act_dict = extr_act.hidden_states

    ov_0shot = {}
    ov_5shot = {}
    for i, (name, act) in enumerate(act_dict.items()):

        # act = act.to(torch.float64).numpy()
        # act, _, inverse = np.unique(act, axis=0, return_index=True, return_inverse=True)

        # assert act

        distances, dist_index, _, _ = compute_distances(
            X=act,
            n_neighbors=50 + 1,
            n_jobs=1,
            working_memory=2048,
            range_scaling=128,
            argsort=False,
        )

        # there are no overlapping datapoints in these representation:
        # indices_0shot = torch.load(
        #     f"{base_dir}/0shot/l{target_layer_indices[i]}_target_inverse.pt"
        # )

        # assert indices_0shot == inverse

        d = Data(distances=(distances, dist_index))

        dist = np.load(f"{base_dir}/0shot/l{target_layer_indices[i]}_dist.npy")
        indices = np.load(f"{base_dir}/0shot/l{target_layer_indices[i]}_index.npy")

        ov_0shot[name] = d.compute_data_overlap(distances=(dist, indices))

        dist = np.load(f"{base_dir}/5shot/l{target_layer_indices[i]}_dist.npy")
        indices = np.load(f"{base_dir}/5shot/l{target_layer_indices[i]}_index.npy")

        # repr_5shot = torch.load(
        #     f"{base_dir}/5shot/l{target_layer_indices[i]}_target.pt"
        # )
        ov_5shot[name] = d.compute_data_overlap(distances=(dist, indices))

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

    prefix = "base_model.model."
    middle = ""

    if world_size > 1:
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

    assert len(embdims) == len(target_layers), (
        f"num embdims: {len(embdims)}",
        f"num target layers: {len(target_layers)}",
    )
    return embdims, dtypes
