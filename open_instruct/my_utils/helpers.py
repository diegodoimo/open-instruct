from pathlib import Path
import sys
import torch


def update_config(train_config, **kwargs):
    not_included = []
    for key in kwargs.keys():
        if not hasattr(train_config, key):
            not_included.append(key)

    if len(not_included) > 0:
        raise ValueError(f"{not_included} not valid configuration parameter(s).")
    else:
        for key, val in kwargs.items():
            # train_config.key = val  #why it its wrong?
            # train_config = dataclasses.replace(train_config, key=val)
            if isinstance(getattr(train_config, key), Path):
                val = Path(val)
            setattr(train_config, key, val)
    return train_config


def print_memory_consumed(rank=None):
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    if rank is not None and rank == 0:
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
    else:
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
    sys.stdout.flush()
