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

    if rank is not None and rank == 0:
        torch.cuda.empty_cache()
        allocated = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
    elif rank is None:
        torch.cuda.empty_cache()
        allocated = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
        sys.stdout.flush()


def save_with_accelerate(accelerator, model, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )
