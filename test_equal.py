import torch
import sys




print("loading")
sys.stdout.flush()

dirpath_actual = '/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/results/llama-2-7b'
actual = torch.load(f"{dirpath_actual}/base_model.model.model.layers.1.input_layernorm_outepoch2.pt")

#dirpath_original = "/u/area/ddoimo/ddoimo/open/geometric_lens/repo/results/validation/llama-2-7b/0shot"
dirpath_original = "/u/area/ddoimo/ddoimo/open/geometric_lens/repo/results/validation/llama-2-7b/0shot"
#expected = torch.load(f"{dirpath_original}/_fsdp_wrapped_module.model.layers.20._fsdp_wrapped_module.input_layernorm_hook_output_target.pt")
expected = torch.load(f"{dirpath_original}/l1_hook_output_target.pt")

print("testing")
sys.stdout.flush()

torch.testing.assert_close(actual, expected)

