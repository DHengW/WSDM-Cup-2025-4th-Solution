from safetensors.torch import load_file
from safetensors.torch import save_file

import torch
lora_dir = '../model_save/qwen2_16bit_load_fintune/fold0/kl_distil4_encode/1best_val_acc_model/adapter_model.safetensors'
d1 = load_file(lora_dir)
lora_dir = '../model_save/qwen2_16bit_load_fintune/fold1/kl_distil4_encode/1best_val_acc_model/adapter_model.safetensors'
d2 = load_file(lora_dir)
lora_dir = '../model_save/qwen2_16bit_load_fintune/fold2/kl_distil4_encode/1best_val_acc_model/adapter_model.safetensors'
d3 = load_file(lora_dir)
lora_dir = '../model_save/qwen2_16bit_load_fintune/fold3/kl_distil4_encode/1best_val_acc_model/adapter_model.safetensors'
d4 = load_file(lora_dir)
lora_dir = '../model_save/qwen2_16bit_load_fintune/fold4/kl_distil4_encode/1best_val_acc_model/adapter_model.safetensors'
d5 = load_file(lora_dir)

d = {}
for k, v in d1.items():
    v = d1[k] + d2[k] + d3[k] + d4[k] + d5[k]
    v = v / 5.
    d[k] = v
save_file(d, "../model_save/temp-merge/adapter_model.safetensors")#final_adapter.bin
