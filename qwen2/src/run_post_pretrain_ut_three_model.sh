#!/bin/bash
set -e


llama_path=../model_path/qwen2
llama_save_path=../model_save/qwen2_4bit_pretrain/pretrain_best_val_acc_model/

sh run_data.sh

sh run_post_pretrain-gen-athene.sh qwen2 ${llama_path}

python3 PreSafeTensor2bin.py --model_path ${llama_path} --safe_lora_path ${llama_save_path}






