#!/bin/bash
set -e



qwen_path=../model_path/qwen2
qwen_path_ut=../model_save/qwen2_16bit_pretrain/best_val_acc_model/adapter.bin



fold=$1
echo run:${fold}

sh run_fintune-gen-distil.sh qwen2 ${qwen_path} ${qwen_path_ut} ${fold}
