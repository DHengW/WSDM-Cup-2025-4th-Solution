#!/bin/bash
set -e



llama_path=../model_path/qwen2
llama_path_ut=../model_save/qwen2_4bit_pretrain/pretrain_best_val_acc_model/adapter.bin

fold=$1
echo run:${fold}

sh run_finetune_athene.sh qwen2 ${llama_path} ${llama_path_ut} ${fold}