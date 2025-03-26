#!/bin/bash

llama_path=../model_path/Llama3N
llama_save_path=../model_save/llama3n_4bit_pretrain/pretrain_best_val_acc_model/

sh run_data.sh

# Process Post-pretrain
sh run_post_pretrain_ut_three_model.sh

python3 PreSafeTensor2bin.py --model_path ${llama_path} --safe_lora_path ${llama_save_path}

# Run Finetune with Post-pretrained weights
sh run_pipeline.sh 0
sh run_pipeline.sh 1
sh run_pipeline.sh 2
sh run_pipeline.sh 3
sh run_pipeline.sh 4

# Run Inference Code to generate Distil Data
sh run_pipeline-infer.sh 0
sh run_pipeline-infer.sh 1
sh run_pipeline-infer.sh 2
sh run_pipeline-infer.sh 3
sh run_pipeline-infer.sh 4