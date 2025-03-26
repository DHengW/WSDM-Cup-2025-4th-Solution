#!/bin/bash
set -e



athene_path=../model_path/qwen2
athene_save_path=../model_save/qwen2_4bit_load_fintune/


fold=$1
echo run:${fold}

python3 safeTensor2bin.py --model_path ${athene_path} --safe_lora_path ${athene_save_path} --fold ${fold}

sh run_infer_athene.sh qwen2 ${athene_path} ${athene_save_path} ${fold}