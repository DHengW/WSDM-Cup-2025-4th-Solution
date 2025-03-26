#!/bin/bash


#### input
name=$1
MODEL_PATH=$2
model_name="pretrain"
MODEL_USE="16bit"
echo $name
echo $MODEL_PATH
# exit 0
# 0,1,2,3,4,5,6,7
DATA_DIR=../data/processed_data/
ZERO_STAGE=2
OUTPUT=../model_save/${name}_${MODEL_USE}_${model_name}
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}
MASTER_PORT=20345
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3,4,5,6,7 train_gen_model_nice_memory-postpre-swap.py \
       --project_name ${name}_${MODEL_USE}_swap_raw \
       --lora_path "none" \
       --model_name ${model_name} \
       --train_dataset_path ${DATA_DIR}ut_${name}_train_s.parquet \
       --dev_dataset_path  ${DATA_DIR}ut_${name}_dev_s.parquet \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 2 \
       --gradient_accumulation_steps 8 \
       --max_prompt_len 2048 \
       --max_completion_len 256 \
       --earystop 0 \
       --save_batch_steps 8000 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 1  \
       --debug_code 0 \
       --learning_rate 5e-5 \
       --num_warmup_steps 100 \
       --weight_decay 0. \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing