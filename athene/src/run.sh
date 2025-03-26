#!/bin/bash
# Process Post-pretrain
sh run_post_pretrain_ut_three_model.sh

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