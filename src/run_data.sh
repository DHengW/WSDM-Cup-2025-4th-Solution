#!/bin/bash

qwen_path=../model_path/qwen2

python prepare_data_noTTA-Gen.py ${qwen_path} qwen2

python prepare_data_ut_gen-qwen-s.py ${qwen_path} qwen2
