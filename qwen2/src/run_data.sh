#!/bin/bash

llama_path=../model_path/qwen2



python prepare_data_ut_gen-qwen.py ${llama_path} qwen2
python prepare_data_noTTA-Gen.py ${llama_path} qwen2
