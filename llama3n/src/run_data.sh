#!/bin/bash

llama_path=../model_path/Llama3N



python prepare_data_ut_gen-qwen.py ${llama_path} llama3n
python prepare_data_noTTA-Gen.py ${llama_path} llama3n
