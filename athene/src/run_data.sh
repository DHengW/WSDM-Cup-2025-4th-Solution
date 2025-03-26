#!/bin/bash

athene_path=../model_path/athene



python prepare_data_ut_gen-qwen.py ${athene_path} athene
python prepare_data_noTTA-Gen.py ${athene_path} athene
