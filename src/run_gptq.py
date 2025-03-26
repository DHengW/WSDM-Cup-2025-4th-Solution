import torch
import sklearn
import numpy as np
import pandas as pd
import time
import pickle
from threading import Thread
from transformers import AutoTokenizer, LlamaModel
from transformers import LlamaForSequenceClassification, BitsAndBytesConfig,AutoModelForSequenceClassification,Gemma2ForCausalLM,AutoModelForCausalLM
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader,Dataset
from threading import Thread
from tqdm.auto import tqdm
import torch.nn.functional as F

def get_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        # use_fast=cfg.model.tokenizer.use_fast,
        add_eos_token=False,
        # truncation_side=cfg.model.tokenizer.truncation_side,
    )

    tokenizer.padding_side = "left"  # use left padding

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token = tokenizer.eod
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token = tokenizer.im_start
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token = tokenizer.im_end
            tokenizer.eos_token_id = tokenizer.im_end_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


MODEL_NAME = '../model_path/qwen2'
WEIGHTS_PATH = '../model_save/temp-merge'

def load_model(device):
    tokenizer = get_tokenizer(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                               # num_labels=2,
                                                               torch_dtype=torch.float16,
                                                               # attn_implementation="flash_attention_2"
                                                               # quantization_config=bnb_config
                                                              )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model_0 = PeftModel.from_pretrained(base_model, WEIGHTS_PATH)#'/root/autodl-tmp/temp-merge/')#.to('cuda:0')
    model_0=model_0.eval()
    # model_0.print_trainable_parameters()
    model_0 = model_0.merge_and_unload(safe_merge=True)
    return model_0,tokenizer

model,tokenizer = load_model('cuda:0')
model.config.pad_token_id = tokenizer.pad_token_id
model.save_pretrained("../model_save/temp-merge-save")
tokenizer.save_pretrained("../model_save/temp-merge-save")


import torch
from transformers import GPTQConfig, AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM,Gemma2ForCausalLM
from peft import PeftModel
import logging
import pandas as pd
import pickle

import logging
logging.basicConfig(
   format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)
WEIGHTS_PATH = "../model_save/temp-merge-save"
QUANTIZED_MODEL_DIR = "../model_save/final_model"
tokenizer = tokenizer = get_tokenizer(MODEL_NAME)
def build_dataset():
    with open('../data/processed_data/qwen2fold0/train_2048.parquet','rb') as f:
        test = pickle.load(f)
    train = test.sample(n=1024,random_state=2025)[['id','text']]
    return train

text_list = build_dataset()['text'].to_list()


gptq_config = GPTQConfig(bits=8, dataset = text_list,group_size=128,model_seqlen=11000,
                         tokenizer=tokenizer)

merged_quantized_model = AutoModelForCausalLM.from_pretrained(
    WEIGHTS_PATH,
    torch_dtype=torch.float16,
    device_map='cuda:0',
    quantization_config=gptq_config,
    )

merged_quantized_model.save_pretrained(QUANTIZED_MODEL_DIR)
tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)