import argparse
import json
import pickle
import time
import os
import math
import sys
import pandas as pd
import pickle
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from model import EediRanker
from transformers import set_seed, AutoConfig
import random
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification
)
from transformers import optimization

# import deepspeed
# from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from utils.sft_dataset import is_rank_0
import torch.distributed as dist
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import prepare_model_for_kbit_training
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
# import vllm  # noqa: F401, E402

import torch
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    ## Tensorboard logging
    parser.add_argument('--model_path',
                        required=True)
    parser.add_argument('--safe_lora_path',
                        required=True)
    parser.add_argument('--fold',
                        required=True)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                               # num_labels=2,
                                               torch_dtype=torch.bfloat16,
                                               # attn_implementation="flash_attention_2",
                                               quantization_config=bnb_config
                                              )
    lora_path = os.path.join( args.safe_lora_path, f"fold{args.fold}/best_val_acc_model/")
    
    model = PeftModel.from_pretrained(model, lora_path)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    save_dict = model_to_save.state_dict()
    final_d = {}
    for k, v in save_dict.items():
        if "lora" in k:
            final_d[k] = v
    save_path = os.path.join(lora_path,"adapter.bin")
    print(f"Saving to {save_path}")
    torch.save(final_d, save_path)