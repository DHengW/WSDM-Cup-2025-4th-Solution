import time
start_time = time.time()
import time
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import pickle
import torch
import vllm
import sklearn
from typing import List
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
    SchedulerType,
    default_data_collator
)
import torch.nn.functional as F
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
# from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
# from peft import PeftModel
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
# os.environ["VLLM_USE_TRITON_FLASH_ATTN"]=1

@dataclass
class Config:
    model_dir = '../model_save/final_model'
    max_length = 10000
    batch_size = 4
    device = torch.device("cuda")    
    tta = False  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input

cfg = Config()
MAX_LENGTH = cfg.max_length

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

test = pd.read_parquet('../data/test.parquet')
sample_sub = pd.read_csv('../data/sample_submission.csv')
test.shape


def do(x):
    try:
        return "\n".join(eval(x))
    except:
        x = x.replace("[", "").replace("]", "").strip()
        return x
# test['prompt'] = test['prompt'].apply(lambda x: do(x))
test['response_a_str'] = test['response_a'].apply(lambda x: do(x))
test['response_b_str'] = test['response_b'].apply(lambda x: do(x))

def create_rounds(query, answer_a, answer_b,tokenizer):
    messages = [
        {"role": "system", 
         "content": '''You are a judge tasked with evaluating responses from two language models. Select the response that best meets the user's needs based on their query.

**Input:**
<Query>
User's original query.
</Query>

<Response_A>
First model's response.
</Response_A>

<Response_B>
Second model's response.
</Response_B>

**Output:**
Return only one letter:
- A for Response_A
- B for Response_B

**Guidelines:**
- Respond with only A or B.
- Do not provide explanations.
'''
        },
        {
            "role": "user", 
            "content": f'''Here is your input to process now-

<Query>
{query}
</Query>
{'---'*10}
<Response_A>
{answer_a}
</Response_A>
{'---'*10}
<Response_B>
{answer_b}
</Response_B>
'''
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text+' Choice: '


def process_row(row):
    row = pd.Series(row)
    query = ' '.join(row['prompt'].split(' ')[:256])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:4200])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:4200])
    prompt_len = 256
    query_len = len(tokenizer.encode(query))
    answer_a_len = len(tokenizer.encode(answer_a))
    answer_b_len = len(tokenizer.encode(answer_b))

    if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
        query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
            tokenizer.encode(query)[:prompt_len])
        query_len = len(tokenizer.encode(query))
        if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
            remain_len = MAX_LENGTH - query_len
            token_answer_a = tokenizer.encode(answer_a)
            token_answer_b = tokenizer.encode(answer_b)
            if len(token_answer_a) > len(token_answer_b):
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
                        token_answer_b):
                    token_answer_a = token_answer_a[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            else:
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
                        token_answer_a):
                    token_answer_b = token_answer_b[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            answer_a = tokenizer.decode(token_answer_a)
            answer_b = tokenizer.decode(token_answer_b)

    prompt = create_rounds(query, answer_a, answer_b,tokenizer)
    return prompt

test_dicts = test.to_dict(orient='records')
with Pool(cpu_count()) as pool:
    # 使用imap保持结果的顺序，并且提供进度条
    texts = list(tqdm(pool.imap(process_row, test_dicts), total=len(test)))
test['text'] = texts[:]
test['reverse']=False


#######################################----REVERSE----#######################################
test2 = test.copy()
def process_row(row):
    row = pd.Series(row)
    query = ' '.join(row['prompt'].split(' ')[:256])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:4200])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:4200])
    prompt_len = 256
    query_len = len(tokenizer.encode(query))
    answer_a_len = len(tokenizer.encode(answer_a))
    answer_b_len = len(tokenizer.encode(answer_b))
    
    if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
        query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
            tokenizer.encode(query)[:prompt_len])
        query_len = len(tokenizer.encode(query))
        if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
            remain_len = MAX_LENGTH - query_len
            token_answer_a = tokenizer.encode(answer_a)
            token_answer_b = tokenizer.encode(answer_b)
            if len(token_answer_a) > len(token_answer_b):
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
                        token_answer_b):
                    token_answer_a = token_answer_a[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            else:
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
                        token_answer_a):
                    token_answer_b = token_answer_b[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            answer_a = tokenizer.decode(token_answer_a)
            answer_b = tokenizer.decode(token_answer_b)
            
    prompt = create_rounds(query, answer_b, answer_a,tokenizer)
    return prompt

test2_dicts = test2.to_dict(orient='records')
with Pool(cpu_count()) as pool:
    # 使用imap保持结果的顺序，并且提供进度条
    texts = list(tqdm(pool.imap(process_row, test2_dicts), total=len(test2)))
# test2['text'] = texts
# test2['reverse']=True
test['text_reverse'] = texts
# test = pd.concat([test,test2],axis=0)
#######################################----REVERSE----#######################################

test['text_len'] = test['text'].apply(lambda x: len(x.split(' ')))
test = test.sort_values("text_len",ascending=False)
display(test.head(5))


def process_single_entry(data_entry):
    """处理单条数据的逻辑"""
    _, data = data_entry
    text = data['text']#.replace(self.tokenizer.eos_token, "<end>")
    reverse_text = data['text_reverse']
    features = tokenizer(text,padding=False,add_special_tokens=False,return_length=True)
    reverse_features = tokenizer(reverse_text,padding=False,add_special_tokens=False,return_length=True)
    prompt = tokenizer.decode(features["input_ids"], skip_special_tokens=False)
    reverse_prompt = tokenizer.decode(reverse_features["input_ids"], skip_special_tokens=False)
    return features['length'][0],prompt,reverse_prompt

lengths=[]
prompt_list=[]
reverse_prompt_list=[]

print("Preprocessing dataset in parallel...")
# 使用多线程处理数据
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(
        executor.map(process_single_entry, test.iterrows()), 
        total=len(test)
    ))

# 收集结果
for length,prompt,reverse_prompt in results:
    prompt_list.append(prompt)
    lengths.append(length)
    reverse_prompt_list.append(reverse_prompt)

test['lengths'] = lengths
test['prompt_list'] = prompt_list
test['reverse_prompt_list'] = reverse_prompt_list

test = test.sort_values("lengths",ascending=True)
display(test.head(5))

max_model_len = test.lengths.max()+1
print(max_model_len)

a_tok_id = tokenizer("A", add_special_tokens=False)["input_ids"][-1]
b_tok_id = tokenizer("B", add_special_tokens=False)["input_ids"][-1]
print(a_tok_id,b_tok_id)

def get_time(start_time=start_time):
    mid = time.time()
    elapsed_time_seconds = mid - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    return elapsed_time_minutes

midtime = get_time()
print('midtime: ',midtime)

if len(test)>20000:
    total_time = 720
elif len(test)>8000 and len(test)<20000:
    total_time = 285
else:
    total_time = int((len(test)/25000)*720)
print('total time: ',total_time)

llm = vllm.LLM(
    cfg.model_dir,
    quantization="gptq",#"awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.99,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=max_model_len,
    enable_prefix_caching=True,
)
llm.set_tokenizer(tokenizer)
# ------- Prepare -------------------------------------------------------------------#
sampling_params = vllm.SamplingParams(n=1, top_k=1, logprobs=10, max_tokens=1, temperature=0.0, skip_special_tokens=False, allowed_token_ids=[a_tok_id,b_tok_id])
responses = llm.generate(test['prompt_list'], sampling_params, use_tqdm=True)

print("inference done...")

scores = []

for response in responses:
    logprob_dict = response.outputs[0].logprobs[0]

    top_tok_ids = set(list(logprob_dict.keys()))

    a_logit, b_logit= -10.0, -10.0

    if a_tok_id in logprob_dict:
        a_logit = logprob_dict[a_tok_id].logprob

    if b_tok_id in logprob_dict:
        b_logit = logprob_dict[b_tok_id].logprob
    logits = np.array([a_logit, b_logit])
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    normalized_scores = exp_logits / np.sum(exp_logits)
    scores.append(normalized_scores)

test['predsA'] = np.array(scores)[:,0]
test['predsB'] = np.array(scores)[:,1]

llm_infer = get_time() - midtime
print(f"llm_infer: ",llm_infer)

remain_time = total_time - get_time()
print(f"remain_time: ",remain_time)

ratio = 0
alpha = (10/285)*total_time
if remain_time>1:
    ratio = (remain_time-alpha)/llm_infer
print("alpha: ",alpha,"remain ratio:", ratio)

if int(ratio*len(test)) >= 1:
    print("Running ratio:",int(ratio*len(test)))
    test['hard'] = abs(test['predsA']-test['predsB'])
    test = test.sort_values("hard",ascending=True)
    
    number_regen = int(ratio*len(test))
    regen_test = test.copy()[:number_regen]
    
    regen_test = regen_test.sort_values("lengths",ascending=True)
    
    regen_responses = llm.generate(regen_test['reverse_prompt_list'], sampling_params, use_tqdm=True)
    
    print("regen inference done...")
    
    scores = []
    
    for response in regen_responses:
        logprob_dict = response.outputs[0].logprobs[0]
    
        top_tok_ids = set(list(logprob_dict.keys()))
    
        a_logit, b_logit= -10.0, -10.0
    
        if a_tok_id in logprob_dict:
            a_logit = logprob_dict[a_tok_id].logprob
    
        if b_tok_id in logprob_dict:
            b_logit = logprob_dict[b_tok_id].logprob
        logits = np.array([a_logit, b_logit])
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        normalized_scores = exp_logits / np.sum(exp_logits)
        scores.append(normalized_scores)
    
    regen_test['regen_predsB'] = np.array(scores)[:,0]
    regen_test['regen_predsA'] = np.array(scores)[:,1]
    
    regen_test['predsA'] = (regen_test['predsA'] + regen_test['regen_predsA'])/2
    regen_test['predsB'] = (regen_test['predsB'] + regen_test['regen_predsB'])/2
    
    remain = test.copy()[number_regen:]
    regen_test = regen_test.drop(columns=['regen_predsA','regen_predsB'])

if int(ratio*len(test)) >= 1:
    data = pd.concat([remain,regen_test])
else:
    data = test[test['reverse']==False]
#######################################----REVERSE----#######################################
# reverse = test[test['reverse']==True]

# reverse = reverse.rename(columns={'predsA':'predsB2','predsB':'predsA2'})
# data = data.merge(reverse[['id','predsA2','predsB2']],on='id')

# data['predsA'] = (data['predsA']+data['predsA2'])/2
# data['predsB'] = (data['predsB']+data['predsB2'])/2

#######################################----REVERSE----#######################################

data['winner'] = (data['predsA']>data['predsB']).apply(lambda x: 'model_a' if x==True else 'model_b').values
display(data.head())

sample_sub = sample_sub.drop(columns=['winner']).merge(data[['id','winner']],on='id')

sample_sub[['id','winner']].to_csv('../subsubmission.csv', index=False)
display(sample_sub)