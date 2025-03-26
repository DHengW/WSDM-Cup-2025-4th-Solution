#!/usr/bin/env python


import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pickle
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import sys

model_path = sys.argv[1]
save_name = sys.argv[2]
print("model_path:", model_path)
print("save_name:", save_name)

MODEL_NAME = model_path
MAX_LENGTH = 1024
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
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
tokenizer = get_tokenizer()
# tokenizer.pad_token_id = tokenizer.eos_token_id

# In[3]:


train = pd.read_csv("../data/ut_multi.csv")#.head(300)

# In[4]:


train['prompt'] = train['prompt'].apply(lambda x: str([x]))
train['response_a'] = train['response_a'].apply(lambda x: str([x]))
train['response_b'] = train['response_b'].apply(lambda x: str([x]))

# In[5]:


train['prompt'] = train['prompt'].apply(lambda x: "\n".join(eval(x)))


# In[6]:


def do(row):
    if row['winner_model_a'] == 1:
        return "A"
    else:
        return "B"


train['label'] = train.apply(lambda row: do(row), axis=1)
train['label'].value_counts()


# In[7]:


def do(x):
    try:
        null = 'nan'
        return "\n".join(eval(x))
    except:
        x = x.replace("[", "").replace("]", "").strip()
        return x


train['response_a_str'] = train['response_a'].apply(lambda x: do(x))
train['response_b_str'] = train['response_b'].apply(lambda x: do(x))


# In[8]:


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


# In[9]:


# texts = []
# texts_token_len = []
# for _, row in tqdm(train.iterrows()):
#     query = ' '.join(row['prompt'].split(' ')[:256])
#     answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
#     answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
#     prompt_len = 256
#     try:
#         query_len = len(tokenizer.encode(query))
#         answer_a_len = len(tokenizer.encode(answer_a))
#         answer_b_len = len(tokenizer.encode(answer_b))
#     except:
#         texts.append('skip')
#         continue
#     if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
#         query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
#             tokenizer.encode(query)[:prompt_len])
#         query_len = len(tokenizer.encode(query))
#         if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
#             remain_len = MAX_LENGTH - query_len
#             token_answer_a = tokenizer.encode(answer_a)
#             token_answer_b = tokenizer.encode(answer_b)
#             if len(token_answer_a) > len(token_answer_b):
#                 while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
#                         token_answer_b):
#                     token_answer_a = token_answer_a[:-1]
#                 while len(token_answer_a) + len(token_answer_b) > remain_len:
#                     token_answer_a = token_answer_a[:-1]
#                     if len(token_answer_a) + len(token_answer_b) > remain_len:
#                         token_answer_b = token_answer_b[:-1]
#             else:
#                 while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
#                         token_answer_a):
#                     token_answer_b = token_answer_b[:-1]
#                 while len(token_answer_a) + len(token_answer_b) > remain_len:
#                     token_answer_a = token_answer_a[:-1]
#                     if len(token_answer_a) + len(token_answer_b) > remain_len:
#                         token_answer_b = token_answer_b[:-1]
#             answer_a = tokenizer.decode(token_answer_a)
#             answer_b = tokenizer.decode(token_answer_b)
#     prompt = create_rounds(query, answer_a, answer_b)
#     texts.append(prompt)
#     texts_token_len.append(len(tokenizer.encode(prompt)))
def process_row(row):
    row = pd.Series(row)
    query = ' '.join(row['prompt'].split(' ')[:256])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:1400])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:1400])
    prompt_len = 256
    try:
        query_len = len(tokenizer.encode(query))
        answer_a_len = len(tokenizer.encode(answer_a))
        answer_b_len = len(tokenizer.encode(answer_b))
    except:
        return 'skip'
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
    prompts = create_rounds(query, answer_b, answer_a,tokenizer)
    return [prompt,prompts]

train_dicts = train.to_dict(orient='records')
with Pool(cpu_count()) as pool:
    # 使用imap保持结果的顺序，并且提供进度条
    texts = list(tqdm(pool.imap(process_row, train_dicts), total=len(train)))
train['text'] = [text[0] for text in texts]
train['text2'] = [text[1] for text in texts]

print('train',train.shape)

# In[10]:


# train2 = train.copy()


# def do(row):
#     if row['winner_model_a'] == 1:
#         return "B"
#     else:
#         return "A"


# train2['label'] = train2.apply(lambda row: do(row), axis=1)
# print(train2['label'].value_counts())

# texts = []
# texts_token_len = []
# for _, row in tqdm(train2.iterrows()):
    # query = ' '.join(row['prompt'].split(' ')[:256])
    # answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
    # answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
    # prompt_len = 256
    # try:
    #     query_len = len(tokenizer.encode(query))
    #     answer_a_len = len(tokenizer.encode(answer_a))
    #     answer_b_len = len(tokenizer.encode(answer_b))
    # except:
    #     texts.append('skip')
    #     continue
    # if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
    #     query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
    #         tokenizer.encode(query)[:prompt_len])
    #     query_len = len(tokenizer.encode(query))
    #     if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
    #         remain_len = MAX_LENGTH - query_len
    #         token_answer_a = tokenizer.encode(answer_a)
    #         token_answer_b = tokenizer.encode(answer_b)
    #         if len(token_answer_a) > len(token_answer_b):
    #             while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
    #                     token_answer_b):
    #                 token_answer_a = token_answer_a[:-1]
    #             while len(token_answer_a) + len(token_answer_b) > remain_len:
    #                 token_answer_a = token_answer_a[:-1]
    #                 if len(token_answer_a) + len(token_answer_b) > remain_len:
    #                     token_answer_b = token_answer_b[:-1]
    #         else:
    #             while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
    #                     token_answer_a):
    #                 token_answer_b = token_answer_b[:-1]
    #             while len(token_answer_a) + len(token_answer_b) > remain_len:
    #                 token_answer_a = token_answer_a[:-1]
    #                 if len(token_answer_a) + len(token_answer_b) > remain_len:
    #                     token_answer_b = token_answer_b[:-1]
    #         answer_a = tokenizer.decode(token_answer_a)
    #         answer_b = tokenizer.decode(token_answer_b)
    # prompt = create_rounds(query, answer_b, answer_a)
    # texts.append(prompt)
    # texts_token_len.append(len(tokenizer.encode(prompt)))
# def process_row(row):
#     row = pd.Series(row)
#     query = ' '.join(row['prompt'].split(' ')[:256])
#     answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
#     answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
#     prompt_len = 256
#     try:
#         query_len = len(tokenizer.encode(query))
#         answer_a_len = len(tokenizer.encode(answer_a))
#         answer_b_len = len(tokenizer.encode(answer_b))
#     except:
#         return 'skip'
#     if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
#         query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
#             tokenizer.encode(query)[:prompt_len])
#         query_len = len(tokenizer.encode(query))
#         if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
#             remain_len = MAX_LENGTH - query_len
#             token_answer_a = tokenizer.encode(answer_a)
#             token_answer_b = tokenizer.encode(answer_b)
#             if len(token_answer_a) > len(token_answer_b):
#                 while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
#                         token_answer_b):
#                     token_answer_a = token_answer_a[:-1]
#                 while len(token_answer_a) + len(token_answer_b) > remain_len:
#                     token_answer_a = token_answer_a[:-1]
#                     if len(token_answer_a) + len(token_answer_b) > remain_len:
#                         token_answer_b = token_answer_b[:-1]
#             else:
#                 while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
#                         token_answer_a):
#                     token_answer_b = token_answer_b[:-1]
#                 while len(token_answer_a) + len(token_answer_b) > remain_len:
#                     token_answer_a = token_answer_a[:-1]
#                     if len(token_answer_a) + len(token_answer_b) > remain_len:
#                         token_answer_b = token_answer_b[:-1]
#             answer_a = tokenizer.decode(token_answer_a)
#             answer_b = tokenizer.decode(token_answer_b)
#     prompt = create_rounds(query, answer_b, answer_a,tokenizer)
#     return prompt

# train2_dicts = train2.to_dict(orient='records')
# with Pool(cpu_count()) as pool:
#     # 使用imap保持结果的顺序，并且提供进度条
#     texts = list(tqdm(pool.imap(process_row, train2_dicts), total=len(train2)))
# train2['text'] = texts
# print('train2',train2.shape)

# In[11]:


def do(x):
    if x == "A":
        return 0
    else:
        return 1

def dos(x):
    if x == 0:
        return 1
    else:
        return 0

train['label'] = train['label'].apply(lambda x: do(x))
train['label2'] = train['label'].apply(lambda x: dos(x))



train = train.sample(frac=1., random_state=2023)
# train2 = train2.sample(frac=1., random_state=2023)



# train = train.drop_duplicates("id")
# train2 = train2.drop_duplicates("id")



train_all = train#pd.concat([train, train2], axis=0)
print('train_all',train_all.shape)
train_all['id'] = train_all.index


train_all['text_len'] = train_all['text'].apply(lambda x: len(x.split(' ')))



# train_all = train_all[train_all['text_len'] < 1200]

print('skip num: ',train_all[train_all['text']=='skip'].shape)

train_all = train_all[train_all['text']!='skip']

print(train_all['label'].value_counts())

with open(
        f"../data/processed_data/ut_{save_name}_train_s.parquet",
        'wb') as f:
    pickle.dump(train_all, f)

with open(
        f"../data/processed_data/ut_{save_name}_dev_s.parquet",
        'wb') as f:
    pickle.dump(train_all.sample(n=100), f)