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

def random_swap_columns(df):
    # 确保必要的列存在于DataFrame中
    required_columns = ['response_a', 'response_b', 'winner_model_a', 'winner_model_b']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("DataFrame must contain the columns: " + ", ".join(required_columns))
    
    # 创建一个布尔mask，用于决定哪些行需要交换（True表示交换）
    swap_mask = np.random.rand(len(df)) < 0.5
    
    # 对需要交换的行进行列值的交换
    # 使用copy()避免设置with copy警告
    temp_response = df.loc[swap_mask, 'response_a'].copy()
    df.loc[swap_mask, 'response_a'] = df.loc[swap_mask, 'response_b']
    df.loc[swap_mask, 'response_b'] = temp_response
    
    temp_winner_model = df.loc[swap_mask, 'winner_model_a'].copy()
    df.loc[swap_mask, 'winner_model_a'] = df.loc[swap_mask, 'winner_model_b']
    df.loc[swap_mask, 'winner_model_b'] = temp_winner_model
    
    return df

model_path = sys.argv[1]
save_name = sys.argv[2]
print("model_path:", model_path)
print("save_name:", save_name)

MODEL_NAME = model_path
MAX_LENGTH = 1024
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

# In[3]:


train = pd.read_csv("../../data/ut_multi.csv")#.head(50)
train = random_swap_columns(train.copy()) 
print(train.winner_model_a.value_counts(),train.winner_model_b.value_counts())

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
         "content": '''You are a skilled judge evaluating responses from two large language models(LLMs). Your task is to select the response that best meets the user's needs based on the query provided.

**Input Format:**
<Query>
[User's original query to both LLMs]
</Query>

<Response_A>
[First LLM's response]
</Response_A>

<Response_B>
[Second LLM's response]
</Response_B>

**Your Task:**
Carefully analyze both <Response_A> and <Response_B> in relation to the Query. Determine which response is more likely to be selected by a user based on the following criteria:
- Completeness in addressing the query
- Accuracy of information
- Clarity and coherence
- Conciseness vs appropriate detail
- Helpful examples or explanations when needed
- Professional yet engaging tone
- Sound reasoning and logic
- Format and presentation

**Output:**
Respond with only a single letter:
- A if <Response_A> is better.
- B if <Response_B> is better.

**Important Notes:**
- Provide only the letter A or B as your response.
- No explanations are needed.
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
    answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
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
    return prompt

train_dicts = train.to_dict(orient='records')
with Pool(cpu_count()) as pool:
    # 使用imap保持结果的顺序，并且提供进度条
    texts = list(tqdm(pool.imap(process_row, train_dicts), total=len(train)))
train['text'] = texts

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

# # texts = []
# # texts_token_len = []
# # for _, row in tqdm(train2.iterrows()):
#     # query = ' '.join(row['prompt'].split(' ')[:256])
#     # answer_a = ' '.join(row['response_a_str'].split(' ')[:700])
#     # answer_b = ' '.join(row['response_b_str'].split(' ')[:700])
#     # prompt_len = 256
#     # try:
#     #     query_len = len(tokenizer.encode(query))
#     #     answer_a_len = len(tokenizer.encode(answer_a))
#     #     answer_b_len = len(tokenizer.encode(answer_b))
#     # except:
#     #     texts.append('skip')
#     #     continue
#     # if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
#     #     query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
#     #         tokenizer.encode(query)[:prompt_len])
#     #     query_len = len(tokenizer.encode(query))
#     #     if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
#     #         remain_len = MAX_LENGTH - query_len
#     #         token_answer_a = tokenizer.encode(answer_a)
#     #         token_answer_b = tokenizer.encode(answer_b)
#     #         if len(token_answer_a) > len(token_answer_b):
#     #             while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
#     #                     token_answer_b):
#     #                 token_answer_a = token_answer_a[:-1]
#     #             while len(token_answer_a) + len(token_answer_b) > remain_len:
#     #                 token_answer_a = token_answer_a[:-1]
#     #                 if len(token_answer_a) + len(token_answer_b) > remain_len:
#     #                     token_answer_b = token_answer_b[:-1]
#     #         else:
#     #             while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
#     #                     token_answer_a):
#     #                 token_answer_b = token_answer_b[:-1]
#     #             while len(token_answer_a) + len(token_answer_b) > remain_len:
#     #                 token_answer_a = token_answer_a[:-1]
#     #                 if len(token_answer_a) + len(token_answer_b) > remain_len:
#     #                     token_answer_b = token_answer_b[:-1]
#     #         answer_a = tokenizer.decode(token_answer_a)
#     #         answer_b = tokenizer.decode(token_answer_b)
#     # prompt = create_rounds(query, answer_b, answer_a)
#     # texts.append(prompt)
#     # texts_token_len.append(len(tokenizer.encode(prompt)))
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

train['label'] = train['label'].apply(lambda x: do(x))
# train2['label'] = train2['label'].apply(lambda x: do(x))



train = train.sample(frac=1., random_state=2023)
# train2 = train2.sample(frac=1., random_state=2023)



# train = train.drop_duplicates("id")
# train2 = train2.drop_duplicates("id")



train_all = pd.concat([train], axis=0) #pd.concat([train, train2], axis=0)
print('train_all',train_all.shape)
train_all['id'] = train_all.index


train_all['text_len'] = train_all['text'].apply(lambda x: len(x.split(' ')))



train_all = train_all[train_all['text_len'] < 1200]

print('skip num: ',train_all[train_all['text']=='skip'].shape)

train_all = train_all[train_all['text']!='skip']

print(train_all['label'].value_counts())

with open(
        f"../data/processed_data/ut_{save_name}_train.parquet",
        'wb') as f:
    pickle.dump(train_all, f)

with open(
        f"../data/processed_data/ut_{save_name}_dev.parquet",
        'wb') as f:
    pickle.dump(train_all.sample(n=100), f)