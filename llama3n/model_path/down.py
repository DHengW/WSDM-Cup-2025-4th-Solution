#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/Llama-3.1-Nemotron-70B-Instruct-HF',local_dir='/root/autodl-tmp/src/4th_solution/llama3n/model_path/Llama3N')