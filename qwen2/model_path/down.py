#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-72B-Instruct',local_dir='/root/autodl-tmp/src/4th_solution/qwen2/model_path/qwen2')