
## Hardware Specifications
- **CPU Cores**: 120
- **Memory**: 960 GB
- **GPU**: NVIDIA Tesla A100 80G
- **Number of GPUs**: 8
- - **OS/Platform**: Linux
## Third-Party Software
- **Python**: 3.10.8
- **PyTorch**: 2.5.1+cu124
- **CUDA**: 11.8
- **cuDNN**: 8.6.0
# Installation python packages
```bash
pip install -r requirements.txt
```

# Explanation of directory tree
```
.
├── src # The process of training submission model (qwen2.5-14b-instruct)
├── model_path # pretrained model path (qwen2.5-14b-instruct)
│   └── qwen2
│       ├── model-00001-of-00004.safetensors
│       └── model-.........
├── data # train data and other data
│   ├── processed_data
│   │   ├── qwen2fold0
│   │   ├── qwen2fold1
│   │   ├── qwen2fold2
│   │   ├── qwen2fold3
│   │   └── qwen2fold4
│   ├── sample_submission.csv
│   ├── test.parquet
│   ├── train.parquet
│   ├── notie_lmsys.csv
│   └── ut_multi.csv
├── model_save # save path for train model
├── athene # The process of training distillation model 1 (Nexusflow/Athene-V2-Chat)
│   ├── data
│   ├── model_path # pretrained model path (Nexusflow/Athene-V2-Chat)
│   │   └── athene
│   │       ├── model-00001-of-00031.safetensors
│   │       ├── model-.........
│   │       └── vocab.json
│   ├── model_save # save path for train model AND generated distillation data
│   └── src
├── llama3n # The process of training distillation model 2 , same dir architect as athene
│   ├── data
│   ├── model_path
│   │   └── Llama3N # pretrained model path (nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)
│   │       ├── model-00001-of-00030.safetensors
│   │       └── model-.........
│   ├── model_save
│   └── src
├── qwen2 # The process of training distillation model 3 , same dir architect as athene
│   ├── data
│   ├── model_path
│   │   └── qwen2 # pretrained model path (Qwen/Qwen2.5-72B-Instruct)
│   │       ├── model-00001-of-00030.safetensors
│   │       └── model-.........
│   ├── model_save
│   └── src
└── sub # output dir

```

# Data Preparation
The Datasets used for Training have already placed in ***./data***, you can also find them in below links:
1. [Competition datasets](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data) train.parquet/ test.parquet/ submission.csv
2. [Processed datasets](https://www.kaggle.com/datasets/daihengwei/wsdm-ultrafeedback-c4aimultirewardbench-processed) from other source:
    - notie_lmsys.csv —— Postprocessed from last competition [LMSYS](https://www.kaggle.com/c/lmsys-chatbot-arena/data)
    - ut_multi.csv —— Postprocessed from [ultraFeedback](https://www.kaggle.com/datasets/thedrcat/llm-human-preference-data-ultrafeedback) and [C4AI-Community/multilingual-reward-bench](https://huggingface.co/datasets/C4AI-Community/multilingual-reward-bench)

# Model Download Preparation
Download these three models to the model_path folder:<br>
- Nexusflow/Athene-V2-Chat (https://huggingface.co/Nexusflow/Athene-V2-Chat) (rename as athene)<br>
- nvidia/Llama-3.1-Nemotron-70B-Instruct-HF (https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) (rename as llama3n)<br>
- Qwen/Qwen2.5-72B-Instruct (https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) (rename as qwen2)<br>
- qwen2.5-14b-instruct (https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) (rename as qwen2)


# train model script
（
The time estimates are based on a single machine with 8 NVIDIA A100 80GB GPUs, if there are multiple 8-GPU A100 nodes available, you can train a fold on each node simultaneously）
1. **Train & Inference Distillation model 1/2/3** (Takes 560 hours)
    ```bash
    cd ./athene/src
    sh run.sh

    cd ./llama3n/src
    sh run.sh

    cd ./qwen2/src
    sh run.sh
    ```
After the Train & Inference is completed, the distilled data generated is stored in the **model_save** directories corresponding to the three distillation models. To save time for reproducing the results, the distilled data already placed in directories. This way, **you can directly train the final 14B submission model follow instructions in part "Fast train script"** :

2.  **Train final model** (Takes 48 hours)
    ```bash
    cd ./src
    sh run.sh 
    ```

4. **Merge LoRA and Quantize**

    ```python
    python3 merge_weights.py
    ```
    ```bash
    cp ../model_save/qwen2_16bit_load_fintune/fold0/kl_distil4_encode/1best_val_acc_model/adapter_config.json ../model_save/temp-merge
    ```
    ```python
    python3 run_gptq.py
    ```
Here, the LoRA layers of the 5-fold qwen2.5-14b models are merged and then quantized to 8-bit using GPTQ.
5. **Predict Test Set**
    ```python
    python3 wsdm_inference.py
    ```
Once the previous steps are completed, you can directly run this script to make predictions.The final results will be saved in ./sub/submission.csv
If there is a new test set, you can directly replace ./data/test.parquet for prediction.<br>


# Direct prediction
If you want to directly predict using my best model, please first download the model (https://www.kaggle.com/models/daihengwei/wsdm-qwen2.5/transformers/711gptq8b128/1) to the model_save folder and rename the folder to final_model. Then replace the file ./data/test.parquet and run the following script. This script is consistent with the Kaggle online inference script and will use GPU 0 and GPU 1.<br>
    ```python
    python3 wsdm_inference.py
    ```

# Fast train script
Since training three 70b large models is extremely slow, I have provided the samples before distillation (with the probability distribution to be distilled already predicted). This way, you can directly distill and train the final qwen2.5-14b model.

1. Download distillation data generated by three 70B models, link:
    ```
    https://www.kaggle.com/datasets/daihengwei/wsdm-4th-distillation-data-three-models
    ```
2. After downloading the data, place it in the corresponding location within the training code directory:
    ```
    distil_data
    │
    ├── athene
    │   └── model_save
    │       ├── athene_4bit_load_fintune # replace the same name dictionary in full code repo
    │       │   ├── fold0
    │       │   │   └── 0_preds.parquet
    │       │   ├── fold1
    │       │   │   └── 1_preds.parquet
    │       │   ├── fold2
    │       │   │   └── 2_preds.parquet
    │       │   ├── fold3
    │       │   │   └── 3_preds.parquet
    │       │   └── fold4
    │       │       └── 4_preds.parquet
    │       └── athene_4bit_pretrain
    │           └── pretrain_best_val_acc_model
    ├── llama3n
    │   └── model_save
    │       ├── llama3n_4bit_load_fintune # replace the same name dictionary in full code repo
    │       │   ├── fold0
    │       │   │   └── 0_preds.parquet
    │       │   ├── fold1
    │       │   │   └── 1_preds.parquet
    │       │   ├── fold2
    │       │   │   └── 2_preds.parquet
    │       │   ├── fold3
    │       │   │   └── 3_preds.parquet
    │       │   └── fold4
    │       │       └── 4_preds.parquet
    │       └── llama3n_4bit_pretrain
    │           └── pretrain_best_val_acc_model
    └── qwen2
        └── model_save
            ├── qwen2_4bit_load_fintune # replace the same name dictionary in full code repo
            │   ├── fold0
            │   │   └── 0_preds.parquet
            │   ├── fold1
            │   │   └── 1_preds.parquet
            │   ├── fold2
            │   │   └── 2_preds.parquet
            │   ├── fold3
            │   │   └── 3_preds.parquet
            │   └── fold4
            │       └── 4_preds.parquet
            └── qwen2_4bit_pretrain
                └── pretrain_best_val_acc_model
    ```
3. Directly distill and train the final qwen2.5-14b model
    ```bash
    cd ./src
    sh run.sh 
    ```
    ```python
    python3 merge_weights.py
    ```
    ```bash
    cp ../model_save/qwen2_16bit_load_fintune/fold0/kl_distil4_encode/1best_val_acc_model/adapter_config.json ../model_save/temp-merge
    ```
    ```python
    python3 run_gptq.py
    python3 wsdm_inference.py
    ```

