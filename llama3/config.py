import os
import torch
from transformers import (
    BitsAndBytesConfig 
)
from peft import LoraConfig  

PROJECT_ROOT = os.getenv('LLAMA3_PROJECT_ROOT', os.path.dirname(os.path.abspath(__file__))) 
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def get_lora_config():
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj']
    return LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )