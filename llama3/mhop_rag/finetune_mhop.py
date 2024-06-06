import dotenv
from datasets import load_from_disk, concatenate_datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel     # type: ignore
from datasets import load_from_disk, concatenate_datasets
import numpy as np
from transformers.training_args import TrainingArguments
from llama3.utils import get_llama3
from trl import SFTTrainer

dotenv.load_dotenv()
ds = load_from_disk("./data/mhop_qns")
def extract_prompt_completion(example):
    # Refined pattern for more robust matching
    sl = example['response'].lower().split('react step by step answer') 
    # if(len(sl) >= 2) :
    prompt = sl[0].split("question")[1].strip(" :\n")
    return {"prompt": prompt, "completion": sl[1].strip(" :\n")}
    # return 

# Apply the function
dataset = ds.map(extract_prompt_completion)
dataset = dataset.remove_columns(["user_message", "response"]) 


# n_epochs = 1
# n_split = 1
# cache = "/data1/sahilkamble/new_cache_dir"
model = "Meta-Llama-3-8B-Instruct"
save_dir = f"./models/{model}_ft_mhop"
target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj']

adapter_path = './models/Meta-Llama-3-8B-Instruct_ft_1_1split'
model, tokenizer = get_llama3(adapter_path=adapter_path)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

tokenizer.pad_token = tokenizer.eos_token
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset, # type: ignore
    # eval_dataset=eval_data,
    args=TrainingArguments(
        auto_find_batch_size=True,
        # resume_from_checkpoint="chkpt_path",
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        # gradient_accumulation_steps=grad_acc_steps,
        # eval_accumulation_steps=grad_acc_steps,
        # warmup_steps=10,
        warmup_ratio=0.1,
        # max_steps=20,
        # learning_rate=2e-4,
        # fp16=True,
        bf16=True,
        # logging_steps=20/grad_acc_steps,
        # evaluation_strategy="steps",
        # eval_steps=20/grad_acc_steps,
        output_dir=save_dir+"_outputs",
        # optim="paged_adamw_8bit",
        # num_train_epochs=n_epochs
    ),

    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train(resume_from_checkpoint=True) # type: ignore
trainer.save_model(save_dir)
