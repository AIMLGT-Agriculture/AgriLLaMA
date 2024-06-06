import torch
from transformers import (
    Trainer,
    TrainingArguments
)
import transformers
from peft import (
    prepare_model_for_kbit_training, 
    get_peft_model
    )     # type: ignore
from datasets import load_from_disk, concatenate_datasets
from llama3.utils import get_llama3
from llama3.config import (
    DATA_DIR, 
    MODELS_DIR, 
    get_lora_config
)
import os

n_epochs = 1
n_split = 1
# cache = "/data1/sahilkamble/new_cache_dir"
model = "Meta-Llama-3-8B-Instruct"
data_path = os.path.join(DATA_DIR, "KARNATAKA_processed_dataset_llama_3_8B_instruct")
adapter_path = os.path.join(MODELS_DIR, "Meta-Llama-3-8B-Instruct_ft_1_ep_1split")
save_dir = os.path.join(MODELS_DIR, f"{model}_ft_pop_kar_{n_epochs}_ep_{n_split}split")
eval_spit = "split_1"

# model_id = f"meta-llama/{model}"
# target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj']


# token = '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' 
# # adapters_name = './models/Meta-Llama-3-8B-Instruct_ft_1_1split'
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# cache = "/data1/sahilkamble/new_cache_dir"

# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token = token)

model, tokenizer = get_llama3(adapter_path=adapter_path)

# model = PeftModel.from_pretrained(model, adapters_name)
# model = model.merge_and_unload()
# tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# config = LoraConfig(
#     r=16, 
#     lora_alpha=32,
#     target_modules=target_modules,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
model = get_peft_model(model, get_lora_config())


data = load_from_disk(data_path)
train_data=data
if(n_split == 1):
    train_data = data["split_1"]
elif(n_split == 3):
    train_data = concatenate_datasets([data["split_1"], data["split_2"], data["split_3"]]) # type: ignore
eval_data = data[eval_spit].select(range(1)) # type: ignore

grad_acc_steps = 4

tokenizer.pad_token = tokenizer.eos_token
trainer = Trainer(
    model=model,
    train_dataset=train_data, # type: ignore
    eval_dataset=eval_data,
    args=TrainingArguments(
        # per_device_batch_size=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=grad_acc_steps,
        eval_accumulation_steps=grad_acc_steps,
        warmup_ratio=0.1,
        # warmup_steps=10,
        # max_steps=20,
        # learning_rate=2e-4,
        # fp16=True,
        bf16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=20,
        output_dir=save_dir+"_outputs",
        optim="paged_adamw_8bit",
        num_train_epochs=n_epochs
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(save_dir)
