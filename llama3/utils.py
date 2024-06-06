import os
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
)
from typing import List, Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
from llama3.config import get_bnb_config

class LlamaIndexCustomLLM(CustomLLM):  # PEP 8 naming convention
    """Custom LLM wrapper for llama_index, designed for models like LLaMA."""
    llm: LlamaForCausalLM = None
    tokenizer: PreTrainedTokenizerFast = None
    num_output: int = 100
    model_name: str = "Custom LLM" 
    context_window: int = 4096

    def __init__(self, llm, tokenizer, num_output=256, context_window=4096, model_name="Custom LLM"):
        """Initialize with LLM, tokenizer, and optional parameters."""
        super().__init__()  # Call the base class constructor
        self.llm = llm 
        self.tokenizer = tokenizer
        self.num_output = num_output
        self.context_window = context_window
        self.model_name = model_name  # Store for metadata


    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata (dynamically from the underlying model)."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )


    @llm_completion_callback()
    def complete(self, prompt: str, formatted:bool = False, **kwargs: Any) -> CompletionResponse:
        """Generate a completion for the given prompt."""
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            gen_tokens = self.llm.generate(**inputs, max_new_tokens=self.num_output, pad_token_id=self.tokenizer.eos_token_id)
        input_ids = inputs["input_ids"][0]  # Get input token IDs 
        output_ids = gen_tokens[0][len(input_ids):]
        gen_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return CompletionResponse(text=gen_text)


    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = self.llm.generate(**inputs, max_new_tokens=1)
        while True :
            text=self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            yield CompletionResponse(text=text.split()[-1], delta=text.split()[-1])
            inputs = self.tokenizer(text, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs, max_new_tokens=1)
            if(outputs[0][-1] == self.tokenizer.eos_token_id):
                break



def get_llama3_tokenizer(
    adapter_path=None, 
    hf_key=None, 
    model_id="meta-llama/Meta-Llama-3-8B-Instruct", 
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id if not adapter_path else adapter_path,  # Load tokenizer from model or adapter directory
            token=hf_key
        )
    except :
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_key)
    tokenizer.pad_token = tokenizer.eos_token  
    return tokenizer


def get_llama3_model(
    adapter_path=None, 
    hf_key=None, 
    model_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    bnb_config=None,
    device_map="auto"
):
    if hf_key is None:
        hf_key = os.getenv("HF_KEY")  
    # Default BitsAndBytes configuration
    if bnb_config is None:
        bnb_config = get_bnb_config()
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
    # Load the model with quantization and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        token=hf_key  # Pass the token directly here
    )
    # Load and merge adapter weights (if provided)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload() 
    return model

def get_llama3(
    adapter_path=None, 
    hf_key=None, 
    model_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    bnb_config=None,
    device_map="auto"
):
    return get_llama3_model(adapter_path, hf_key, model_id, bnb_config, device_map), get_llama3_tokenizer(adapter_path, hf_key, model_id)


def get_openai_response(usr_msg="",sys_msg="", model="gpt-4"):
    key = os.getenv("OPENAI_KEY")  
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": sys_msg
        },
        {
            "role": "user",
            "content": usr_msg
        }
       
    ],
    temperature=1,
    max_tokens=1088,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )
    output = response.choices[0].message.content.strip()
    return output