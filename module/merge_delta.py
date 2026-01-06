from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def merge_lora_to_base_model():
    model_name_or_path = 'llm_weight/Meta-Llama-3-8B-Instruct'
    adapter_name_or_path = 'llm_code/code/module/ckpt/checkpoints'
    save_path = 'llm_code/code/module/ckpt/Llama-3-8B-train-merge'

    config = AutoConfig.from_pretrained(adapter_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )

    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    return model,tokenizer,

if __name__ == '__main__':
    merge_lora_to_base_model()



