import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并到base model中
"""
def merge_lora_to_base_model(adapter_name_or_path,model_name_or_path):

    save_path = '/Llama-3-8B-train-merge'

    config = AutoConfig.from_pretrained(adapter_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True
    )
    print('Loading model from base model...')
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'},
        config=config
    )
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(
            torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(
            torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional model weights...')
    if os.path.exists(os.path.join(adapter_name_or_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(adapter_name_or_path, 'non_lora_trainables.bin'), map_location='cpu')

    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    return model


if __name__ == '__main__':
    merge_lora_to_base_model()



