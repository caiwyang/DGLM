#!/bin/bash

deepspeed --include localhost:0,1 --master_addr 127.0.0.1 --master_port 28457 train_sft.py\
    --model openllama_peft \
    --stage 1\
    --data_path  llm_code/module/data/output_ace2004.json\
    --language_ckpt_path bert-large-cased\
    --vicuna_ckpt_path pretrained_ckpt/llama-7b\
    --max_tgt_len 400\
    --save_path  ./ckpt/pandagpt_7b_v1.1_peft/\
    --log_path ./ckpt/pandagpt_7b_v1.1_peft/log_rest/

