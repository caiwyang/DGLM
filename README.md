# Model Weights and Training Guide

## 1. Overview

This project is based on **LLaMA / Vicuna** and uses **LoRA** for efficient parameter fine-tuning (SFT).  
It supports:
- Continued training from a base model using custom instruction datasets.
- Saving LoRA fine-tuned weights for later use or merging.
- Multi-GPU distributed training with **DeepSpeed** + **NCCL**.

The final weights can be:
- Used directly with LoRA-enabled inference scripts.
- Merged into the base model for standalone deployment.

---

## 2. Requirements

Please ensure the following dependencies are installed:

```bash
python >= 3.9
torch >= 2.0
transformers >= 4.30
deepspeed >= 0.9
tqdm
numpy
```

Install with:
```bash
pip install torch transformers deepspeed tqdm numpy
```

---

## 3. File Structure

| File / Folder | Description |
|---------------|-------------|
| `train.py` | Main training entry point (this script) |
| `model.py` | Model building and loading logic |
| `datasets.py` | Dataset loading and DataLoader construction |
| `config.py` | Training configuration loader |
| `header.py` | Common imports and utility functions |
| `dsconfig/*.json` | DeepSpeed configuration files |
| `pretrained_ckpt/` | Directory for saving fine-tuned weights |

---

## 4. Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (must match DeepSpeed config) | `openllama_peft` |
| `--data_path` | Training dataset path (JSON) | `./code/module/datasets/data4stf.json` |
| `--save_path` | Model checkpoint save directory | `./code/pretrained_ckpt/Llama-stf_mlp_ckpt2/` |
| `--log_path` | Training log save directory | `./code/pretrained_ckpt/Llama-stf_mlp_ckpt2/log_rest/` |
| `--language_ckpt_path` | Language model path (BERT, etc.) | `./bert-large-cased` |
| `--vicuna_ckpt_path` | Base Vicuna/LLaMA model path | `./Meta-Llama-3-8B-Instruct` |
| `--delta_ckpt_path` | Stage-1 delta parameters | `./ckpt/pretrained_ckpt` |
| `--max_tgt_len` | Maximum target sequence length | `512` |
| `--stage` | Training stage (1/2) | `1` |
| `--local_rank` | Distributed training local GPU rank | `0` |

---

## 5. Usage

### Single GPU
```bash
python train.py   --model openllama_peft   --data_path ./data/train.json   --save_path ./output/   --vicuna_ckpt_path ./vicuna-7b
```

### Multi-GPU (DeepSpeed)
```bash
deepspeed --num_gpus=8 train.py   --model openllama_peft   --data_path ./data/train.json   --save_path ./output/   --vicuna_ckpt_path ./vicuna-7b
```

---

## 6. Training Workflow

1. **Set environment variables** (auto-handled in script).
2. **Load configuration** (`config.py` + CLI arguments).
3. **Initialize distributed environment** (DeepSpeed + NCCL).
4. **Load dataset** (`datasets.py`).
5. **Build model** (`model.py`).
6. **Train** (supports multi-stage SFT).
7. **Save checkpoints** every 10k steps.
8. **Save final LoRA weights**.

---

## 7. Output Weights

After training, weights are stored in:
```
save_path/
  ├── adapter_config.json
  ├── adapter_model.bin
  ├── tokenizer.model
  ├── tokenizer_config.json
  └── ...
```

These are **LoRA fine-tuning weights** and require the base model for inference.  
To merge weights into the base model:

```python
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

base_model = LlamaForCausalLM.from_pretrained("./vicuna-7b")
tokenizer = LlamaTokenizer.from_pretrained("./vicuna-7b")
model = PeftModel.from_pretrained(base_model, "./output/")
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

---

## 8. Logs & Resuming

- Logs are saved in `log_path/`
- Resume training by setting `--delta_ckpt_path` to your checkpoint folder
- You can also load `model_checkpoint_step_xxx` to continue training

---

## 9. Notes

- `--vicuna_ckpt_path` must point to a valid base model.
- For multi-GPU runs, ensure NCCL environment variables are properly set.
- Training data must be JSON format compatible with `datasets.py`.
- `save_path` and `log_path` will be created automatically.

---
