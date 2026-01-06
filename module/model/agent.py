import os

from header import *

from collections import OrderedDict
import torch
import deepspeed
import json
import types
import datetime
import logging
import os

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        if args['stage'] == 2:
            self.load_stage_1_parameters(args["delta_ckpt_path"])
            print(f'[!] load stage 1 checkpoint from {args["delta_ckpt_path"]}')

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model, 
            model_parameters=self.model.parameters(),
            config_params=ds_params, 
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc
    
    # def save_model(self, path, current_step):
    #     # only save trainable model parameters
    #     param_grad_dic = {
    #         k: v.requires_grad for (k, v) in self.ds_engine.module.named_parameters()
    #     }
    #     state_dict = self.ds_engine.module.state_dict()
    #     checkpoint = OrderedDict()
    #     for k, v in self.ds_engine.module.named_parameters():
    #         if v.requires_grad:
    #             checkpoint[k] = v
    #     torch.save(checkpoint, f'{path}/pytorch_model.pt')
    #
    #
    #     # save tokenizer
    #     self.model.llama_tokenizer.save_pretrained(path)
    #     # save configuration
    #     self.model.llama_model.config.save_pretrained(path)
    #     print(f'[!] save model into {path}')


    # def load_stage_1_parameters(self, path):
    #     delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
    #     self.model.load_state_dict(delta_ckpt, strict=False)

    def save_model(self, path):
        # 确保目录存在
        os.makedirs(path, exist_ok=True)

        # 规范化路径，移除多余的斜杠
        save_path = os.path.normpath(os.path.join(path, 'pytorch_model.pt'))

        # 保存模型
        state_dict = self.ds_engine.module.state_dict()
        trainable_params = OrderedDict()
        for k, v in state_dict.items():
            if self.ds_engine.module.get_parameter(k).requires_grad:
                trainable_params[k] = v

        try:
            torch.save(trainable_params, save_path)
            print(f'[!] Successfully saved model to {save_path}')
        except Exception as e:
            print(f'[!] Error saving model: {str(e)}')

        # 保存分词器和配置
        try:
            self.model.llama_tokenizer.save_pretrained(path)
            self.model.llama_model.config.save_pretrained(path)
        except Exception as e:
            print(f'[!] Error saving tokenizer/config: {str(e)}')

    def save_model_step(self, path, current_step):
        if deepspeed.dist.get_rank() != 0:
            return  # 如果不是主进程，跳过保存模型

        # 确保目录存在
        os.makedirs(path, exist_ok=True)

        # 获取模型的state_dict
        state_dict = self.ds_engine.module.state_dict()

        # 只保存可训练的参数
        trainable_params = OrderedDict()
        for k, v in state_dict.items():
            if self.ds_engine.module.get_parameter(k).requires_grad:
                trainable_params[k] = v

        # 创建保存文件的完整路径（包括文件名）
        save_path = os.path.join(path, f"pytorch_model_step_{current_step}.pt")

        # 使用torch.save保存可训练参数
        try:
            torch.save(trainable_params, save_path)
            print(f'[!] Successfully saved model to {save_path}')
        except Exception as e:
            print(f'[!] Error saving model: {str(e)}')

    def load_stage_1_parameters(self, path):
        # 规范化路径
        load_path = os.path.normpath(path)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f'Checkpoint path does not exist: {load_path}')

        try:
            delta_ckpt = torch.load(load_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(delta_ckpt, strict=False)
            print(f'[!] Successfully loaded checkpoint from {load_path}')
        except Exception as e:
            print(f'[!] Error loading checkpoint: {str(e)}')
            raise

    def maybe_zero_3(self,param, ignore_status=False, name=None):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param

    # Borrowed from peft.utils.get_peft_model_state_dict
    def get_peft_state_maybe_zero_3(self,named_params, bias):
        if bias == "none":
            to_return = {k: t for k, t in named_params if "lora_" in k}
        elif bias == "all":
            to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            maybe_lora_bias = {}
            lora_bias_names = set()
            for k, t in named_params:
                if "lora_" in k:
                    to_return[k] = t
                    bias_name = k.split("lora_")[0] + "bias"
                    lora_bias_names.add(bias_name)
                elif "bias" in k:
                    maybe_lora_bias[k] = t
            for k, t in maybe_lora_bias:
                if bias_name in lora_bias_names:
                    to_return[bias_name] = t
        else:
            raise NotImplementedError
        to_return = {k: self.maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
        return to_return

    def get_mm_adapter_state_maybe_zero_3(self,named_params, keys_to_match):
        to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
        to_return = {k: self.maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
        return to_return

    def save_model_sft(self, path):
        os.makedirs(path, exist_ok=True)
        try:
            original_model = self.ds_engine.module

            # 1. 保存 LoRA 配置
            if hasattr(original_model, "peft_config"):
                original_model.peft_config.save_pretrained(path)
            else:
                # 手动创建并保存配置
                lora_config = LoraConfig(
                    r=8,  # 使用实际的训练参数
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                lora_config.save_pretrained(path)

            # 2. 保存模型权重
            weight_to_save = self.get_mm_adapter_state_maybe_zero_3(
                original_model.named_parameters(),
                ['llama_proj']
            )
            original_model.llama_model.config.save_pretrained(path)

            # 3. 保存各种权重文件
            torch.save(weight_to_save, os.path.join(path, 'non_lora_trainables.bin'))

            adapter_model_dict = self.get_peft_state_maybe_zero_3(
                original_model.named_parameters(),
                "none"
            )
            # 保存为 .bin 文件
            torch.save(adapter_model_dict, os.path.join(path, 'adapter_model.bin'))

            # 保存为 .safetensors 文件
            try:
                from safetensors.torch import save_file
                save_file(adapter_model_dict, os.path.join(path, 'adapter_model.safetensors'))
            except ImportError:
                print("safetensors not installed. Skipping safetensors save.")

            adapter_model_dict_all = self.get_mm_adapter_state_maybe_zero_3(
                original_model.named_parameters(),
                "none"
            )
            torch.save(adapter_model_dict_all, os.path.join(path, 'adapter_model_all.bin'))

            print(f'[!] Successfully saved model and config to {path}')

        except Exception as e:
            print(f'[!] Error saving model/config: {str(e)}')
            print(f'[!] Error details: {e.__class__.__name__}')
            import traceback
            traceback.print_exc()

    def save_model_sft2(self, path):

        if deepspeed.dist.get_rank() != 0:
            return  # 如果不是主进程，跳过保存模型
        # 确保目录存在
        os.makedirs(path, exist_ok=True)

        try:
            original_model = self.ds_engine.module

            # 1. 保存 LoRA 配置
            if hasattr(original_model, "peft_config"):
                original_model.peft_config.save_pretrained(path)
            else:
                # 手动创建并保存配置
                lora_config = LoraConfig(
                    r=8,  # 使用实际的训练参数
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                lora_config.save_pretrained(path)

            # 2. 保存模型权重
            weight_to_save = self.get_mm_adapter_state_maybe_zero_3(
                original_model.named_parameters(),
                ['llama_proj']
            )
            original_model.llama_model.config.save_pretrained(path)

            # 3. 保存各种权重文件
            torch.save(weight_to_save, os.path.join(path, 'non_lora_trainables.bin'))

            adapter_model_dict = self.get_peft_state_maybe_zero_3(
                original_model.named_parameters(),
                "none"
            )
            # 保存为 .bin 文件
            torch.save(adapter_model_dict, os.path.join(path, 'adapter_model.bin'))

            # 保存为 .safetensors 文件
            try:
                from safetensors.torch import save_file
                save_file(adapter_model_dict, os.path.join(path, 'adapter_model.safetensors'))
            except ImportError:
                print("safetensors not installed. Skipping safetensors save.")

            adapter_model_dict_all = self.get_mm_adapter_state_maybe_zero_3(
                original_model.named_parameters(),
                "none"
            )
            torch.save(adapter_model_dict_all, os.path.join(path, 'adapter_model_all.bin'))

            print(f'[!] Successfully saved model and config to {path}')

        except Exception as e:
            print(f'[!] Error saving model/config: {str(e)}')
            print(f'[!] Error details: {e.__class__.__name__}')
            import traceback
            traceback.print_exc()