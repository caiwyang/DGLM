import os
import argparse
import re

import torch
import random
import time
import logging
import numpy as np
from torch import nn
 # 直接导入你的模型类
from datasets import *  # 需要自己定义的数据集加载模块
from config import *  # 需要自己定义的配置模块
from module.model.openllama_evaluate import OpenLLAMAPEFTModel

# 设置环境变量
os.environ["PATH"] += os.pathsep + "./anaconda3/envs/llmds/bin"

# 解析命令行参数
def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='openllama_peft')
    parser.add_argument('--data_path', type=str,
                        default='./module/datasets/test_CrossNER_AI.json')
    parser.add_argument('--test_data_path', type=str,
                        default='./module/datasets/test_CrossNER_AI.json')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='./ckpt/pandagpt_7b_v1.1_peft/')
    parser.add_argument('--log_path', type=str, default='./ckpt/pandagpt_7b_v1.1_peft/log_rest/')
    # model configurations
    parser.add_argument('--language_ckpt_path', type=str,
                        default='./basetest/cache/bert-large-cased')
    parser.add_argument('--vicuna_ckpt_path', type=str,
                        default='./Meta-Llama-3-8B-Instruct')
    parser.add_argument('--delta_ckpt_path', type=str,
                        default='./ckpt/pretrained_ckpt')
    parser.add_argument('--max_tgt_len', type=int, default=200)
    parser.add_argument('--stage', type=int, default=1)
    return parser.parse_args()

# 设置随机种子
def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 配置环境
def config_env(args):
    args['root_dir'] = './module/'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    set_random_seed(args['seed'])

# 创建目录
def build_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# 加载LoRA权重和映射器权重
def load_lora_and_proj_weights(model, lora_ckpt_path, proj_ckpt_path):
    # 加载LoRA权重
    if os.path.exists(lora_ckpt_path):
        print(f"Loading LoRA weights from {lora_ckpt_path}")
        lora_weights = torch.load(lora_ckpt_path)
        model.load_state_dict(lora_weights, strict=False)
    else:
        print(f"No LoRA weights found at {lora_ckpt_path}")

    # 加载映射器权重
    if os.path.exists(proj_ckpt_path):
        print(f"Loading projection weights from {proj_ckpt_path}")
        proj_weights = torch.load(proj_ckpt_path)
        model.llama_proj.load_state_dict(proj_weights)
    else:
        print(f"No projection weights found at {proj_ckpt_path}")

# 主函数
# def main(**args):
#     # 配置环境
#     config_env(args)
#
#     # 设置日志
#     build_directory(args['save_path'])
#     build_directory(args['log_path'])
#     if args['log_path']:
#         logging.basicConfig(
#             format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#             level=logging.DEBUG,
#             filename=f'{args["log_path"]}/train_{time.asctime()}.log',
#             filemode='w'
#         )
#
#     # 加载测试数据
#     test_data = SupervisedDataset(args['data_path'], args['language_ckpt_path'])
#     # 初始化模型
#     model = OpenLLAMAPEFTModel(**args).cuda()
#
#     saved_state = torch.load(
#         "./pretrained_ckpt/mlp1_ckpt/non_lora_trainables.bin", map_location='cpu')
#     # 确保在DeepSpeed包装之前加载权重
#     # # 加载权重到 llama_proj
#     # model.llama_proj.weight.data.copy_(saved_state['llama_proj.weight'])
#     # model.llama_proj.bias.data.copy_(saved_state['llama_proj.bias'])
#     # 加载权重到 llama_proj_mlp
#     model.llama_proj_mlp.linear.weight.data.copy_(saved_state['llama_proj_mlp.linear.weight'])
#     model.llama_proj_mlp.linear.bias.data.copy_(saved_state['llama_proj_mlp.linear.bias'])
#     lora_weights = torch.load('./pretrained_ckpt/mlp1_ckpt/adapter_model.bin')
#     model.load_state_dict(lora_weights, strict=False)
#
#     print(f'[!] Initialized OpenLLAMAPEFTModel.')
#
#     # 模型准备
#     model.eval()  # 设置为评估模式
#
#     # 推理过程
#     for test_batch in test_data:
#         sentence = test_batch['sentence']
#         prompt_text = test_batch['output_texts']
#         response = model.generate({
#             'prompt': prompt_text,
#             'sentence': sentence,
#             'top_p': 9,
#             'temperature': 0.2,
#             'max_tgt_len': args['max_tgt_len'],
#             'modality_embeds': []
#         },model)
#         print(response)

import json
import re
import logging


def clean_output_text(output_text):
    """
    清洗模型生成的输出文本，确保其为有效的 JSON 格式。
    仅提取第一个有效的 JSON 对象，不修复未转义的双引号。
    """
    try:
        logging.info("原始输出文本: %s", output_text)

        # 1. 提取第一个有效的 JSON 对象
        first_brace = output_text.find('{')
        if first_brace == -1:
            logging.error("无法找到 '{'，无法提取 JSON")
            return {}

        brace_count = 0
        end_index = -1
        for i in range(first_brace, len(output_text)):
            if output_text[i] == '{':
                brace_count += 1
            elif output_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_index = i
                    break

        if end_index == -1:
            logging.error("无法找到匹配的 '}'，无法提取完整的 JSON")
            return {}

        json_str = output_text[first_brace:end_index + 1]
        logging.info("提取的 JSON 字符串: %s", json_str)

        # 2. 尝试解析提取的 JSON 字符串
        pred_json = json.loads(json_str)
        logging.info("成功解析 JSON")
        return pred_json

    except json.JSONDecodeError as e:
        logging.error("JSON 解码错误: %s", e)
        return {}
    except Exception as ex:
        logging.error("处理输出文本时发生异常: %s", ex)
        return {}


# def format_labels(label_dict):
#     """
#     将标签字典转换为"类别-标签"的列表，并移除标签中的空格。
#     """
#     formatted = []
#     for category, labels in label_dict.items():
#         for label in labels:
#             # 移除类别和标签中的空格
#             normalized_category = category.replace(' ', '')
#             normalized_label = label.replace(' ', '')
#             # 组合类别和标签
#             formatted_label = f"{normalized_category}-{normalized_label}"
#             formatted.append(formatted_label)
#     return formatted
def format_labels(label_dict):
    """
    将标签字典转换为 '关系-实体1-实体2' 格式。
    每个关系类型只允许最多一个实体对（列表或嵌套列表），否则返回空。
    """
    formatted = []

    for category, value in label_dict.items():
        category_str = category.replace(' ', '')

        # 忽略空列表
        if not value:
            continue

        # case 1: 嵌套对，期望是 [['a', 'b']]
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], list)
            and len(value[0]) == 2
            and all(isinstance(v, str) for v in value[0])
        ):
            h, t = value[0]
            formatted.append(f"{category_str}-{h.replace(' ', '')}-{t.replace(' ', '')}")

        # case 2: 扁平对，期望是 ['a', 'b']
        elif (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(v, str) for v in value)
        ):
            formatted.append(f"{category_str}-{value[0].replace(' ', '')}-{value[1].replace(' ', '')}")

        # case 3: 超过一个实体对或格式异常，整体无效
        else:
            return []

    return formatted





def main(**args):
    # 配置环境
    config_env(args)

    # 设置日志
    build_directory(args['save_path'])
    build_directory(args['log_path'])
    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.strftime("%Y%m%d_%H%M%S")}.log',
            filemode='w'
        )

    # 加载测试数据
    test_data = SupervisedDataset(args['data_path'], args['language_ckpt_path'])

    # 初始化模型
    model = OpenLLAMAPEFTModel(**args).cuda()

    # 加载权重
    saved_state = torch.load(
        "./pretrained_ckpt/Llama-stf_mlp_ckpt/non_lora_trainables.bin", map_location='cpu')

    # 加载特定模块的权重
    model.llama_proj_mlp.linear.weight.data.copy_(saved_state['llama_proj_mlp.linear.weight'])
    model.llama_proj_mlp.linear.bias.data.copy_(saved_state['llama_proj_mlp.linear.bias'])

    # 加载 LoRA 权重
    lora_weights = torch.load('./pretrained_ckpt/Llama-stf_mlp_ckpt/adapter_model.bin')
    model.load_state_dict(lora_weights, strict=False)

    print(f'[!] Initialized OpenLLAMAPEFTModel.')

    # 设置为评估模式
    model.eval()

    # 初始化结果列表
    results = []

    # 推理过程
    for test_batch in tqdm(test_data, desc="Processing batches", unit="batch"):
        sentence = test_batch['sentence']
        prompt_text = test_batch['output_texts']

        # 准备生成输入
        generate_input = {
            'prompt': prompt_text,
            'sentence': sentence,
            'top_p': 0.95,  # 调整为合理的 top_p 值，例如 0.95
            'temperature': 0.2,
            'max_tgt_len': args['max_tgt_len'],
            'modality_embeds': []
        }

        # 调用 generate 函数
        response = model.generate(generate_input, model)

        # 清洗和解析输出文本
        pred_labels_dict = clean_output_text(response)

        # 格式化预测标签
        formatted_pre = format_labels(pred_labels_dict)

        # 获取真实标签
        # 假设 test_batch 中包含 'gpt_labels' 字段
        # 提取 gpt 标签
        true_labels_dict = {}
        for entry in test_batch['output_texts']:
            if entry['from'] == 'gpt':
                true_labels_dict = entry['value']
                break

        # 格式化为统一输出格式
        formatted_true = format_labels(true_labels_dict)

        # 创建合并后的条目
        merged_entry = {
            "sentence": sentence,
            "true": formatted_true,
            "pre": formatted_pre
        }

        # 添加到结果列表
        results.append(merged_entry)

        # 打印当前轮次的结果
        print(json.dumps(merged_entry, ensure_ascii=False, indent=4))

    # 将结果保存到新的 JSON 文件中
    output_file = './merged_test_ai.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"合并完成，结果已保存到 '{output_file}'")


if __name__ == "__main__":
    # 解析命令行参数并运行主函数
    args = parser_args()
    args = vars(args)
    main(**args)
