#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



import copy
import os
import json

import numpy as np
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, bert_path: str):
        super(SupervisedDataset, self).__init__()

        with open(data_path, 'r') as f:   #  读取文件   图像名字和conversion（human和机器）
            json_data = json.load(f)
            # for debug:
            #json_data = json_data[:100000]
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        # tokenizer =AutoTokenizer.from_pretrained(bert_path, use_fast=False)


        self.sentence_list, self.caption_list = [], []
        for item in json_data:  #
            one_image_name, one_caption = item["sentence"], item["conversation"] #  获取图像名字和caption信息（caption是conversation）
            # TODO: stage 2 dataset format is invalid  # caption是列表。存储的是conversion
            if not isinstance(one_image_name, str):
                one_image_name = str(one_image_name)
            encoded_input = tokenizer.encode(one_image_name, add_special_tokens=True)
            # bert_input = tokenizer.convert_tokens_to_ids(one_image_name)
            # qwen_inputs = tokenizer(one_image_name, return_tensors="pt")
            # encoded_input = qwen_inputs
            max_length = 512
            if len(encoded_input) <= max_length:
                bert_input = encoded_input
                caption= one_caption
                self.sentence_list.append(bert_input)
                self.caption_list.append(caption)
            else:
                print(f"Sentence '{one_image_name}' is too long and will be discarded.")

        print(f'[!] collect {len(self.sentence_list)} samples for training')

    def __len__(self): # number of instances
        return len(self.sentence_list)

    #def __getitem__(self, i) -> Dict[str, torch.Tensor]: # how to get item, 取一个样本
    def __getitem__(self, i):
        return dict(sentence=self.sentence_list[i], output_texts=self.caption_list[i])

    def collate(self, instances):
        sentence, output_texts = tuple([instance[key] for instance in instances] for key in ("sentence", "output_texts"))
        return dict(
            sentence=sentence,
            output_texts=output_texts
        )


# if __name__ == "__main__":
#     path_sentence = "/data/output_ace2004.json"
#     path_image = "/data/output_ace2004.json"
#     ss= SupervisedDataset(path_sentence)
#     sampler = torch.utils.data.RandomSampler(ss)
