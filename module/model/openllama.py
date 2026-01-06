import json
import os

from header import *
import torch.nn.functional as F
from peft import PeftModel

# from module.module.model.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer, AutoConfig

import torch
from torch import nn
from torch.nn.utils import rnn

from transformers import AutoModel

from merge import merge_lora_to_base_model


class SelectElement(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

# def build_one_instance(tokenizer, conversation):
#     text_list = []
#     turn_num = len(conversation)
#     input_ids, target_ids = [], []
#     for i in range(turn_num):
#         turn = conversation[i]
#         role = turn['from']
#         if i == 0: # the first human turn
#             assert role == 'human'
#             text = '</sen>' + turn['value'] + '\n### Assistant:'
#             one_input_id = tokenizer(text, add_special_tokens=False).input_ids
#             input_ids += one_input_id
#             target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
#         else:
#             if role == 'human':
#                 text = 'Human: ' + turn['value'] + '\n### Assistant:'
#                 one_input_id = tokenizer(text, add_special_tokens=False).input_ids
#                 input_ids += one_input_id
#                 target_ids += [-100]*len(one_input_id)
#             elif role == 'gpt':
#                 # text = turn['value'] + '\n###'
#                 if isinstance(turn['value'], list):
#                     # 将列表的内容组合成字符串
#                     text = ''.join([str(item) for item in turn['value']]) + '\n###'
#                 else:
#                     text = turn['value'] + '\n###'
#                 one_input_id = tokenizer(text, add_special_tokens=False).input_ids
#                 input_ids += one_input_id
#                 target_ids += one_input_id
#             else:
#                 raise Exception('Wrong Role!!!')
#         text_list.append(text)
#         assert len(input_ids) == len(target_ids)
#     return text_list, input_ids, target_ids

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    attention_mask = []

    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']

        if i == 0:  # the first human turn
            system_instruction_pre = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
            system_instruction_sft = """You are a powerful natural language processing model focused on recognizing and classifying named entities (NERs) from a given text.Your goal is to label all named entities in a text and classify them according to their type."""
            text = "<|start_header_id|>system<|end_header_id|>\n\n" + system_instruction_pre + "<|eot_id|>"+ '<|start_header_id|>user<|end_header_id|>\n\n' + "<|reserved_special_token_0|>"+"\n\n"+turn[
                    'value'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids

            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
            attention_mask += [1] * len(one_input_id)
        else:
            if role == 'human':
                text = '<|start_header_id|>user<|end_header_id|>\n\n' + turn[
                    'value'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
                attention_mask += [1] * len(one_input_id)
            elif role == 'gpt':
                text = json.dumps(turn['value'], ensure_ascii=False, indent=4)

                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
                attention_mask += [1] * len(one_input_id)
            else:
                raise Exception('Wrong Role!!!')

        text_list.append(text)

    return input_ids, target_ids, attention_mask


def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        one_input_ids, one_target_ids,_ = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

def merge_lora_to_base_model(adapter_name_or_path,model_name):


    print('Loading model from base model...')



    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model_name, adapter_name_or_path, device_map={'': 'cpu'})
    print(model)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    print(model)

    return model

# PROMPT_START = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        language_ckpt_path = args['language_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        lora_weight = args['save_path']
        max_tgt_len = args['max_tgt_len']  #token最大长度
        stage = args['stage']

        print (f'Initializing Language encoder from {language_ckpt_path} ...')
        # self.language_encoder, self.language_hidden_size = \
        # imagebind_model.imagebind_huge(pretrained=True, store_path=language_ckpt_path)
        self.language_encoder = AutoModel.from_pretrained(language_ckpt_path)
        self.language_hidden_size = self.language_encoder.config.hidden_size
        self.pad_token_id  = 0
        # free Language encoder
        for name, param in self.language_encoder.named_parameters():
            param.requires_grad = False   # 冻结bert参数
        self.language_encoder.eval()
        print ('Language encoder initialized.')

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')

        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        #
        # # self.llama_model = AutoModelForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = AutoModelForCausalLM.from_pretrained(vicuna_ckpt_path, ignore_mismatched_sizes=True)
        # # 冻结大语言模型的所有权重
        # for param in self.llama_model.parameters():
        #     param.requires_grad = False
        #

        # # 冻结大模型参数
        # for name, param in self.llama_model.named_parameters():
        #     param.requires_grad = False

        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = AutoTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)


        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"


        print ('Language decoder initialized.')

        # self.llama_proj = nn.Linear(
        #     self.language_hidden_size, self.llama_model.config.hidden_size
        # )   # 训练（全连接映射层）

        self.llama_proj_mlp = MLP(self.language_hidden_size, self.llama_model.config.hidden_size, dropout=0)

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()


    def encode_sentence(self, sentence_input_ids, sentence_attention_mask):


        with torch.no_grad():
            # 将 embeddings 和 heads 输出都转换为 Half 精度
            embeddings = self.language_encoder(sentence_input_ids, sentence_attention_mask)[0].half()  # 转换为 Half

        # 将 llama_proj 的输出也设置为 Half 精度
        inputs_llama = self.llama_proj_mlp(embeddings) # bsz x 1 x llama_size

        # 确保 atts_llama 也使用 Half 精度
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.half).to(self.device)  # bsz x 1
        return inputs_llama


    def encode_qwen_sentence(self, sentence_input_ids, sentence_attention_mask):


        with torch.no_grad():
            # 将 embeddings 和 heads 输出都转换为 Half 精度
            outputs = self.language_encoder(**sentence_input_ids, output_hidden_states=True)
            qwen_embedding_layer = self.language_encoder.get_input_embeddings()
            prompt_input_ids, prompt_attention_mask = sentence_input_ids.input_ids, sentence_input_ids.attention_mask
            qwen_embedding = qwen_embedding_layer(prompt_input_ids)

            # embeddings = self.language_encoder(sentence_input_ids, sentence_attention_mask)[0].half()  # 转换为 Half

        # 将 llama_proj 的输出也设置为 Half 精度
        inputs_llama = self.llama_proj_mlp(qwen_embedding) # bsz x 1 x llama_size

        # 确保 atts_llama 也使用 Half 精度
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.half).to(self.device)  # bsz x 1
        return inputs_llama


    def _merge_prompt_ids_with_sentence_features(self, sentence_features, prompt_embeds, prompt_ids, attention_mask, labels):
        prompt_ids = prompt_ids.to(self.device)  # bsz x s2
        labels = labels.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2

        sentence_batch, sentence_length, embed_dim = sentence_features.shape
        placeholder = 128002
        batch_size, sequence_length = prompt_ids.shape
        pad_token_id = self.llama_tokenizer.pad_token_id  # 直接获取 pad_token 的 ID
        left_padding = not torch.sum(prompt_ids[:, -1] == pad_token_id)

        # left_padding = not torch.sum(prompt_ids[:, -1] == torch.tensor(self.llama_tokenizer.pad_token))
        # 1. Create a mask to know where special image tokens are
        special_sentence_token_mask = prompt_ids == placeholder  # tensor( [[ True, False, False, False, False, False, False, False, False, False,False, False, False]])
        num_special_sentence_tokens = torch.sum(special_sentence_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_sentence_tokens.max() * (
                    sentence_length - 1)) + sequence_length  # 句子长度+提示长度-1
        batch_indices, non_sentence_indices = torch.where(prompt_ids != placeholder)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_sentence - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_sentence_token_mask * (sentence_length - 1) + 1), -1) - 1
        nb_sentence_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_sentence_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_sentence_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )  # 创建空的embedding
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=prompt_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), -100, dtype=prompt_ids.dtype, device=prompt_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = prompt_embeds.device
        batch_indices, non_sentence_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_sentence_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = prompt_embeds[batch_indices, non_sentence_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_sentence_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_sentence_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        sentence_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        sentence_to_overwrite &= sentence_to_overwrite.cumsum(-1) - 1 >= nb_sentence_pad[:, None].to(target_device)

        if sentence_to_overwrite.sum() != sentence_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of sentence tokens is {torch.sum(special_sentence_token_mask)} while"
                f" the number of sentence given to the model is {sentence_batch}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[sentence_to_overwrite] = sentence_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= sentence_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


    def forward(self, inputs): # image的path和文本的对话conversation

        # bert的使用
        sentences = inputs['sentence']
        max_length = max(len(sentence) for sentence in sentences)
        input_ids = [sentence + [self.pad_token_id] * (max_length - len(sentence)) for sentence in
                     sentences]

        input_ids = torch.tensor(input_ids).cuda()
        attention_mask = (input_ids != self.pad_token_id).long().cuda()

        sentence_embeds= self.encode_sentence(input_ids, attention_mask) # 编码image

        # # qwen使用
        # sentences = inputs['sentence']
        # # 统一 `input_ids` 和 `attention_mask` 维度（需要进行填充）
        # input_ids_list = [item['input_ids'].squeeze(0) for item in sentences]  # 去掉 batch 维度
        # attention_mask_list = [item['attention_mask'].squeeze(0) for item in sentences]  # 同理
        #
        # # 计算最大长度，进行填充（padding）
        # max_length = max([ids.shape[0] for ids in input_ids_list])
        # # 填充 input_ids 和 attention_mask
        # padded_input_ids = rnn.pad_sequence(input_ids_list, batch_first=True,
        #                                                    padding_value=0).cuda()
        # padded_attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        # # 词向量计算
        # embedding_layer = self.language_encoder.get_input_embeddings().cuda()
        # inputs_llama = embedding_layer(padded_input_ids).cuda() # 形状: (batch_size, max_seq_len, embedding_dim)
        # sentence_embeds = self.llama_proj_mlp(inputs_llama)


        output_texts = inputs['output_texts']
        llama_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, self.max_tgt_len)

        # # 冻结
        # #
        # with torch.no_grad():
        #     inputs_embeds = self.llama_model.model.embed_tokens(llama_ids.to(self.device))
        inputs_embeds = self.llama_model.model.model.embed_tokens(llama_ids.to(self.device))
        # inputs_embeds, targets, attention_mask = self.prompt_wrap(sentence_embeds, llama_ids, target_ids, attention_mask)
        inputs_embeds, attention_mask, targets, position_ids = self._merge_prompt_ids_with_sentence_features(
            sentence_embeds, inputs_embeds, llama_ids, attention_mask, target_ids)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        if valid_mask.sum().item() > 0:
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        else:
            gen_acc = 0  # 或者其他合适的默认值

        return loss, gen_acc

    def extract_feature(self, inputs):
        sentences = inputs['sentence']     #  从输入中提取所有句子
        max_length = max(len(sentence) for sentence in sentences)
        input_ids = [sentence + [self.pad_token_id] * (max_length - len(sentence)) for sentence in
                     sentences]
        input_ids = torch.tensor(input_ids).cuda()
        attention_mask = (input_ids != self.pad_token_id).long().cuda()

        sentence_embeds = self.encode_sentence(input_ids, attention_mask)

        return sentence_embeds


    def prepare_generation_embedding(self, prompt,sentence,model):
        batch_size = 1

        # 使用 LLaMA3 系统部分模板
        system_instruction = """You are a powerful natural language processing model focused on recognizing and classifying named entities (NERs) from a given text. Your goal is to label all named entities in a text and classify them according to their type."""
        system_prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_instruction + "<|eot_id|>")
        system_tokens = self.llama_tokenizer(system_prompt,
                                             return_tensors="pt",
                                             add_special_tokens=False).to(self.device)
        system_embeds = model.base_model.model.model.embed_tokens(system_tokens.input_ids).expand(batch_size, -1,
                                                                                                  -1)  # bsz x s1 x embed_dim
        # 用户输入的部分
        header = "<|start_header_id|>user<|end_header_id|>\n\n"
        header_tokens = self.llama_tokenizer(header,
                                             return_tensors="pt",
                                             add_special_tokens=False).to(self.device)
        header_embeds = model.base_model.model.model.embed_tokens(header_tokens.input_ids)

        prompt_with_eot = f"{prompt}<|eot_id|>"
        prompt_tokens = self.llama_tokenizer(prompt_with_eot,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device)
        prompt_embeds = model.base_model.model.model.embed_tokens(prompt_tokens.input_ids)

        content_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.llama_tokenizer(content_prompt,
                                           return_tensors="pt",
                                           add_special_tokens=False).to(self.device)
        user_embeds = model.base_model.model.model.embed_tokens(user_tokens.input_ids).expand(batch_size, -1,
                                                                                              -1)  # bsz x s2 x embed_dim
        sentence = sentence.expand(batch_size, -1,-1)

        # 拼接 embeddings：BOS -> 系统 -> 特征 -> 用户
        inputs_embeds = torch.cat([system_embeds, header_embeds, sentence, prompt_embeds, user_embeds],
                                  dim=1)  # bsz x (1+s1+s2+s3) x embed_dim     sys+user+prompt+assistant
        # Attention mask
        atts_llama = torch.ones(inputs_embeds.shape[:2], dtype=torch.long).to(self.device)
        return inputs_embeds, atts_llama

    def generate(self, inputs):
        '''
            inputs = {
                'image_paths': optional,
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature,
                'modality_embeds': None or torch.tensor,
                'modality_cache': save the image cache,
            }
        '''
        # 准备嵌入
        save_path = './code/pretrained_ckpt/third_ckpt/model_checkpoint_step_60000'
        model = merge_lora_to_base_model(
            save_path, self.llama_model
        )


        prompt = inputs['prompt']
        sentence = self.extract_feature(inputs)


        for batch_conversation,batch_sentence in zip(prompt, sentence):
            for instance in batch_conversation:
                role = instance['from']
                if role == 'human':
                    input_embeds, atts_llama = self.prepare_generation_embedding(instance['value'],batch_sentence,model)
                    # 定义生成停止条件
                    stopping_criteria = StoppingCriteriaList(
                        [StoppingCriteriaSub(stops=[self.llama_tokenizer.eos_token_id], encounters=1)]
                    )

                    # 模型生成
                    outputs = model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=atts_llama,
                        max_new_tokens=inputs['max_tgt_len'],
                        top_p=inputs['top_p'],
                        temperature=inputs['temperature'],
                        do_sample=True,
                        use_cache=True,
                        stopping_criteria=stopping_criteria,
                    )

                    # 解码输出，提取 `assistant` 内容
                    output_text = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=False)
                    if "<|start_header_id|>assistant<|end_header_id|>" in output_text:
                        output_text = output_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[-1]
                    output_text = output_text.split("<|eot_id|>", 1)[0]

        return output_text

