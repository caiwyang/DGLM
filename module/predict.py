import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from module.datasets import SupervisedDataset, load_sft_dataset
from module.model.openllama import OpenLLAMAPEFTModel

# init the model
args = {
    'model': 'openllama_peft',
    'data_path': '/datasets/process_datasets/output_test_ace2004.json',
    'language_ckpt_path': '/bert-large-cased',
    'vicuna_ckpt_path': '/vicuna_ckpt/llama-7b',
    'delta_ckpt_path': '/ckpt/pandagpt_7b_v1.1_peft/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 13b model over ...')
tokenizer = AutoTokenizer.from_pretrained(args['language_ckpt_path'])


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><module class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></module></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(
    input,
    sentence,
    max_length,
    top_p,
    temperature,
    modality_cache
):
    if sentence is None :
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        print(f'[!] image path: {sentence}')

    prompt_text = f'</sen>{sentence}\n### Assistant:'
    prompt_text += f' {input}'

    response = model.generate({
        'prompt': prompt_text,
        'sentence': sentence,
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    chatbot=parse_text(input), parse_text(response)
    return chatbot




test_data, test_iter, sampler = load_sft_dataset(**args)

# maximum total number
current_step = 0
for epoch_i in tqdm(range(args['epochs'])):
    for batch in test_iter:
        results = predict(batch['output_texts'], batch['sentence'], args['max_tgt_len'], 10, 0.5, [])
        current_step += 1



for i, result in enumerate(results):
    print(f"Response {i + 1}:", result)
