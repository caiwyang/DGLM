from os.path import join

from deepspeed import comm

from header import *
from datasets import *
from model import *
from config import *
import logging
import os
from torch.utils.cpp_extension import load
# 检查环境变量并设置默认值
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "28457")

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='openllama_peft')
    parser.add_argument('--data_path', type=str,default='./module/datasets/pretraining_datasets/train4pretrain.json')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str,default='./ckpt/Llama-qwen_mlp/')
    parser.add_argument('--log_path', type=str, default='./ckpt/Llama-qwen_mlp/log_rest/')
    # model configurations
    parser.add_argument('--language_ckpt_path', type=str, default='./llm_weight/qwen2-7b')
    parser.add_argument('--language_ckpt', type=str, default='./llm_weight/glove/glove/glove.6B.300d.txt')# the path that stores the imagebind checkpoint
    parser.add_argument('--vicuna_ckpt_path', type=str, default='./Meta-Llama-3-8B-Instruct') # the path that stores the vicuna checkpoint
    parser.add_argument('--delta_ckpt_path', type=str, default='./ckpt/pretrained_ckpt') # the delta parameters trained in stage 1
    parser.add_argument('--max_tgt_len', type=int,default=400) # the maximum sequence length
    parser.add_argument('--stage', type=int,default=1) # the maximum sequence length
    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '29500')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = './module/'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def save_model_step(model, path, current_step):
    if dist.get_rank() != 0:
        return  # 如果不是主进程，跳过保存模型

    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    # 保存模型的 state_dict
    try:
        torch.save(
            model.state_dict(),
            os.path.join(path, f"pytorch_model_step_{current_step}.pt")
        )
        print(f'[!] Successfully saved pytorch_model.pt to {path} at step {current_step}')
    except Exception as e:
        print(f'[!] Error saving pytorch_model.pt: {str(e)}')


def save_model(agent, checkpoint_path):
    """
    保存模型权重
    """
    torch.save(agent.get_model_state(), checkpoint_path)

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'./module/dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])  # pandagpt保存路径
    build_directory(args['log_path'])  # log日志保存路径

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    
    train_data, train_iter, sampler = load_sft_dataset(args,args['data_path'])  #载入数据集  训练数据，训练迭代和采样

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    logging.basicConfig(level=logging.DEBUG)
    agent = load_model(args)
    torch.distributed.barrier()

    path = args['save_path']
    os.makedirs(path, exist_ok=True)

    # begin to train
    pbar = tqdm(total=length)    # maximum total number
    current_step = 0
    checkpoint_interval = 10000
    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            current_step += 1

            # 每达到checkpoint_interval步数时保存模型
            if current_step % checkpoint_interval == 0:
                agent.save_model_step(path, current_step)

    torch.distributed.barrier()
    agent.save_model(path)
    logging.info(f"Final checkpoint saved at step {current_step}")

    # checkpoint = torch.load(f'{path}pytorch_model.pt')
    # agent.load_state_dict(checkpoint['model_state_dict'])
    # final_save_path = join(path)
    # agent.save_model(final_save_path,epoch_i)


    print('Finished Training')



if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
