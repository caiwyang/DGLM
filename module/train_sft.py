from os.path import join

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







def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='openllama_peft')
    parser.add_argument('--data_path', type=str,default='/datasets/data4stf.json')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str,default='/pretrained_ckpt/Llama-stf_mlp_ckpt2/')
    parser.add_argument('--log_path', type=str, default='/pretrained_ckpt/Llama-stf_mlp_ckpt2/log_rest/')
    # model configurations
    parser.add_argument('--language_ckpt_path', type=str,
                        default='/bert-large-cased')  # the path that stores the imagebind checkpoint
    parser.add_argument('--vicuna_ckpt_path', type=str,
                        default='/Meta-Llama-3-8B-Instruct')  # the path that stores the vicuna checkpoint
    parser.add_argument('--delta_ckpt_path', type=str, default='./ckpt/pretrained_ckpt') # the delta parameters trained in stage 1
    parser.add_argument('--max_tgt_len', type=int,default=512) # the maximum sequence length
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

    train_data, train_iter, sampler = load_sft_dataset(args,args['data_path'])

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    logging.basicConfig(level=logging.DEBUG)
    agent = load_model(args)

    torch.distributed.barrier()

    path = args['save_path']
    os.makedirs(path, exist_ok=True)

    # begin to train
    pbar = tqdm(total=length)
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar
            )
            current_step += 1
            if current_step % 10000 == 0:
                save_folder = f"./pretrained_ckpt/Llama-stf_mlp_ckpt2/model_checkpoint_step_{current_step}"
                agent.save_model_sft2(save_folder)
                print(f"Model checkpoint saved at step {current_step} in folder: {save_folder}")

    # 保存最终LoRA模型
    output_dir = path

    agent.save_model_sft(output_dir)

    print('Finished Training')



if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
