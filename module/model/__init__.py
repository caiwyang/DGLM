from .agent import DeepSpeedAgent
from .openllama import OpenLLAMAPEFTModel
import os


from torch.utils.cpp_extension import load


def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

# 进入 load_model（openllama）

def load_model(args):
    import torch
    agent_name = args['models'][args['model']]['agent_name']  # agent_name=DeepSdeedAgent
    model_name = args['models'][args['model']]['model_name']  # model_name=OpenLAMAPEFTModel
    if model_name == "OpenLLAMAPEFTModel":
        model = OpenLLAMAPEFTModel(**args)
        # 加载projector权重
        saved_state = torch.load(
            "./ckpt/Llama-3-8B-Instruct_mlp/pytorch_model_step_170000.pt",map_location='cpu')
        # 确保在DeepSpeed包装之前加载权重
        # 加载权重到 llama_proj
        # model.llama_proj.weight.data.copy_(saved_state['llama_proj.weight'])
        # model.llama_proj.bias.data.copy_(saved_state['llama_proj.bias'])

        # 加载权重到 llama_proj_mlp
        model.llama_proj_mlp.linear.weight.data.copy_(saved_state['llama_proj_mlp.linear.weight'])
        model.llama_proj_mlp.linear.bias.data.copy_(saved_state['llama_proj_mlp.linear.bias'])

    else:
        raise ValueError(f"Model {model_name} not found.")
    print("Before agent instantiation")
    print(f"Available global names: {list(globals().keys())}")
    print(f"Looking for {agent_name} in globals()")
    import torch._C
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)

    agent = globals()[agent_name](model, args)
    print("Agent instantiated successfully")
    return agent
