import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_HOME"] = ""  #/mnt/nvme_share/wangty/cuda-12.4
# os.environ["LD_LIBRARY_PATH"] = "" #$CUDA_HOME/lib64:$LD_LIBRARY_PATH
import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
model_path = "/hdd/wangty/model/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)
import os
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    raise ImportError("safetensors library not found. Install with: pip install safetensors")

def load_safetensors_dict(file_path):
    tensors = {}
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def merge_safetensors(source_file, target_file, output_file):
    print(f"正在读取官方底模 (Source): {source_file}")
    source_tensors = load_safetensors_dict(source_file)
    
    print(f"正在读取微调权重 (Target): {target_file}")
    target_tensors = load_safetensors_dict(target_file)
    
    source_keys = set(source_tensors.keys())
    target_keys = set(target_tensors.keys())
    missing_keys = source_keys - target_keys
    
    print(f"发现微调权重缺失 {len(missing_keys)} 个参数 (通常是没有微调的 ViT 等部分)")
    
    if len(missing_keys) == 0:
        print("未发现缺失参数，无需合并。")
        return True
        
    merged_tensors = target_tensors.copy()
    print("正在将底模的缺失参数复制到微调权重中...")
    for key in missing_keys:
        merged_tensors[key] = source_tensors[key]
        
    print(f"正在保存完整的合并权重至: {output_file}")
    save_file(merged_tensors, output_file)
    print("合并完成！\n")
    return True

base_model_path = "/hdd/wangty/model/BAGEL-7B-MoT"
base_ema_file = os.path.join(base_model_path, "ema.safetensors")

finetuned_checkpoint_path = '/hdd/wangty/diffuser_workdir/bagel/workdir/test/result/new'
finetuned_ema_file = os.path.join(finetuned_checkpoint_path, "ema.safetensors")

merged_ema_file = os.path.join(finetuned_checkpoint_path, "merged_ema.safetensors")

if not os.path.exists(merged_ema_file):
    print("未检测到合并后的完整权重，准备开始合并...")
    merge_safetensors(base_ema_file, finetuned_ema_file, merged_ema_file)
else:
    print(f"检测到已存在的完整权重文件: {merged_ema_file}，将直接加载。")


max_mem_per_gpu = "80GiB"  

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print("Device Map:", device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# 加载合并后的完整权重 (merged_ema.safetensors)
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=merged_ema_file,  # <-- 这里改为加载刚刚合并好的完整文件
    device_map=device_map,
    offload_buffers=False,
    dtype=torch.bfloat16,
    force_hooks=True,
    #offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded successfully with complete parameters!')
from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
inference_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=1.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
    image_shapes=(512,512),
)
#prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
prompt='a cervical spine MRI sagittal T2 image'
print(prompt)
print('-' * 10)
output_dict = inferencer(text=prompt, **inference_hyper)
display(output_dict['image'])