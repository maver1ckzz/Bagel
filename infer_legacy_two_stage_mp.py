#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU legacy two-stage inference")
    parser.add_argument("--input-json", type=str, required=True, help="JSON file containing a list of {prompt,img_path,id,obj}")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated middle images and jsonl outputs")
    parser.add_argument("--model-path", type=str, required=True, help="Directory containing llm_config.json/vit_config.json/ae.safetensors")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint safetensors. Defaults to model-path/ema.safetensors")
    parser.add_argument("--base-model-path", type=str, default=None, help="Optional base model dir used to fill missing weights")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated CUDA device ids. Default: use all visible GPUs")
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--cfg-text-scale", type=float, default=3.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval-start", type=float, default=0.4)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", type=str, default="global")
    parser.add_argument("--text-temperature", type=float, default=0.3)
    parser.add_argument("--max-think-token-n", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries whose image path already exists")
    return parser.parse_args()


def resolve_gpus(gpus_arg: Optional[str]) -> List[int]:
    if gpus_arg:
        return [int(x) for x in gpus_arg.split(",") if x.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return list(range(len([x for x in visible.split(",") if x.strip()])))
    return list(range(torch.cuda.device_count()))


def load_state_dict_with_fallback(checkpoint_path: str, base_model_path: Optional[str]) -> Dict[str, torch.Tensor]:
    state_dict = load_file(checkpoint_path, device="cpu")
    if base_model_path is None:
        return state_dict

    base_candidates = [
        os.path.join(base_model_path, "ema.safetensors"),
        os.path.join(base_model_path, "model.safetensors"),
    ]
    base_ckpt = next((p for p in base_candidates if os.path.exists(p)), None)
    if base_ckpt is None:
        raise FileNotFoundError(f"Could not find base model safetensors under {base_model_path}")

    base_state_dict = load_file(base_ckpt, device="cpu")
    merged = dict(base_state_dict)
    merged.update(state_dict)
    return merged


def build_inferencer(args: argparse.Namespace, cuda_index: int) -> InterleaveInferencer:
    device = torch.device(f"cuda:{cuda_index}")
    torch.cuda.set_device(device)
    infer_dtype = torch.bfloat16

    model_path = args.model_path
    checkpoint_path = args.checkpoint_path or os.path.join(model_path, "ema.safetensors")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict = load_state_dict_with_fallback(checkpoint_path, args.base_model_path)
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(f"[GPU {cuda_index}] load_state_dict: {msg}")
    del model_state_dict

    model = model.to(device=device, dtype=infer_dtype).eval()
    vae_model = vae_model.to(device=device, dtype=infer_dtype).eval()
    print(f"[GPU {cuda_index}] inference dtype: {infer_dtype}")

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


def save_result_image(image: Image.Image, output_dir: Path, item: Dict[str, Any]) -> str:
    sample_dir = output_dir / "img" / str(item["id"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = sample_dir / f'{item["obj"]}.png'
    image.save(image_path)
    return str(image_path)


def merge_result_files(output_dir: Path) -> Path:
    merged_path = output_dir / "results.jsonl"
    rank_files = sorted(output_dir.glob("results.rank*.jsonl"))
    with merged_path.open("w", encoding="utf-8") as fout:
        for rank_file in rank_files:
            with rank_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    return merged_path


def run_worker(worker_rank: int, cuda_index: int, items: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    inferencer = build_inferencer(args, cuda_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"results.rank{worker_rank}.jsonl"

    cfg_interval = (args.cfg_interval_start, args.cfg_interval_end)

    with result_path.open("w", encoding="utf-8") as fp:
        for local_idx, item in enumerate(items, start=1):
            image_output_path = output_dir / "img" / str(item["id"]) / f'{item["obj"]}.png'
            if args.skip_existing and image_output_path.exists():
                print(f"[GPU {cuda_index}] skip existing {image_output_path}")
                continue

            try:
                image = Image.open(item["img_path"]).convert("RGB")
                output = inferencer(
                    image=image,
                    text=item["prompt"],
                    image_first=True,
                    do_sample=args.do_sample,
                    text_temperature=args.text_temperature,
                    max_think_token_n=args.max_think_token_n,
                    cfg_text_scale=args.cfg_text_scale,
                    cfg_img_scale=args.cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=args.timestep_shift,
                    num_timesteps=args.num_timesteps,
                    cfg_renorm_min=args.cfg_renorm_min,
                    cfg_renorm_type=args.cfg_renorm_type,
                )
                saved_image_path = save_result_image(output["image"], output_dir, item)
                record = {
                    "id": item["id"],
                    "obj": item["obj"],
                    "img_path": item["img_path"],
                    "prompt": item["prompt"],
                    "generated_image_path": saved_image_path,
                    "generated_text": output["text"],
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                fp.flush()
                print(f"[GPU {cuda_index}] {local_idx}/{len(items)} done: {item['id']} {item['obj']}")
            except Exception as exc:
                record = {
                    "id": item.get("id"),
                    "obj": item.get("obj"),
                    "img_path": item.get("img_path"),
                    "prompt": item.get("prompt"),
                    "error": repr(exc),
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                fp.flush()
                print(f"[GPU {cuda_index}] failed: {item.get('id')} {item.get('obj')} -> {exc}")


def shard_items(items: List[Dict[str, Any]], world_size: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(world_size)]
    for idx, item in enumerate(items):
        shards[idx % world_size].append(item)
    return shards


def main() -> None:
    args = parse_args()
    gpus = resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs available for inference.")

    items = json.loads(Path(args.input_json).read_text())
    assert isinstance(items, list), "input json must contain a list"
    for item in items:
        for key in ("prompt", "img_path", "id", "obj"):
            if key not in item:
                raise ValueError(f"Missing key '{key}' in item: {item}")

    shards = shard_items(items, len(gpus))
    mp.set_start_method("spawn", force=True)
    processes = []
    for rank, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        proc = mp.Process(target=run_worker, args=(rank, gpu, shard, args))
        proc.start()
        processes.append(proc)

    exit_code = 0
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            exit_code = proc.exitcode

    if exit_code != 0:
        raise SystemExit(exit_code)

    merged_path = merge_result_files(Path(args.output_dir))
    print(f"Merged results written to {merged_path}")


if __name__ == "__main__":
    main()
