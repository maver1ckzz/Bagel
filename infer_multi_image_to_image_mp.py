#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import torch.multiprocessing as mp
import torch
from PIL import Image
from tqdm.auto import tqdm

from infer_legacy_two_stage_mp import build_inferencer, resolve_gpus
from modeling.bagel.qwen2_navit import NaiveCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU multi-image to image inference")
    parser.add_argument("--input-json", type=str, required=True, help="JSON list of {id,image1,image2,prompt}")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--cfg-text-scale", type=float, default=3.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval-start", type=float, default=0.4)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", type=str, default="global")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def shard_items(items: List[Dict[str, Any]], world_size: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(world_size)]
    for idx, item in enumerate(items):
        shards[idx % world_size].append(item)
    return shards


def get_output_image_path(output_dir: Path, item: Dict[str, Any]) -> Path:
    return output_dir / "img" / f'{item["id"]}.png'


def chunked(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def merge_result_files(output_dir: Path) -> Path:
    merged_path = output_dir / "results.jsonl"
    rank_files = sorted(output_dir.glob("results.rank*.jsonl"))
    with merged_path.open("w", encoding="utf-8") as fout:
        for rank_file in rank_files:
            with rank_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    return merged_path


def move_generation_input_to_device(
    generation_input: Dict[str, Any],
    device: torch.device,
    float_dtype: torch.dtype | None = None,
) -> Dict[str, Any]:
    moved = {}
    for key, value in generation_input.items():
        if torch.is_tensor(value):
            if float_dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def init_batch_context(inferencer, batch_size: int) -> Dict[str, Any]:
    return {
        "kv_lens": [0] * batch_size,
        "ropes": [0] * batch_size,
        "past_key_values": NaiveCache(inferencer.model.config.llm_config.num_hidden_layers),
    }


def batch_update_text(inferencer, texts: List[str], gen_context: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model_dtype = next(inferencer.model.parameters()).dtype
    generation_input, kv_lens, ropes = inferencer.model.prepare_prompts(
        curr_kvlens=gen_context["kv_lens"],
        curr_rope=gen_context["ropes"],
        prompts=texts,
        tokenizer=inferencer.tokenizer,
        new_token_ids=inferencer.new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device, float_dtype=model_dtype)
    past_key_values = inferencer.model.forward_cache_update_text(gen_context["past_key_values"], **generation_input)
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def batch_update_images(
    inferencer,
    images: List[Image.Image],
    gen_context: Dict[str, Any],
    device: torch.device,
    vae: bool,
    vit: bool,
) -> Dict[str, Any]:
    assert vae or vit
    model_dtype = next(inferencer.model.parameters()).dtype
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    if vae:
        generation_input, kv_lens, ropes = inferencer.model.prepare_vae_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=images,
            transforms=inferencer.vae_transform,
            new_token_ids=inferencer.new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device, float_dtype=model_dtype)
        past_key_values = inferencer.model.forward_cache_update_vae(
            inferencer.vae_model, past_key_values, **generation_input
        )

    if vit:
        generation_input, kv_lens, ropes = inferencer.model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=images,
            transforms=inferencer.vit_transform,
            new_token_ids=inferencer.new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device, float_dtype=model_dtype)
        past_key_values = inferencer.model.forward_cache_update_vit(past_key_values, **generation_input)

    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def batch_generate_images_from_multi_inputs(
    inferencer,
    image1_list: List[Image.Image],
    image2_list: List[Image.Image],
    prompts: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Image.Image]:
    model_dtype = next(inferencer.model.parameters()).dtype
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        resized_image1 = [inferencer.vae_transform.resize_transform(img) for img in image1_list]
        resized_image2 = [inferencer.vae_transform.resize_transform(img) for img in image2_list]
        image_shapes = [img.size[::-1] for img in resized_image2]

        gen_context = init_batch_context(inferencer, len(prompts))
        cfg_text_context = init_batch_context(inferencer, len(prompts))
        cfg_img_context = init_batch_context(inferencer, len(prompts))

        gen_context = batch_update_images(inferencer, resized_image1, gen_context, device, vae=True, vit=True)
        gen_context = batch_update_images(inferencer, resized_image2, gen_context, device, vae=True, vit=True)
        cfg_text_context = deepcopy(gen_context)
        gen_context = batch_update_text(inferencer, prompts, gen_context, device)
        cfg_img_context = batch_update_text(inferencer, prompts, cfg_img_context, device)

        generation_input = inferencer.model.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=image_shapes,
            new_token_ids=inferencer.new_token_ids,
        )
        generation_input_cfg_text = inferencer.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=image_shapes,
        )
        generation_input_cfg_img = inferencer.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=image_shapes,
        )

        generation_input = move_generation_input_to_device(generation_input, device, float_dtype=model_dtype)
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device, float_dtype=model_dtype)
        generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device, float_dtype=model_dtype)

        unpacked_latents = inferencer.model.generate_image(
            past_key_values=gen_context["past_key_values"],
            cfg_text_past_key_values=cfg_text_context["past_key_values"],
            cfg_img_past_key_values=cfg_img_context["past_key_values"],
            num_timesteps=args.num_timesteps,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=(args.cfg_interval_start, args.cfg_interval_end),
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
            timestep_shift=args.timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
        )

    return [inferencer.decode_image(latent, image_shape) for latent, image_shape in zip(unpacked_latents, image_shapes)]


def run_worker(worker_rank: int, cuda_index: int, items: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    inferencer = build_inferencer(args, cuda_index)
    device = next(inferencer.model.parameters()).device
    output_dir = Path(args.output_dir)
    (output_dir / "img").mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"results.rank{worker_rank}.jsonl"

    progress = tqdm(
        chunked(items, args.batch_size),
        desc=f"GPU{cuda_index}",
        position=worker_rank,
        leave=True,
    )

    with result_path.open("w", encoding="utf-8") as fp:
        for batch in progress:
            pending = []
            for item in batch:
                output_image_path = get_output_image_path(output_dir, item)
                if args.skip_existing and output_image_path.exists():
                    continue
                pending.append(item)

            if not pending:
                progress.set_postfix(mode="skip")
                continue

            try:
                image1_list = [Image.open(item["image1"]).convert("RGB") for item in pending]
                image2_list = [Image.open(item["image2"]).convert("RGB") for item in pending]
                prompts = [item["prompt"] for item in pending]
                generated_images = batch_generate_images_from_multi_inputs(
                    inferencer,
                    image1_list=image1_list,
                    image2_list=image2_list,
                    prompts=prompts,
                    args=args,
                    device=device,
                )

                for item, image in zip(pending, generated_images):
                    output_image_path = get_output_image_path(output_dir, item)
                    image.save(output_image_path)
                    record = {
                        "id": item["id"],
                        "image1": item["image1"],
                        "image2": item["image2"],
                        "prompt": item["prompt"],
                        "generated_image_path": str(output_image_path),
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fp.flush()
                progress.set_postfix(done=len(pending), mode="saved")
            except Exception as exc:
                for item in pending:
                    record = {
                        "id": item.get("id"),
                        "image1": item.get("image1"),
                        "image2": item.get("image2"),
                        "prompt": item.get("prompt"),
                        "error": repr(exc),
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fp.flush()
                progress.set_postfix(done=len(pending), mode="failed")

    progress.close()


def validate_items(items: List[Dict[str, Any]]) -> None:
    for item in items:
        for key in ("id", "image1", "image2", "prompt"):
            if key not in item:
                raise ValueError(f"Missing key '{key}' in item: {item}")


def main() -> None:
    args = parse_args()
    gpus = resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs available for inference.")

    items = json.loads(Path(args.input_json).read_text())
    if not isinstance(items, list):
        raise ValueError("input json must contain a list")
    validate_items(items)

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
