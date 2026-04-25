#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm.auto import tqdm

from infer_legacy_two_stage_mp import build_inferencer, resolve_gpus
from modeling.bagel.qwen2_navit import NaiveCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy two-phase batched inference")
    parser.add_argument("--input-json", type=str, required=True, help="JSON list of {prompt,img_path,id,obj}")
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
    parser.add_argument("--text-temperature", type=float, default=0.3)
    parser.add_argument("--max-text-token-n", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stage1-prompt-key", type=str, default="prompt")
    parser.add_argument("--stage2-prompt-key", type=str, default="diagnosis_prompt")
    parser.add_argument("--skip-existing-images", action="store_true")
    parser.add_argument("--skip-existing-diagnosis", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip stage1 image generation and only run stage2 diagnosis")
    return parser.parse_args()


def chunked(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def shard_items(items: List[Dict[str, Any]], world_size: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(world_size)]
    for idx, item in enumerate(items):
        shards[idx % world_size].append(item)
    return shards


def get_generated_image_path(output_dir: Path, item: Dict[str, Any]) -> Path:
    return output_dir / "img" / str(item["id"]) / f'{item["obj"]}.png'


def get_diagnosis_record_path(output_dir: Path, item: Dict[str, Any]) -> Path:
    return output_dir / "diagnosis" / str(item["id"]) / f'{item["obj"]}.json'


def get_stage2_prompt(item: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.stage2_prompt_key in item and item[args.stage2_prompt_key]:
        return item[args.stage2_prompt_key]
    return item[args.stage1_prompt_key]


def get_stage1_manifest_path(output_dir: Path) -> Path:
    return output_dir / "stage1_results.jsonl"


def merge_stage_result_files(output_dir: Path, prefix: str) -> Path:
    merged_path = output_dir / f"{prefix}.jsonl"
    rank_files = sorted(output_dir.glob(f"{prefix}.rank*.jsonl"))
    with merged_path.open("w", encoding="utf-8") as fout:
        for rank_file in rank_files:
            with rank_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    return merged_path


def load_stage1_generated_paths(output_dir: Path) -> Dict[str, str]:
    manifest_path = get_stage1_manifest_path(output_dir)
    if not manifest_path.exists():
        return {}

    generated_paths = {}
    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            if "generated_image_path" not in record:
                continue
            key = f'{record.get("id")}::{record.get("obj")}'
            generated_paths[key] = record["generated_image_path"]
    return generated_paths


def resolve_generated_image_path(
    output_dir: Path,
    item: Dict[str, Any],
    generated_paths: Dict[str, str],
) -> Path:
    key = f'{item["id"]}::{item["obj"]}'
    if key in generated_paths:
        return Path(generated_paths[key])
    return get_generated_image_path(output_dir, item)


def move_generation_input_to_device(
    generation_input: Dict[str, Any],
    device: torch.device,
    float_dtype: Optional[torch.dtype] = None,
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


def batch_generate_images(
    inferencer,
    original_images: List[Image.Image],
    prompts: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Image.Image]:
    model_dtype = next(inferencer.model.parameters()).dtype
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        resized_images = [inferencer.vae_transform.resize_transform(img) for img in original_images]
        image_shapes = [img.size[::-1] for img in resized_images]

        gen_context = init_batch_context(inferencer, len(resized_images))
        cfg_text_context = init_batch_context(inferencer, len(resized_images))
        cfg_img_context = init_batch_context(inferencer, len(resized_images))

        gen_context = batch_update_images(inferencer, resized_images, gen_context, device, vae=True, vit=True)
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


def batch_generate_texts(
    inferencer,
    original_images: List[Image.Image],
    prompts: List[str],
    generated_images: List[Image.Image],
    args: argparse.Namespace,
    device: torch.device,
) -> List[str]:
    model_dtype = next(inferencer.model.parameters()).dtype
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        original_images_resized = [inferencer.vae_transform.resize_transform(img) for img in original_images]
        generated_images_resized = [inferencer.vae_transform.resize_transform(img) for img in generated_images]

        gen_context = init_batch_context(inferencer, len(prompts))
        # Keep stage-2 context aligned with the single-sample image_first path:
        # original image -> text prompt -> generated image, and both images enter via VAE+ViT.
        gen_context = batch_update_images(
            inferencer,
            original_images_resized,
            gen_context,
            device,
            vae=True,
            vit=True,
        )
        gen_context = batch_update_text(inferencer, prompts, gen_context, device)
        gen_context = batch_update_images(
            inferencer,
            generated_images_resized,
            gen_context,
            device,
            vae=True,
            vit=True,
        )

        generation_input = inferencer.model.prepare_start_tokens(
            gen_context["kv_lens"], gen_context["ropes"], inferencer.new_token_ids
        )
        generation_input = move_generation_input_to_device(generation_input, device, float_dtype=model_dtype)

        generated = inferencer.model.generate_text(
            past_key_values=gen_context["past_key_values"],
            max_length=args.max_text_token_n,
            do_sample=args.do_sample,
            temperature=args.text_temperature,
            end_token_id=None,
            **generation_input,
        )

    eos_token_id = inferencer.new_token_ids["eos_token_id"]
    outputs = []
    for i in range(generated.shape[1]):
        token_ids = generated[:, i].tolist()
        if eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(eos_token_id) + 1]
        output = inferencer.tokenizer.decode(token_ids)
        if "<|im_start|>" in output and "<|im_end|>" in output:
            output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        outputs.append(output)
    return outputs


def run_stage1_worker(worker_rank: int, cuda_index: int, items: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    inferencer = build_inferencer(args, cuda_index)
    device = next(inferencer.model.parameters()).device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"stage1_results.rank{worker_rank}.jsonl"

    batches = chunked(items, args.batch_size)
    progress = tqdm(
        batches,
        desc=f"Stage1 GPU{cuda_index}",
        position=worker_rank,
        leave=True,
    )

    with result_path.open("w", encoding="utf-8") as fp:
        for batch in progress:
            pending = []
            for item in batch:
                image_output_path = get_generated_image_path(output_dir, item)
                if args.skip_existing_images and image_output_path.exists():
                    print(f"[Stage1 GPU {cuda_index}] skip existing {image_output_path}")
                    continue
                pending.append(item)
            if not pending:
                continue

            try:
                original_images = [Image.open(item["img_path"]).convert("RGB") for item in pending]
                prompts = [item[args.stage1_prompt_key] for item in pending]
                generated_images = batch_generate_images(inferencer, original_images, prompts, args, device)

                for item, prompt, image in zip(pending, prompts, generated_images):
                    image_output_path = get_generated_image_path(output_dir, item)
                    image_output_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_output_path)
                    record = {
                        "id": item["id"],
                        "obj": item["obj"],
                        "img_path": item["img_path"],
                        "stage1_prompt": prompt,
                        "generated_image_path": str(image_output_path),
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fp.flush()
                progress.set_postfix(done=len(pending), mode="generated")
            except Exception as exc:
                for item in pending:
                    record = {
                        "id": item.get("id"),
                        "obj": item.get("obj"),
                        "img_path": item.get("img_path"),
                        "stage1_prompt": item.get(args.stage1_prompt_key),
                        "error": repr(exc),
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fp.flush()
                print(f"[Stage1 GPU {cuda_index}] batch failed -> {exc}")
                progress.set_postfix(done=len(pending), mode="failed")

    progress.close()


def run_stage2_worker(worker_rank: int, cuda_index: int, items: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    inferencer = build_inferencer(args, cuda_index)
    device = next(inferencer.model.parameters()).device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"stage2_results.rank{worker_rank}.jsonl"
    generated_paths = load_stage1_generated_paths(output_dir)

    batches = chunked(items, args.batch_size)
    progress = tqdm(
        batches,
        desc=f"Stage2 GPU{cuda_index}",
        position=worker_rank,
        leave=True,
    )

    with result_path.open("w", encoding="utf-8") as fp:
        for batch in progress:
            pending = []
            for item in batch:
                diagnosis_path = get_diagnosis_record_path(output_dir, item)
                if args.skip_existing_diagnosis and diagnosis_path.exists():
                    print(f"[Stage2 GPU {cuda_index}] skip existing {diagnosis_path}")
                    continue
                pending.append(item)
            if not pending:
                continue

            try:
                original_images = [Image.open(item["img_path"]).convert("RGB") for item in pending]
                generated_image_paths = [resolve_generated_image_path(output_dir, item, generated_paths) for item in pending]
                generated_images = [Image.open(path).convert("RGB") for path in generated_image_paths]
                prompts = [get_stage2_prompt(item, args) for item in pending]
                diagnoses = batch_generate_texts(inferencer, original_images, prompts, generated_images, args, device)

                for item, prompt, diagnosis, generated_image_path in zip(
                    pending, prompts, diagnoses, generated_image_paths
                ):
                    record = {
                        "id": item["id"],
                        "obj": item["obj"],
                        "img_path": item["img_path"],
                        "generated_image_path": str(generated_image_path),
                        "stage2_prompt": prompt,
                        "diagnosis": diagnosis,
                    }
                    diagnosis_path = get_diagnosis_record_path(output_dir, item)
                    diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnosis_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fp.flush()
                progress.set_postfix(done=len(pending), mode="diagnosed")
            except Exception as exc:
                for item in pending:
                    generated_image_path = resolve_generated_image_path(output_dir, item, generated_paths)
                    record = {
                        "id": item.get("id"),
                        "obj": item.get("obj"),
                        "img_path": item.get("img_path"),
                        "generated_image_path": str(generated_image_path),
                        "stage2_prompt": get_stage2_prompt(item, args),
                        "error": repr(exc),
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fp.flush()
                print(f"[Stage2 GPU {cuda_index}] batch failed -> {exc}")
                progress.set_postfix(done=len(pending), mode="failed")

    progress.close()


def launch_stage(stage_name: str, worker_fn, gpus: List[int], shards: List[List[Dict[str, Any]]], args: argparse.Namespace) -> None:
    mp.set_start_method("spawn", force=True)
    processes = []
    for rank, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        proc = mp.Process(target=worker_fn, args=(rank, gpu, shard, args))
        proc.start()
        processes.append(proc)

    exit_code = 0
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            exit_code = proc.exitcode
    if exit_code != 0:
        raise SystemExit(exit_code)

    merged = merge_stage_result_files(Path(args.output_dir), stage_name)
    print(f"{stage_name} merged results written to {merged}")


def validate_items(items: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    for item in items:
        for key in ("img_path", "id", "obj", args.stage1_prompt_key):
            if key not in item:
                raise ValueError(f"Missing key '{key}' in item: {item}")


def validate_stage2_inputs(items: List[Dict[str, Any]], output_dir: Path) -> None:
    generated_paths = load_stage1_generated_paths(output_dir)
    missing = []
    for item in items:
        generated_path = resolve_generated_image_path(output_dir, item, generated_paths)
        if not generated_path.exists():
            missing.append(str(generated_path))
            if len(missing) >= 5:
                break
    if missing:
        raise FileNotFoundError(
            "skip-stage1 was set, but generated images are missing. "
            f"Examples: {missing}"
        )


def main() -> None:
    args = parse_args()
    gpus = resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs available for inference.")

    items = json.loads(Path(args.input_json).read_text())
    assert isinstance(items, list), "input json must contain a list"
    validate_items(items, args)
    shards = shard_items(items, len(gpus))
    output_dir = Path(args.output_dir)

    print("Bagel bottom-level support status:")
    print("- prepare_prompts / prepare_vit_images / prepare_vae_images / prepare_vae_latent already support batched list inputs.")
    print("- generate_image supports packed batched generation.")
    print("- generate_text is partially batched, but end-token stopping is batch=1 oriented; this script uses fixed max tokens and trims manually.")
    print("- stage1_results.jsonl stores metadata/index records, not image bytes; generated images are written under output_dir/img/...")

    if args.skip_stage1:
        print("=== Stage 1: skipped ===")
        validate_stage2_inputs(items, output_dir)
    else:
        print("=== Stage 1: batch generate all middle images ===")
        launch_stage("stage1_results", run_stage1_worker, gpus, shards, args)

    print("=== Stage 2: batch final diagnosis ===")
    launch_stage("stage2_results", run_stage2_worker, gpus, shards, args)


if __name__ == "__main__":
    main()
