#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image
from tqdm.auto import tqdm

from data.data_utils import pil_img2rgb
from infer_legacy_two_stage_mp import build_inferencer, resolve_gpus
from modeling.bagel.qwen2_navit import NaiveCache


SUPPORTED_BACKENDS = {"generate_image", "generate_text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven batched inference pipeline")
    parser.add_argument("--input-json", type=str, required=True, help="JSON list of samples")
    parser.add_argument("--pipeline-config", type=str, required=True, help="YAML pipeline config")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=str,
        default="4",
        help="Global batch size or per-round batch sizes, e.g. '8' or '8,4,2'",
    )
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--skip-existing-final", action="store_true")
    parser.add_argument(
        "--skip-existing-stage-output",
        action="store_true",
        help="Skip any round whose declared output already exists",
    )
    parser.add_argument(
        "--start-round",
        type=str,
        default=None,
        help="Start from a specific round name or 1-based round index",
    )
    return parser.parse_args()


def chunked(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def shard_items(items: List[Dict[str, Any]], world_size: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(world_size)]
    for idx, item in enumerate(items):
        shards[idx % world_size].append(item)
    return shards


def merge_result_files(output_dir: Path) -> Path:
    merged_path = output_dir / "results.jsonl"
    rank_files = sorted(output_dir.glob("results.rank*.jsonl"))
    with merged_path.open("w", encoding="utf-8") as fout:
        for rank_file in rank_files:
            with rank_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    return merged_path


def merge_state_files(output_dir: Path) -> Path:
    merged_path = output_dir / "state.jsonl"
    rank_files = sorted(output_dir.glob("state.rank*.jsonl"))
    with merged_path.open("w", encoding="utf-8") as fout:
        for rank_file in rank_files:
            with rank_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    return merged_path


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
    vae: bool = True,
    vit: bool = True,
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


def decode_batch_texts(inferencer, generated: torch.Tensor) -> List[str]:
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


def apply_stop_strings(
    texts: List[str],
    stop_on_substring: Optional[Any],
    include_stop: bool = True,
) -> List[str]:
    if stop_on_substring is None:
        return texts

    if isinstance(stop_on_substring, str):
        stop_strings = [stop_on_substring]
    elif isinstance(stop_on_substring, list) and all(isinstance(item, str) for item in stop_on_substring):
        stop_strings = stop_on_substring
    else:
        raise TypeError("params.stop_on_substring must be a string or a list of strings")

    truncated = []
    for text in texts:
        best_pos = None
        best_stop = None
        for stop_string in stop_strings:
            pos = text.find(stop_string)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_stop = stop_string
        if best_pos is None:
            truncated.append(text)
            continue
        end_pos = best_pos + len(best_stop) if include_stop else best_pos
        truncated.append(text[:end_pos])
    return truncated


def validate_pipeline_config(config: Dict[str, Any]) -> None:
    if "base_input" not in config or not isinstance(config["base_input"], list):
        raise ValueError("pipeline config requires a list field 'base_input'")
    if "rounds" not in config or not isinstance(config["rounds"], list) or not config["rounds"]:
        raise ValueError("pipeline config requires a non-empty list field 'rounds'")
    for round_cfg in config["rounds"]:
        backend = round_cfg.get("backend")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Supported: {sorted(SUPPORTED_BACKENDS)}")


def validate_input_spec_list(specs: Sequence[Dict[str, Any]], field_name: str) -> None:
    for spec in specs:
        if spec.get("type") not in {"image", "text"}:
            raise ValueError(f"{field_name} item must have type=image|text, got {spec}")
        if "key" not in spec and "value" not in spec:
            raise ValueError(f"{field_name} item must have either key or value: {spec}")


def normalize_state(sample: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(sample)
    state["_history"] = list(sample.get("history", []))
    return state


def serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for key, value in state.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, list):
            serialized[key] = value
        elif isinstance(value, dict):
            serialized[key] = value
    if "_history" in state:
        serialized["history"] = state["_history"]
    return serialized


def resolve_text_value(state: Dict[str, Any], spec: Dict[str, Any]) -> str:
    value = spec["value"] if "value" in spec else state[spec["key"]]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"Text input must resolve to str, got {type(value)} from spec {spec}")
    return value


def resolve_image_value(state: Dict[str, Any], spec: Dict[str, Any]) -> Image.Image:
    value = spec["value"] if "value" in spec else state[spec["key"]]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Image input must resolve to PIL.Image or image path str, got {type(value)} from spec {spec}")


def build_stage_inputs(states: List[Dict[str, Any]], refs: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    return [[dict(spec) for spec in refs] for _ in states]


def parse_batch_size_schedule(batch_size_arg: str) -> List[int]:
    values = [part.strip() for part in batch_size_arg.split(",") if part.strip()]
    if not values:
        raise ValueError("--batch-size must contain at least one positive integer")
    schedule = [int(value) for value in values]
    if any(value <= 0 for value in schedule):
        raise ValueError("--batch-size values must be positive integers")
    return schedule


def resolve_start_round_index(rounds: Sequence[Dict[str, Any]], start_round: Optional[str]) -> int:
    if start_round is None:
        return 0
    if start_round.isdigit():
        index = int(start_round) - 1
        if not 0 <= index < len(rounds):
            raise ValueError(f"--start-round index out of range: {start_round}")
        return index
    for index, round_cfg in enumerate(rounds):
        if round_cfg["name"] == start_round:
            return index
    raise ValueError(f"--start-round '{start_round}' does not match any round name")


def load_existing_states(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    candidates = [output_dir / "state.jsonl", output_dir / "results.jsonl"]
    state: Dict[str, Dict[str, Any]] = {}
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is not None:
        with source_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                state[str(record["id"])] = record
        return state

    rank_files = sorted(output_dir.glob("state.rank*.jsonl"))
    if not rank_files:
        rank_files = sorted(output_dir.glob("results.rank*.jsonl"))
    for rank_file in rank_files:
        with rank_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                state[str(record["id"])] = record
    return state


def find_existing_state_source(output_dir: Path) -> Optional[Path]:
    preferred = [output_dir / "state.jsonl", output_dir / "results.jsonl"]
    for path in preferred:
        if path.exists():
            return path

    rank_files = sorted(output_dir.glob("state.rank*.jsonl"))
    if rank_files:
        return rank_files[0]

    rank_files = sorted(output_dir.glob("results.rank*.jsonl"))
    if rank_files:
        return rank_files[0]

    return None


def merge_loaded_state(sample: Dict[str, Any], loaded: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if loaded is None:
        return sample
    merged = dict(sample)
    for key, value in loaded.items():
        if key == "id":
            continue
        merged[key] = value
    return merged


def build_batch_context_from_sequence(
    inferencer,
    states: List[Dict[str, Any]],
    refs: Sequence[Dict[str, Any]],
    device: torch.device,
    default_image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Tuple[int, int]]]:
    gen_context = init_batch_context(inferencer, len(states))
    cfg_text_context = init_batch_context(inferencer, len(states))
    cfg_img_context = init_batch_context(inferencer, len(states))
    image_shapes: Optional[List[Tuple[int, int]]] = None

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for spec in refs:
            if spec["type"] == "text":
                texts = [resolve_text_value(state, spec) for state in states]
                cfg_text_context = deepcopy(gen_context)
                gen_context = batch_update_text(inferencer, texts, gen_context, device)
                cfg_img_context = batch_update_text(inferencer, texts, cfg_img_context, device)
            elif spec["type"] == "image":
                images = [resolve_image_value(state, spec) for state in states]
                image_shapes = [
                    inferencer.vae_transform.resize_transform(pil_img2rgb(img)).size[::-1]
                    for img in images
                ]
                gen_context = batch_update_images(inferencer, images, gen_context, device, vae=True, vit=True)
                cfg_text_context = deepcopy(gen_context)
            else:
                raise ValueError(f"Unsupported input spec type: {spec['type']}")

    if image_shapes is None:
        if default_image_size is None:
            raise ValueError("No image found in accumulated context and no params.image_size provided for image generation.")
        image_shapes = [tuple(default_image_size) for _ in states]

    return gen_context, cfg_text_context, cfg_img_context, image_shapes


def batch_generate_images_from_sequence(
    inferencer,
    states: List[Dict[str, Any]],
    refs: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    device: torch.device,
) -> List[Image.Image]:
    model_dtype = next(inferencer.model.parameters()).dtype
    image_size = params.get("image_size")
    gen_context, cfg_text_context, cfg_img_context, image_shapes = build_batch_context_from_sequence(
        inferencer,
        states,
        refs,
        device,
        default_image_size=tuple(image_size) if image_size is not None else None,
    )

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
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
            num_timesteps=params.get("num_timesteps", 50),
            cfg_text_scale=params.get("cfg_text_scale", 3.0),
            cfg_img_scale=params.get("cfg_img_scale", 1.5),
            cfg_interval=tuple(params.get("cfg_interval", [0.4, 1.0])),
            cfg_renorm_min=params.get("cfg_renorm_min", 0.0),
            cfg_renorm_type=params.get("cfg_renorm_type", "global"),
            timestep_shift=params.get("timestep_shift", 3.0),
            enable_taylorseer=params.get("enable_taylorseer", False),
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


def batch_generate_texts_from_sequence(
    inferencer,
    states: List[Dict[str, Any]],
    refs: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    device: torch.device,
) -> List[str]:
    model_dtype = next(inferencer.model.parameters()).dtype
    gen_context, _, _, _ = build_batch_context_from_sequence(
        inferencer,
        states,
        refs,
        device,
        default_image_size=(1024, 1024),
    )

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = inferencer.model.prepare_start_tokens(
            gen_context["kv_lens"], gen_context["ropes"], inferencer.new_token_ids
        )
        generation_input = move_generation_input_to_device(generation_input, device, float_dtype=model_dtype)
        generated = inferencer.model.generate_text(
            past_key_values=gen_context["past_key_values"],
            max_length=params.get("max_text_token_n", 256),
            do_sample=params.get("do_sample", False),
            temperature=params.get("text_temperature", 0.3),
            end_token_id=None,
            **generation_input,
        )

    outputs = decode_batch_texts(inferencer, generated)
    outputs = apply_stop_strings(
        outputs,
        params.get("stop_on_substring"),
        include_stop=params.get("include_stop_substring", True),
    )
    return outputs


def get_round_batch_size(round_cfg: Dict[str, Any], round_index: int, batch_size_schedule: Sequence[int]) -> int:
    if "batch_size" in round_cfg:
        return int(round_cfg["batch_size"])
    if round_index < len(batch_size_schedule):
        return batch_size_schedule[round_index]
    return batch_size_schedule[-1]


def get_output_image_filename(state: Dict[str, Any], round_cfg: Dict[str, Any]) -> str:
    output_cfg = round_cfg.get("output", {})
    filename_key = output_cfg.get("image_name_key")
    if filename_key is None:
        return f"{state['id']}.png"
    filename_value = state.get(filename_key)
    if filename_value is None:
        raise ValueError(
            f"Round '{round_cfg['name']}' requires image_name_key='{filename_key}', "
            f"but sample id={state.get('id')} is missing that field."
        )
    return f"{filename_value}.png"


def get_output_image_path(output_dir: Path, output_key: str, filename: str) -> Path:
    return output_dir / "images" / output_key / filename


def get_final_output_spec(rounds: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    final_image_round = None
    final_image_key = None
    final_text_key = None
    for round_cfg in rounds:
        output_cfg = round_cfg.get("output", {})
        if "image" in output_cfg:
            final_image_round = round_cfg
            final_image_key = output_cfg["image"]
        if "text" in output_cfg:
            final_text_key = output_cfg["text"]
    return final_image_round, final_image_key, final_text_key


def maybe_skip_final(
    state: Dict[str, Any],
    output_dir: Path,
    final_image_round: Optional[Dict[str, Any]],
    final_image_key: Optional[str],
    final_text_key: Optional[str],
) -> bool:
    if final_image_key is not None:
        final_path = get_output_image_path(
            output_dir,
            final_image_key,
            get_output_image_filename(state, final_image_round),
        )
        return final_path.exists()
    if final_text_key is not None:
        return final_text_key in state and isinstance(state[final_text_key], str)
    return False


def maybe_skip_round_output(
    state: Dict[str, Any],
    round_cfg: Dict[str, Any],
    output_dir: Path,
) -> bool:
    output_cfg = round_cfg.get("output", {})
    image_key = output_cfg.get("image")
    text_key = output_cfg.get("text")

    if image_key is not None:
        image_path = get_output_image_path(output_dir, image_key, get_output_image_filename(state, round_cfg))
        if image_path.exists():
            state[image_key] = str(image_path)
            return True
        return False

    if text_key is not None and isinstance(state.get(text_key), str):
        return True

    return False


def persist_states(state_path: Path, states: Sequence[Dict[str, Any]]) -> None:
    with state_path.open("w", encoding="utf-8") as fp:
        for state in states:
            fp.write(json.dumps(serialize_state(state), ensure_ascii=False) + "\n")


def log(message: str) -> None:
    print(message, flush=True)


def run_worker(
    worker_rank: int,
    cuda_index: int,
    items: List[Dict[str, Any]],
    args: argparse.Namespace,
    pipeline_config: Dict[str, Any],
) -> None:
    log(f"[Worker {worker_rank} | GPU {cuda_index}] loading model")
    inferencer = build_inferencer(args, cuda_index)
    device = next(inferencer.model.parameters()).device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"results.rank{worker_rank}.jsonl"
    state_path = output_dir / f"state.rank{worker_rank}.jsonl"

    base_input = pipeline_config["base_input"]
    rounds = pipeline_config["rounds"]
    final_image_round, final_image_key, final_text_key = get_final_output_spec(rounds)
    start_round_index = resolve_start_round_index(rounds, args.start_round)
    batch_size_schedule = parse_batch_size_schedule(args.batch_size)
    log(
        f"[Worker {worker_rank} | GPU {cuda_index}] model ready, samples={len(items)}, "
        f"start_round={start_round_index + 1}:{rounds[start_round_index]['name']}, "
        f"batch_schedule={batch_size_schedule}"
    )

    states = [normalize_state(item) for item in items]
    for state in states:
        state["_cumulative_input_refs"] = [dict(spec) for spec in base_input]
        for previous_round in rounds[:start_round_index]:
            state["_cumulative_input_refs"].extend([dict(spec) for spec in previous_round.get("extra_input", [])])

    active_states = states
    if args.skip_existing_final:
        active_states = [
            state for state in states
            if not maybe_skip_final(state, output_dir, final_image_round, final_image_key, final_text_key)
        ]
        log(
            f"[Worker {worker_rank} | GPU {cuda_index}] skip-existing-final enabled: "
            f"active={len(active_states)}, skipped={len(states) - len(active_states)}"
        )
    else:
        log(f"[Worker {worker_rank} | GPU {cuda_index}] active samples={len(active_states)}")

    progress = tqdm(
        total=len(states),
        desc=f"GPU{cuda_index}",
        position=worker_rank,
        leave=True,
    )
    progress.update(len(states) - len(active_states))

    persist_states(state_path, states)

    for round_index, round_cfg in enumerate(rounds[start_round_index:], start=start_round_index):
        backend = round_cfg["backend"]
        extra_input = round_cfg.get("extra_input", [])
        params = round_cfg.get("params", {})
        output_cfg = round_cfg.get("output", {})
        round_batch_size = get_round_batch_size(round_cfg, round_index, batch_size_schedule)

        if not active_states:
            log(f"[Worker {worker_rank} | GPU {cuda_index}] no active samples left, stop before round '{round_cfg['name']}'")
            break

        round_states = active_states
        skipped_this_round = 0
        if args.skip_existing_stage_output:
            round_states = []
            for state in active_states:
                if maybe_skip_round_output(state, round_cfg, output_dir):
                    state["_history"].append(
                        {"round": round_cfg["name"], "backend": backend, "skipped": True, "reason": "existing_output"}
                    )
                    state["_cumulative_input_refs"] = state["_cumulative_input_refs"] + [dict(spec) for spec in extra_input]
                    skipped_this_round += 1
                else:
                    round_states.append(state)

        log(
            f"[Worker {worker_rank} | GPU {cuda_index}] start round {round_index + 1}/{len(rounds)} "
            f"'{round_cfg['name']}' backend={backend} batch_size={round_batch_size} "
            f"run={len(round_states)} skipped={skipped_this_round}"
        )

        for state in round_states:
            state["_round_refs"] = state["_cumulative_input_refs"] + [dict(spec) for spec in extra_input]

        batches = chunked(round_states, round_batch_size)
        round_progress = tqdm(
            batches,
            desc=f"GPU{cuda_index}:{round_cfg['name']}",
            position=worker_rank,
            leave=False,
        )

        for batch_states in round_progress:
            refs = batch_states[0]["_round_refs"]
            if backend == "generate_image":
                generated_images = batch_generate_images_from_sequence(
                    inferencer=inferencer,
                    states=batch_states,
                    refs=refs,
                    params=params,
                    device=device,
                )
                output_key = output_cfg["image"]
                for state, image in zip(batch_states, generated_images):
                    image_path = get_output_image_path(
                        output_dir,
                        output_key,
                        get_output_image_filename(state, round_cfg),
                    )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                    state[output_key] = str(image_path)
                    state["_history"].append(
                        {"round": round_cfg["name"], "backend": backend, "image_key": output_key, "path": str(image_path)}
                    )
            elif backend == "generate_text":
                generated_texts = batch_generate_texts_from_sequence(
                    inferencer=inferencer,
                    states=batch_states,
                    refs=refs,
                    params=params,
                    device=device,
                )
                output_key = output_cfg["text"]
                for state, text in zip(batch_states, generated_texts):
                    state[output_key] = text
                    state["_history"].append(
                        {"round": round_cfg["name"], "backend": backend, "text_key": output_key}
                    )
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            for state in batch_states:
                state["_cumulative_input_refs"] = refs

        round_progress.close()
        for state in active_states:
            if "_round_refs" in state:
                del state["_round_refs"]
        persist_states(state_path, states)
        log(
            f"[Worker {worker_rank} | GPU {cuda_index}] finished round '{round_cfg['name']}', "
            f"state saved to {state_path}"
        )

    with result_path.open("w", encoding="utf-8") as fp:
        for state in states:
            fp.write(json.dumps(serialize_state(state), ensure_ascii=False) + "\n")
            fp.flush()
    merge_result = len(states) - progress.n
    if merge_result > 0:
        progress.update(merge_result)

    progress.close()
    log(f"[Worker {worker_rank} | GPU {cuda_index}] finished, results saved to {result_path}")


def validate_sample_fields(samples: List[Dict[str, Any]], base_input: Sequence[Dict[str, Any]]) -> None:
    for sample in samples:
        if "id" not in sample:
            raise ValueError(f"Each sample must contain an 'id' field: {sample}")
        for spec in base_input:
            if "key" in spec and spec["key"] not in sample:
                raise ValueError(f"Missing base_input key '{spec['key']}' in sample id={sample.get('id')}")


def main() -> None:
    args = parse_args()
    gpus = resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs available for inference.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[Main] output_dir={output_dir}")

    pipeline_config = yaml.safe_load(Path(args.pipeline_config).read_text())
    validate_pipeline_config(pipeline_config)
    validate_input_spec_list(pipeline_config["base_input"], "base_input")
    for round_cfg in pipeline_config["rounds"]:
        validate_input_spec_list(round_cfg.get("extra_input", []), f"round[{round_cfg['name']}].extra_input")
    resolve_start_round_index(pipeline_config["rounds"], args.start_round)
    parse_batch_size_schedule(args.batch_size)

    samples = json.loads(Path(args.input_json).read_text())
    if not isinstance(samples, list):
        raise ValueError("input json must contain a list")
    log(
        f"[Main] loaded pipeline config from {args.pipeline_config}: "
        f"rounds={[round_cfg['name'] for round_cfg in pipeline_config['rounds']]}"
    )
    log(f"[Main] loaded {len(samples)} samples from {args.input_json}")
    existing_state_source = find_existing_state_source(output_dir)
    existing_states = load_existing_states(output_dir)
    if existing_state_source is not None:
        log(f"[Main] restored existing state for {len(existing_states)} samples from {existing_state_source}")
    else:
        log("[Main] no existing state found")
    merged_samples = [merge_loaded_state(sample, existing_states.get(str(sample["id"]))) for sample in samples]
    validate_sample_fields(merged_samples, pipeline_config["base_input"])

    shards = shard_items(merged_samples, len(gpus))
    start_round_index = resolve_start_round_index(pipeline_config["rounds"], args.start_round)
    log(
        f"[Main] launching {len(gpus)} GPUs, start_round={start_round_index + 1}:"
        f"{pipeline_config['rounds'][start_round_index]['name']}, batch_size={args.batch_size}"
    )
    mp.set_start_method("spawn", force=True)
    processes = []
    for rank, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        proc = mp.Process(target=run_worker, args=(rank, gpu, shard, args, pipeline_config))
        proc.start()
        processes.append(proc)

    exit_code = 0
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            exit_code = proc.exitcode
    if exit_code != 0:
        raise SystemExit(exit_code)

    merge_state_files(output_dir)
    merged_path = merge_result_files(output_dir)
    print(f"Merged results written to {merged_path}")


if __name__ == "__main__":
    main()
