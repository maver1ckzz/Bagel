from pathlib import Path
import json
import math
import random

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Source annotations. The file should load into a dict-like object:
# data[patient_id]['zyzg'] -> list of length 9
SOURCE_DATA_PATH = Path("/hdd/wangty/new_task/LLaMA-Factory/task/dataset/process_data.json")

# Full sagittal image root: /.../{patient_id}.png
SAG_ROOT = Path("/mnt/nvme_share/wangty/img/sag_512")

# Local crop root: /.../{patient_id}/{segment}.png, where segment is one of:
# 3, 3-4, 4, 4-5, 5, 5-6, 6, 6-7, 7
CROP_ROOT = Path("/mnt/nvme_share/wangty/img/sag_crop_64")

# Train split file used by the earlier notebook.
ID_SPLIT_PATH = Path("/hdd/wangty/new_task/LLaMA-Factory/task/dataset/id/id_split.txt")

# Output paths.
OUTPUT_ROOT = Path("/hdd/wangty/diffuser_workdir/bagel_example/editing")
PARQUET_DIR = OUTPUT_ROOT / "sag_multi_cot_zyzg"
PARQUET_INFO_DIR = OUTPUT_ROOT / "parquet_info"
PARQUET_INFO_PATH = PARQUET_INFO_DIR / "sag_multi_cot_zyzg_nas.json"

SHARD_SIZE = 3000
ROW_GROUP_SIZE = 128
RANDOM_SEED = 42

# 9 labels: C3, C3-4, ..., C7.
LEVEL_SPECS = [
    {"segment": "3", "level": "C3", "region": "the C3 vertebral body and adjacent spinal canal"},
    {"segment": "3-4", "level": "C3-4", "region": "the C3-4 disc space and adjacent spinal canal"},
    {"segment": "4", "level": "C4", "region": "the C4 vertebral body and adjacent spinal canal"},
    {"segment": "4-5", "level": "C4-5", "region": "the C4-5 disc space and adjacent spinal canal"},
    {"segment": "5", "level": "C5", "region": "the C5 vertebral body and adjacent spinal canal"},
    {"segment": "5-6", "level": "C5-6", "region": "the C5-6 disc space and adjacent spinal canal"},
    {"segment": "6", "level": "C6", "region": "the C6 vertebral body and adjacent spinal canal"},
    {"segment": "6-7", "level": "C6-7", "region": "the C6-7 disc space and adjacent spinal canal"},
    {"segment": "7", "level": "C7", "region": "the C7 vertebral body and adjacent spinal canal"},
]

# The user description mentions labels 0, 1, 2 and also says moderate / severe.
# This mapping supports both the 3-class and 4-class cases. Adjust if your source differs.
ZYZG_LABEL_MAP = {
    0: "normal",
    1: "mild spinal canal stenosis",
    2: "moderate to severe spinal canal stenosis",
    # 3: "severe spinal canal stenosis",
}

PROMPT_TEMPLATES = [
    "Assess spinal canal stenosis at the {level} level on this sagittal cervical spine MRI.",
    "Please diagnose the degree of spinal canal stenosis at {level} from this sagittal cervical spine MRI.",
    "For the sagittal cervical spine MRI, determine the spinal canal stenosis grade at {level}.",
]

FOCUS_TEMPLATES = [
    "To assess spinal canal stenosis at {level}, I should first focus on {region}.",
    "To diagnose the canal status at {level}, the relevant area to inspect first is {region}.",
    "The next step is to zoom in on {region} so the {level} spinal canal can be evaluated clearly.",
]

DIAGNOSIS_TEMPLATES = [
    "Based on the focused region, the spinal canal at {level} is {label}.",
    "The local image indicates {label} at the {level} level.",
    "From the highlighted region, the {level} spinal canal shows {label}.",
]

random.seed(RANDOM_SEED)
def load_source_data(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r") as f:
            return json.load(f)
    if suffix == ".jsonl":
        records = {}
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = str(item.get("id", item.get("patient_id")))
                records[item_id] = item
        return records
    raise ValueError(f"Unsupported SOURCE_DATA_PATH suffix: {path.suffix}")


def image_to_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def load_train_ids() -> list[str]:
    with ID_SPLIT_PATH.open("r") as f:
        train_id, val_id, test_id = eval(f.read())
    print(f"train ids: {len(train_id)}, val ids: {len(val_id)}, test ids: {len(test_id)}")
    return [str(x) for x in train_id]


def normalize_patient_record(patient_id: str, data_dict: dict):
    if patient_id in data_dict:
        return data_dict[patient_id]
    if patient_id.isdigit() and int(patient_id) in data_dict:
        return data_dict[int(patient_id)]
    return None


def normalize_label(raw_label):
    if raw_label is None:
        return None
    if isinstance(raw_label, str) and raw_label.strip() == "":
        return None
    try:
        raw_label = int(raw_label)
    except (TypeError, ValueError):
        return None
    return ZYZG_LABEL_MAP.get(raw_label)


def build_segment_list(level: str, region: str, full_image_bytes: bytes, crop_image_bytes: bytes, label_text: str):
    prompt = random.choice(PROMPT_TEMPLATES).format(level=level)
    focus = random.choice(FOCUS_TEMPLATES).format(level=level, region=region)
    diagnosis = random.choice(DIAGNOSIS_TEMPLATES).format(level=level, label=label_text)
    return [
        {"type": "input_image", "content": full_image_bytes},
        {"type": "input_text", "content": prompt},
        {"type": "output_text", "content": focus},
        {"type": "output_image", "content": crop_image_bytes},
        {"type": "output_text", "content": diagnosis},
    ]


def iter_samples(patient_ids: list[str], source_data: dict):
    missing_records = []
    invalid_zyzg = []
    missing_original = []
    missing_crop = []
    skipped_labels = []

    for patient_id in tqdm(patient_ids):
        record = normalize_patient_record(patient_id, source_data)
        if record is None:
            missing_records.append(patient_id)
            continue

        zyzg = record.get("zyzg")
        if not isinstance(zyzg, (list, tuple)) or len(zyzg) != 9:
            invalid_zyzg.append((patient_id, zyzg))
            continue

        sag_path = SAG_ROOT / f"{patient_id}.png"
        if not sag_path.exists():
            missing_original.append(str(sag_path))
            continue

        patient_crop_dir = CROP_ROOT / patient_id
        if not patient_crop_dir.exists():
            missing_crop.append(str(patient_crop_dir))
            continue

        full_image_bytes = image_to_bytes(sag_path)
        for spec, raw_label in zip(LEVEL_SPECS, zyzg):
            label_text = normalize_label(raw_label)
            if label_text is None:
                skipped_labels.append((patient_id, spec["level"], raw_label))
                continue

            crop_path = patient_crop_dir / f"{spec['segment']}.png"
            if not crop_path.exists():
                missing_crop.append(str(crop_path))
                continue

            yield {
                "id": patient_id,
                "task": "zyzg",
                "level": spec["level"],
                "segment": spec["segment"],
                "label_text": label_text,
                "segment_list": build_segment_list(
                    level=spec["level"],
                    region=spec["region"],
                    full_image_bytes=full_image_bytes,
                    crop_image_bytes=image_to_bytes(crop_path),
                    label_text=label_text,
                ),
            }

    print(f"missing source records: {len(missing_records)}")
    print(f"invalid zyzg entries: {len(invalid_zyzg)}")
    print(f"missing original images: {len(missing_original)}")
    print(f"missing crop paths: {len(missing_crop)}")
    print(f"skipped unknown labels: {len(skipped_labels)}")
    if missing_records[:5]:
        print("first missing records:", missing_records[:5])
    if invalid_zyzg[:3]:
        print("first invalid zyzg:", invalid_zyzg[:3])
    if missing_crop[:5]:
        print("first missing crop paths:", missing_crop[:5])
    if skipped_labels[:5]:
        print("first skipped labels:", skipped_labels[:5])


source_data = load_source_data(SOURCE_DATA_PATH)
train_ids = load_train_ids()
samples = list(iter_samples(train_ids, source_data))
print(f"built samples: {len(samples)}")
assert samples, "No samples were built. Check SOURCE_DATA_PATH, SAG_ROOT and CROP_ROOT."

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_INFO_DIR.mkdir(parents=True, exist_ok=True)

parquet_info = {}
num_shards = math.ceil(len(samples) / SHARD_SIZE)
for shard_idx in tqdm(range(num_shards)):
    start = shard_idx * SHARD_SIZE
    end = min(start + SHARD_SIZE, len(samples))
    shard_samples = samples[start:end]
    parquet_path = PARQUET_DIR / f"datapart-{shard_idx:05d}.parquet"

    df = pd.DataFrame(shard_samples)
    df.to_parquet(parquet_path, index=False, row_group_size=ROW_GROUP_SIZE)

    parquet_file = pq.ParquetFile(parquet_path)
    parquet_info[str(parquet_path)] = {
        "num_row_groups": parquet_file.num_row_groups,
        "num_rows": parquet_file.metadata.num_rows,
    }
    print(f"wrote {parquet_path}: rows={len(df)}, row_groups={parquet_file.num_row_groups}")

with PARQUET_INFO_PATH.open("w") as f:
    json.dump(parquet_info, f, indent=2)

print("parquet dir:", PARQUET_DIR)
print("parquet info:", PARQUET_INFO_PATH)
print("dataset_info.py entry:")
print(
    "    'sag_multi_cot_zyzg': {\n"
    f"        'data_dir': '{PARQUET_DIR}',\n"
    f"        'num_files': {num_shards},\n"
    f"        'num_total_samples': {len(samples)},\n"
    f"        'parquet_info_path': '{PARQUET_INFO_PATH}',\n"
    "    },"
)
