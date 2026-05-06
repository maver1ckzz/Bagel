# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, UnifiedEditIterableDatasetV2
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'unified_edit_v2': UnifiedEditIterableDatasetV2,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/hdd/wangty/diffuser_workdir/bagel_example/t2i',
            'num_files': 1,
            'num_total_samples': 10,
        },
        't2i_full': {
            'data_dir': '/mnt/wangty/dataset/bagel/t2i_full_test',
            'num_files': 1,
            'num_total_samples': 1561,
        },
        'tra_full':{
            'data_dir': '/mnt/wangty/dataset/bagel/tra_full',
            'num_files': 4,
            'num_total_samples': 14049,
        }
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
		'sag_crop': {
            'data_dir': '/hdd/wangty/diffuser_workdir/bagel_example/editing/sag_crop_64',
            'num_files': 5,
            'num_total_samples': 14049,
            "parquet_info_path": '/hdd/wangty/diffuser_workdir/bagel_example/editing/parquet_info/sag_crop_64_nas.json', # information of the parquet files
		},
		
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'unified_edit_v2': {
        'multi_inf': {
            'data_dir': '/hdd/wangty/diffuser_workdir/bagel_example/editing/sag_multi_cot_zyzg_format_new',
            'num_files': 5,
            'num_total_samples': 14049,
            'parquet_info_path': '/hdd/wangty/diffuser_workdir/bagel_example/editing/parquet_info/sag_multi_cot_zyzg_format_new_nas.json',
        },
        'zjxxz_crop_format':{
            'data_dir': '/hdd/wangty/diffuser_workdir/bagel_example/editing/sag_multi_cot_zjxxz_format_new',
            'num_files': 5,
            'num_total_samples': 6244,
            'parquet_info_path': '/hdd/wangty/diffuser_workdir/bagel_example/editing/parquet_info/sag_multi_cot_zjxxz_format_new_nas.json',
        },
    }
}