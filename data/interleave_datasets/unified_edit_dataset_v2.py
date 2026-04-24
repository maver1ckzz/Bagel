# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
支持任意图文交错格式的数据集类。

segment_list 格式：
[
    {"type": "input_image", "content": <image_bytes>},
    {"type": "input_text", "content": "请诊断 C3-4 椎间盘"},
    {"type": "output_text", "content": "需要先裁剪C3-4区域"},
    {"type": "output_image", "content": <image_bytes>},   # 中间输出图：vae(带噪) + vae(clean) + vit
    {"type": "output_text", "content": "C3-4椎间盘轻度突出"},
    {"type": "last_image", "content": <image_bytes>},     # 最后输出图：vae(带噪)
]

支持的 type：
- input_image : vit + vae(clean), loss=0
- input_text  : text, loss=0
- output_text  : text, loss=1
- output_image : vae(带噪) + vae(clean) + vit, loss=1 for vae(带噪) only
- last_image   : vae(带噪) only, loss=1
"""

import io
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDatasetV2(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """
    支持任意图文交错格式的 Dataset。
    
    通过 segment_list 定义序列中每个 segment 的类型和内容，
    框架自动处理 loss 计算、attention 模式等细节。
    """
    
    # Segment type 到处理方式的映射
    SEGMENT_HANDLERS = {
        # input_image: vit + vae(clean), 都 loss=0
        "input_image": "_handle_input_image",
        # input_text: 文本 token, loss=0
        "input_text": "_handle_input_text",
        # output_text: 文本 token, loss=1
        "output_text": "_handle_output_text",
        # output_image: vae(带噪) + vae(clean) + vit, vae(带噪) loss=1
        "output_image": "_handle_output_image",
        # last_image: vae(带噪) only, loss=1
        "last_image": "_handle_last_image",
    }
    
    def parse_row(self, row):
        """解析 segment_list，构建训练数据"""
        segment_list = row["segment_list"]
        
        data = self._init_data()
        
        for segment in segment_list:
            seg_type = segment["type"]
            content = segment["content"]
            
            if seg_type not in self.SEGMENT_HANDLERS:
                raise ValueError(f"Unknown segment type: {seg_type}. "
                                f"Supported types: {list(self.SEGMENT_HANDLERS.keys())}")
            
            handler = getattr(self, self.SEGMENT_HANDLERS[seg_type])
            data = handler(data, content)
        
        return data
    
    def _handle_input_image(self, data, image_bytes):
        """处理 input_image: vit + vae(clean), 都 loss=0"""
        img = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        
        # 添加 vit 特征（输入图像理解）
        data = self._add_image(
            data,
            img,
            need_loss=False,
            need_vae=False,
            need_vit=True,
            enable_cfg=True,
        )
        
        # 添加 vae clean 特征
        data = self._add_image(
            data,
            img,
            need_loss=False,
            need_vae=True,
            need_vit=False,
            enable_cfg=True,
        )
        
        return data
    
    def _handle_input_text(self, data, text):
        """处理 input_text: 文本 token, loss=0"""
        data = self._add_text(data, text, need_loss=False, enable_cfg=True)
        return data
    
    def _handle_output_text(self, data, text):
        """处理 output_text: 文本 token, loss=1"""
        data = self._add_text(data, text, need_loss=True, enable_cfg=False)
        return data
    
    def _handle_output_image(self, data, image_bytes):
        """
        处理 output_image: 中间输出图
        vae(带噪) + vae(clean) + vit
        vae(带噪) loss=1, 其他 loss=0
        """
        img = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        
        # 1. vae(带噪) - 用于去噪学习，loss=1
        data = self._add_image(
            data,
            img,
            need_loss=True,    # 计算 MSE Loss
            need_vae=True,
            need_vit=False,
            enable_cfg=False,  # 生成任务，关闭 CFG
        )
        
        # 2. vae(clean) - 作为后续文本生成的条件信息
        data = self._add_image(
            data,
            img,
            need_loss=False,
            need_vae=True,
            need_vit=False,
            enable_cfg=False,
        )
        
        # 3. vit - 提供语义特征，作为后续文本生成的条件信息
        data = self._add_image(
            data,
            img,
            need_loss=False,
            need_vae=False,
            need_vit=True,
            enable_cfg=False,
        )
        
        return data
    
    def _handle_last_image(self, data, image_bytes):
        """
        处理 last_image: 最后输出图
        只有 vae(带噪)，loss=1
        后面没有其他内容，不需要 vae(clean) 和 vit
        """
        img = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
        
        data = self._add_image(
            data,
            img,
            need_loss=True,    # 计算 MSE Loss
            need_vae=False,    # 只需要带噪 VAE
            need_vit=False,
            enable_cfg=False,
        )
        
        return data