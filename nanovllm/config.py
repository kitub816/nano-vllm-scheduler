import os
from dataclasses import dataclass
from transformers import AutoConfig

import torch

@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "prefill_first"

    def __post_init__(self):
        if self.scheduling_policy not in ("prefill_first", "decode_first"):
            raise ValueError(f"Unsupported scheduling policy: {self.scheduling_policy!r}")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 兼容新旧 Transformers 的 dtype 字段
        dtype = getattr(self.hf_config, "dtype", None)               #读取 dtype，不存在就返回 None，不会抛出 AttributeError。

        if dtype is None:
            dtype = getattr(self.hf_config, "torch_dtype", None)     #如果新版字段没有值，就尝试读取旧版的 torch_dtype

        if isinstance(dtype, str):                                   #判断是否为 "bfloat16" 这样的字符串
            dtype = getattr(torch, dtype, None)                      #把字符串转换成真正的 torch.bfloat16

        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported model dtype: {dtype!r}")

        self.hf_config.dtype = dtype                                 #统一配置，后续 ModelRunner 就能正常读取
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
