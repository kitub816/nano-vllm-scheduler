
import os
from pathlib import Path

from nanovllm import LLM, SamplingParams

policy = os.getenv("POLICY", "prefill_first")

model_root = Path.home() / (
    ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
)
model_path = next(model_root.iterdir())

max_decode_steps = int(
    os.getenv("MAX_DECODE_STEPS", "0")
)

llm = LLM(
    str(model_path),
    scheduling_policy=policy,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=512,
    max_num_batched_tokens=256,
    max_num_seqs=1,
    gpu_memory_utilization=0.65,
    max_consecutive_decode_steps=max_decode_steps,
)

# 保存原始调度函数
original_schedule = llm.scheduler.schedule

# 包装调度函数，输出每一轮调度结果
def traced_schedule():
    seqs, is_prefill = original_schedule()

    if seqs:
        phase = "PREFILL" if is_prefill else "DECODE"
        ids = [seq.seq_id for seq in seqs]
        print(f"{phase}: {ids}")

    return seqs, is_prefill

llm.scheduler.schedule = traced_schedule

params = SamplingParams(
    temperature=0.6,
    max_tokens=4,
    ignore_eos=True,
)

outputs = llm.generate(
    [
        "请解释 KV Cache。",
        "请解释 PagedAttention。",
    ],
    params,
    use_tqdm=False,
)

print(f"\nPolicy: {policy}")
for i, output in enumerate(outputs):
    print(f"Request {i}: {output['text']}")
