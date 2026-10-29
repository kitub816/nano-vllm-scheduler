# 第三阶段稳定基线

这里只保存整理后的、小规模可审查材料，不包含逐 Token/逐轮原始追踪和运行日志。

- `workload.json`：12 个请求的精确 Prompt Token ID、目标输出长度、计划到达和配置。
- `summary.csv`：三种策略各三次实测的汇总。
- `cross_run_statistics.csv`：跨运行均值、样本标准差、最小值和最大值。
- `decode_batch_statistics.csv`：各次实测的 Decode 批大小分布与主机轮次耗时。
- `baseline_global_top10.csv`：最初单次实验的全局最长 10 个 ITL。
- `provenance.json`：归档来源与各文件 SHA-256。

工作负载 SHA-256：`525cf90c6681ab54cbf59499062a7a3d4bfcb796a1e421333e4a899669c9a3f1`。
原始实验基于 commit `bc6eba8`，GPU 为 RTX 4060 Laptop 8GB，本地 Qwen3-0.6B。
详细方法、长尾解释和限制见 [中文分析报告](../../benchmark_analysis_zh.md)。

本地完整证据保留在被 Git 忽略的 `results/` 和 `results_reliability/`；它们没有
随本稳定版本上传。汇总值不可替代独立复现实验，也不能用来重新构造逐 Token 时间戳。

在原环境复现：

```bash
.venv/bin/python -m benchmarks.mixed_load_bench \
  --workload-from benchmarks/baselines/stage3/workload.json \
  --repeats 3 --order-mode rotate --output-dir results_reproduce
```

manifest 的模型路径是原机器的本地路径。其他机器需要准备同版本模型并调整本地路径，
保持 Token ID、输出长度、到达时间及引擎参数一致，不下载或提交模型权重。
