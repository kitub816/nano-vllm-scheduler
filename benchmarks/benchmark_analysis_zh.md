# Benchmark 可靠性复核

稳定版本仅归档 [精确工作负载与整理后的汇总](baselines/stage3/README.md)。
下文提到的完整 `results/`、`results_reliability/` 是本地原始证据目录，未上传 Git。

## 测量方法审查

复核保留原调度算法、ModelRunner 和 CUDA Kernel，仅扩展 Benchmark 的观察和导出。

所有请求时间戳、逐轮时间和正式耗时使用同一 worker 内同一个 perf_counter 单调时钟及同一 origin。不同进程的相对时间不直接相减；只汇总各自计算出的指标。GPU 温度/频率快照在正式窗口外采集，不用于请求延迟计算。

ModelRunner.run() 在返回前对采样 GPU Tensor 调用 .tolist()，取得主机可用的 Token 列表；LLMEngine.step() 随后调用 postprocess()。观察器在真实 append_token() 返回后记录一次 Token 时间，postprocess() 完成状态转换和释放后记录请求完成时间。未完成的 Prefill Chunk 不追加 Token，不记录虚假的 TTFT。

TTFT 从实际入队到首 Token；ITL 为同一请求相邻 Token 时间之差（跨请求池化间隔）；完整延迟从实际入队到 postprocess 完成。吞吐量为真实追加的总输出 Token 数 / 暖机后正式耗时，包含计划到达间隙。计划到达只是偏移；接纳延迟另行报告，未混入以入队为起点的 TTFT。

逐轮 trace 将每个 (request_id, output_token_index) 与原始 Token 时间一一核对，验证不重计、不遗漏；Decode 批大小等于该轮实际追加 Token 数。P50/P95/P99 均沿用 (n−1)×q 的 type 7 线性插值。轮次耗时是 engine.step() 的主机总耗时，包含调度、GPU 计算/结果传输、后处理与轻量插桩，不是 CUDA Kernel 耗时。

## 既有单次数据的 ITL 长尾

来源：`results/raw_requests.csv`，原文件未覆盖。每种策略 480 个 ITL 样本。

| 策略 | 均值 ms | P50 ms | P95 ms | P99 ms | Max ms | >100ms | >200ms | >100ms 占总 ITL 时间 | 其余均值 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill_first | 71.167 | 30.974 | 43.663 | 1939.537 | 3024.773 | 12 | 12 | 55.88% | 32.204 |
| decode_first | 30.135 | 28.136 | 41.831 | 53.761 | 60.120 | 0 | 0 | 0.00% | 30.135 |
| decode_first_fair2 | 62.104 | 31.354 | 62.158 | 1441.126 | 2281.536 | 8 | 8 | 44.16% | 35.264 |

Prefill-first 超过 100ms 的 12 个间隔只占 2.5%，但占间隔总时长的 55.88%；均发生在第 1→2 个输出 Token 之间。均值因此可以高于 P95，现有证据不支持把该大小关系本身判定为指标 Bug。它反映请求先得到首 Token、随后长时间没有获得 Decode 服务。

### 三策略合并后的最长 10 个 ITL

| 策略 | request_id | 输出 Token 序号 | ITL ms |
| --- | --- | --- | --- |
| prefill_first | 11 | 1→2 | 3024.773 |
| prefill_first | 10 | 1→2 | 2928.136 |
| prefill_first | 9 | 1→2 | 2486.492 |
| decode_first_fair2 | 10 | 1→2 | 2281.536 |
| decode_first_fair2 | 11 | 1→2 | 2237.734 |
| prefill_first | 7 | 1→2 | 2049.693 |
| prefill_first | 8 | 1→2 | 2049.689 |
| decode_first_fair2 | 9 | 1→2 | 1922.359 |
| prefill_first | 6 | 1→2 | 1910.256 |
| prefill_first | 5 | 1→2 | 1798.008 |

### prefill_first：最长 10 个 ITL

| request_id | 输出 Token 序号 | ITL ms | 起点 s | 终点 s |
| --- | --- | --- | --- | --- |
| 11 | 1→2 | 3024.773 | 0.467643 | 3.492415 |
| 10 | 1→2 | 2928.136 | 0.398126 | 3.326261 |
| 9 | 1→2 | 2486.492 | 0.330192 | 2.816684 |
| 7 | 1→2 | 2049.693 | 0.330182 | 2.379875 |
| 8 | 1→2 | 2049.689 | 0.330189 | 2.379878 |
| 6 | 1→2 | 1910.256 | 0.264559 | 2.174815 |
| 5 | 1→2 | 1798.008 | 0.200043 | 1.998050 |
| 4 | 1→2 | 1054.797 | 0.133400 | 1.188196 |
| 0 | 1→2 | 462.760 | 0.034789 | 0.497549 |
| 1 | 1→2 | 462.757 | 0.034800 | 0.497557 |

### decode_first：最长 10 个 ITL

| request_id | 输出 Token 序号 | ITL ms | 起点 s | 终点 s |
| --- | --- | --- | --- | --- |
| 1 | 3→4 | 60.120 | 0.107905 | 0.168026 |
| 0 | 3→4 | 60.119 | 0.107894 | 0.168012 |
| 1 | 46→47 | 55.757 | 1.614494 | 1.670252 |
| 0 | 46→47 | 55.757 | 1.614483 | 1.670240 |
| 1 | 44→45 | 53.762 | 1.524399 | 1.578161 |
| 0 | 44→45 | 53.761 | 1.524390 | 1.578151 |
| 0 | 4→5 | 51.303 | 0.168012 | 0.219315 |
| 1 | 4→5 | 51.298 | 0.168026 | 0.219324 |
| 3 | 6→7 | 51.148 | 2.226501 | 2.277649 |
| 2 | 6→7 | 51.146 | 2.226493 | 2.277639 |

### decode_first_fair2：最长 10 个 ITL

| request_id | 输出 Token 序号 | ITL ms | 起点 s | 终点 s |
| --- | --- | --- | --- | --- |
| 10 | 1→2 | 2281.536 | 1.035296 | 3.316833 |
| 11 | 1→2 | 2237.734 | 1.247541 | 3.485276 |
| 9 | 1→2 | 1922.359 | 0.829817 | 2.752176 |
| 5 | 1→2 | 1501.919 | 0.466407 | 1.968326 |
| 8 | 1→2 | 1489.385 | 0.829814 | 2.319199 |
| 7 | 1→2 | 1428.298 | 0.829803 | 2.258101 |
| 6 | 1→2 | 1410.019 | 0.648150 | 2.058169 |
| 4 | 1→2 | 894.010 | 0.293323 | 1.187333 |
| 3 | 21→22 | 92.428 | 0.993183 | 1.085611 |
| 2 | 21→22 | 92.426 | 0.993181 | 1.085607 |

## 重复实验设计与逐次结果

相同精确 Token ID、输出长度和计划到达，SHA-256 与原 workload 一致。引擎配置和暖机函数保持一致，每次重新创建独立 worker、模型实例和缓存。三次重复按 A/B/C、B/C/A、C/A/B 轮换顺序；同一 repeat 内采样种子相同（seed+repeat），不保证不同调度下输出内容逐 Token 相同。

| 执行序号 | repeat | 策略 | TTFT 均值 ms | ITL 均值 ms | 输出 tok/s | Decode 轮数 | 平均批大小 | 暖机后/运行后温度 °C | SM 频率前/后 MHz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | prefill_first | 149.419 | 63.207 | 107.937 | 146 | 3.287671 | 46/46 | 1530/1635 |
| 1 | 0 | decode_first | 4336.897 | 27.611 | 48.513 | 356 | 1.348315 | 47/47 | 1890/1560 |
| 2 | 0 | decode_first_fair2 | 501.782 | 58.052 | 100.645 | 148 | 3.243243 | 48/48 | 1890/1620 |
| 3 | 1 | decode_first | 4355.124 | 28.242 | 47.645 | 356 | 1.348315 | 48/48 | 1770/1380 |
| 4 | 1 | decode_first_fair2 | 463.914 | 56.682 | 106.326 | 148 | 3.243243 | 49/48 | 1890/1755 |
| 5 | 1 | prefill_first | 140.334 | 68.234 | 101.012 | 146 | 3.287671 | 49/48 | 1890/1650 |
| 6 | 2 | decode_first_fair2 | 465.915 | 55.755 | 106.846 | 148 | 3.243243 | 49/48 | 1890/1620 |
| 7 | 2 | prefill_first | 151.927 | 65.850 | 100.800 | 146 | 3.287671 | 49/48 | 1890/1545 |
| 8 | 2 | decode_first | 4243.607 | 27.554 | 48.581 | 356 | 1.348315 | 49/48 | 1890/1455 |

前后快照的 GPU 温度范围为 46–49 °C。

前后快照的 SM 频率范围为 1380–1890 MHz。

顺序已轮换，但频率快照存在变化，因此不能宣称硬件时钟保持恒定；三个样本也不足以把轮次耗时变化归因于温度、时钟或调度中的某一个因素。

## Decode 批大小与轮次耗时

原单次实验没有逐轮记录，不能从旧 raw_requests.csv 准确恢复当时的批大小分布或耗时；下表来自本次真实 engine.step() 追踪。

| 策略 | repeat | batch size | 轮数 | Decode 轮数占比 | 平均 engine.step 耗时 ms |
| --- | --- | --- | --- | --- | --- |
| prefill_first | 0 | 1 | 33 | 22.60% | 26.239 |
| prefill_first | 0 | 2 | 1 | 0.68% | 28.112 |
| prefill_first | 0 | 3 | 3 | 2.05% | 27.963 |
| prefill_first | 0 | 4 | 109 | 74.66% | 28.626 |
| decode_first | 0 | 1 | 253 | 71.07% | 26.368 |
| decode_first | 0 | 2 | 82 | 23.03% | 28.973 |
| decode_first | 0 | 3 | 21 | 5.90% | 29.016 |
| decode_first_fair2 | 0 | 1 | 33 | 22.30% | 30.189 |
| decode_first_fair2 | 0 | 2 | 5 | 3.38% | 30.355 |
| decode_first_fair2 | 0 | 3 | 3 | 2.03% | 27.501 |
| decode_first_fair2 | 0 | 4 | 107 | 72.30% | 29.751 |
| decode_first | 1 | 1 | 253 | 71.07% | 26.917 |
| decode_first | 1 | 2 | 82 | 23.03% | 29.928 |
| decode_first | 1 | 3 | 21 | 5.90% | 29.124 |
| decode_first_fair2 | 1 | 1 | 33 | 22.30% | 25.863 |
| decode_first_fair2 | 1 | 2 | 5 | 3.38% | 27.548 |
| decode_first_fair2 | 1 | 3 | 3 | 2.03% | 29.628 |
| decode_first_fair2 | 1 | 4 | 107 | 72.30% | 28.903 |
| prefill_first | 1 | 1 | 33 | 22.60% | 27.647 |
| prefill_first | 1 | 2 | 1 | 0.68% | 28.560 |
| prefill_first | 1 | 3 | 3 | 2.05% | 28.411 |
| prefill_first | 1 | 4 | 109 | 74.66% | 31.125 |
| decode_first_fair2 | 2 | 1 | 33 | 22.30% | 26.189 |
| decode_first_fair2 | 2 | 2 | 5 | 3.38% | 28.740 |
| decode_first_fair2 | 2 | 3 | 3 | 2.03% | 29.452 |
| decode_first_fair2 | 2 | 4 | 107 | 72.30% | 28.640 |
| prefill_first | 2 | 1 | 33 | 22.60% | 30.401 |
| prefill_first | 2 | 2 | 1 | 0.68% | 30.456 |
| prefill_first | 2 | 3 | 3 | 2.05% | 33.034 |
| prefill_first | 2 | 4 | 109 | 74.66% | 30.163 |
| decode_first | 2 | 1 | 253 | 71.07% | 26.640 |
| decode_first | 2 | 2 | 82 | 23.03% | 28.180 |
| decode_first | 2 | 3 | 21 | 5.90% | 29.546 |

每个请求的首 Token 在 Prefill 完成时追加，因此 492 输出 Token 中有 12 个来自 Prefill，其余 480 个来自 Decode。对每次运行都核对 Σ(batch_size×轮数)=480。平均 Decode 批大小为 480/Decode轮数，不能把 Decode 轮数当作 Token 数。

decode_first 在现有 running 可 Decode 时持续 Decode，不接纳 waiting 做 Prefill，因而较少请求同时参与 Decode；prefill_first 提前让更多请求进入 running，fair2 周期性插入 Prefill，后两者更容易构成大批次。这解释了同为 480 个 Decode Token 却需要不同轮数。max_num_seqs=4 是单批上限，running 可以超过 4；_schedule_decode() 将选中的请求放回 running 队首，已有队首请求可能长期占据批次，造成后面请求首 Token 后的长间隔。这里是对现有行为的测量与解释，未修改该行为。

### 新追踪中的长间隔证据（每策略 repeat 0 的最坏间隔）

| 策略 | request_id | Token 序号 | ITL ms | 两端 step | 中间 Prefill/Decode 轮数 | 中间未调度该请求的轮数 |
| --- | --- | --- | --- | --- | --- | --- |
| prefill_first | 11 | 1→2 | 2670.508 | 13→106 | 0/92 | 92 |
| decode_first | 3 | 6→7 | 45.462 | 63→64 | 0/0 | 0 |
| decode_first_fair2 | 10 | 1→2 | 2109.313 | 33→103 | 2/67 | 69 |

## 各指标跨运行统计

下表对每个运行先计算指标，再对三次运行求均值、样本标准差（ddof=1）、最小值、最大值；不把三次原始样本混起来算分位数。时间指标在表中换算为 ms，CSV/JSON 保持秒。每项 n=3，不报告显著性或普遍加速结论。

| 策略 | 指标 | n | 均值 | 样本标准差 | 最小值 | 最大值 |
| --- | --- | --- | --- | --- | --- | --- |
| prefill_first | num_kvcache_blocks | 3 | 105.000000 | 0.000000 | 105.000000 | 105.000000 |
| prefill_first | prefill_steps | 3 | 14.000000 | 0.000000 | 14.000000 | 14.000000 |
| prefill_first | decode_steps | 3 | 146.000000 | 0.000000 | 146.000000 | 146.000000 |
| prefill_first | preemptions | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| prefill_first | requested_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | admitted_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | completed_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | total_output_tokens | 3 | 492.000000 | 0.000000 | 492.000000 | 492.000000 |
| prefill_first | elapsed_seconds | 3 | 4.769966 | 0.183448 | 4.558222 | 4.880954 |
| prefill_first | output_token_throughput | 3 | 103.249495 | 4.060734 | 100.799967 | 107.936833 |
| prefill_first | ttft_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | ttft_mean_s (ms) | 3 | 147.226690 | 6.099491 | 140.334120 | 151.927226 |
| prefill_first | ttft_p50_s (ms) | 3 | 160.228338 | 4.553444 | 156.889590 | 165.415300 |
| prefill_first | ttft_p95_s (ms) | 3 | 252.336403 | 7.688481 | 243.661165 | 258.307400 |
| prefill_first | itl_samples | 3 | 480.000000 | 0.000000 | 480.000000 | 480.000000 |
| prefill_first | itl_mean_s (ms) | 3 | 65.763932 | 2.514660 | 63.207240 | 68.234339 |
| prefill_first | itl_p50_s (ms) | 3 | 29.120535 | 1.038571 | 28.286010 | 30.283656 |
| prefill_first | itl_p95_s (ms) | 3 | 36.004480 | 6.348069 | 31.382632 | 43.242560 |
| prefill_first | request_latency_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | request_latency_mean_s (ms) | 3 | 2777.802093 | 96.258511 | 2677.726297 | 2869.724681 |
| prefill_first | request_latency_p50_s (ms) | 3 | 2722.675117 | 85.226531 | 2632.545172 | 2801.960060 |
| prefill_first | request_latency_p95_s (ms) | 3 | 4115.137548 | 158.598286 | 3934.262527 | 4230.406214 |
| prefill_first | admission_delay_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | admission_delay_mean_s (ms) | 3 | 11.063287 | 0.451149 | 10.765586 | 11.582362 |
| prefill_first | admission_delay_p50_s (ms) | 3 | 9.179519 | 0.887175 | 8.598607 | 10.200718 |
| prefill_first | admission_delay_p95_s (ms) | 3 | 28.436437 | 1.654751 | 26.698816 | 29.993512 |
| prefill_first | torch_compile_frames_during_measurement | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| prefill_first | itl_p99_s (ms) | 3 | 1764.834789 | 82.684388 | 1706.576864 | 1859.471146 |
| prefill_first | itl_max_s (ms) | 3 | 2782.104912 | 108.208054 | 2670.507986 | 2886.571409 |
| prefill_first | itl_over_100ms | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | itl_over_200ms | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| prefill_first | decode_mean_batch_size | 3 | 3.287671 | 0.000000 | 3.287671 | 3.287671 |
| prefill_first | decode_step_mean_s (ms) | 3 | 29.537340 | 1.271508 | 28.069146 | 30.277510 |
| prefill_first | decode_batch_1_rounds | 3 | 33.000000 | 0.000000 | 33.000000 | 33.000000 |
| prefill_first | decode_batch_1_step_mean_s (ms) | 3 | 28.096040 | 2.116992 | 26.239266 | 30.401362 |
| prefill_first | decode_batch_2_rounds | 3 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| prefill_first | decode_batch_2_step_mean_s (ms) | 3 | 29.042692 | 1.243861 | 28.112353 | 30.455512 |
| prefill_first | decode_batch_3_rounds | 3 | 3.000000 | 0.000000 | 3.000000 | 3.000000 |
| prefill_first | decode_batch_3_step_mean_s (ms) | 3 | 29.802615 | 2.807116 | 27.962727 | 33.033622 |
| prefill_first | decode_batch_4_rounds | 3 | 109.000000 | 0.000000 | 109.000000 | 109.000000 |
| prefill_first | decode_batch_4_step_mean_s (ms) | 3 | 29.970933 | 1.260428 | 28.625679 | 31.124597 |
| decode_first | num_kvcache_blocks | 3 | 105.000000 | 0.000000 | 105.000000 | 105.000000 |
| decode_first | prefill_steps | 3 | 14.000000 | 0.000000 | 14.000000 | 14.000000 |
| decode_first | decode_steps | 3 | 356.000000 | 0.000000 | 356.000000 | 356.000000 |
| decode_first | preemptions | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first | requested_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first | admitted_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first | completed_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first | total_output_tokens | 3 | 492.000000 | 0.000000 | 492.000000 | 492.000000 |
| decode_first | elapsed_seconds | 3 | 10.198512 | 0.110967 | 10.127485 | 10.326384 |
| decode_first | output_token_throughput | 3 | 48.246115 | 0.521733 | 47.644947 | 48.580669 |
| decode_first | ttft_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first | ttft_mean_s (ms) | 3 | 4311.875942 | 59.821062 | 4243.606878 | 4355.124235 |
| decode_first | ttft_p50_s (ms) | 3 | 4348.615928 | 96.999871 | 4237.036235 | 4412.858830 |
| decode_first | ttft_p95_s (ms) | 3 | 8589.157484 | 75.674304 | 8509.818511 | 8660.538101 |
| decode_first | itl_samples | 3 | 480.000000 | 0.000000 | 480.000000 | 480.000000 |
| decode_first | itl_mean_s (ms) | 3 | 27.802143 | 0.381610 | 27.553555 | 28.241524 |
| decode_first | itl_p50_s (ms) | 3 | 27.485360 | 0.440419 | 27.201314 | 27.992701 |
| decode_first | itl_p95_s (ms) | 3 | 32.832761 | 1.010940 | 32.248148 | 34.000093 |
| decode_first | request_latency_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first | request_latency_mean_s (ms) | 3 | 5423.977817 | 71.130982 | 5345.764435 | 5484.801248 |
| decode_first | request_latency_p50_s (ms) | 3 | 5475.770718 | 86.349922 | 5376.253277 | 5530.869673 |
| decode_first | request_latency_p95_s (ms) | 3 | 9696.426306 | 98.126958 | 9621.164322 | 9807.410043 |
| decode_first | admission_delay_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first | admission_delay_mean_s (ms) | 3 | 8.483074 | 0.475077 | 7.997106 | 8.946454 |
| decode_first | admission_delay_p50_s (ms) | 3 | 6.673139 | 1.041705 | 5.547316 | 7.602850 |
| decode_first | admission_delay_p95_s (ms) | 3 | 22.736180 | 1.058468 | 21.832511 | 23.900679 |
| decode_first | torch_compile_frames_during_measurement | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first | itl_p99_s (ms) | 3 | 40.621462 | 3.249199 | 36.931474 | 43.054065 |
| decode_first | itl_max_s (ms) | 3 | 46.371264 | 5.868789 | 41.010259 | 52.641670 |
| decode_first | itl_over_100ms | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first | itl_over_200ms | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first | decode_mean_batch_size | 3 | 1.348315 | 0.000000 | 1.348315 | 1.348315 |
| decode_first | decode_step_mean_s (ms) | 3 | 27.343611 | 0.344444 | 27.124028 | 27.740595 |
| decode_first | decode_batch_1_rounds | 3 | 253.000000 | 0.000000 | 253.000000 | 253.000000 |
| decode_first | decode_batch_1_step_mean_s (ms) | 3 | 26.641558 | 0.274560 | 26.367670 | 26.916785 |
| decode_first | decode_batch_2_rounds | 3 | 82.000000 | 0.000000 | 82.000000 | 82.000000 |
| decode_first | decode_batch_2_step_mean_s (ms) | 3 | 29.026960 | 0.875545 | 28.179536 | 29.928148 |
| decode_first | decode_batch_3_rounds | 3 | 21.000000 | 0.000000 | 21.000000 | 21.000000 |
| decode_first | decode_batch_3_step_mean_s (ms) | 3 | 29.228607 | 0.280417 | 29.015790 | 29.546357 |
| decode_first | decode_batch_4_rounds | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first_fair2 | num_kvcache_blocks | 3 | 105.000000 | 0.000000 | 105.000000 | 105.000000 |
| decode_first_fair2 | prefill_steps | 3 | 14.000000 | 0.000000 | 14.000000 | 14.000000 |
| decode_first_fair2 | decode_steps | 3 | 148.000000 | 0.000000 | 148.000000 | 148.000000 |
| decode_first_fair2 | preemptions | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first_fair2 | requested_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first_fair2 | admitted_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first_fair2 | completed_requests | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first_fair2 | total_output_tokens | 3 | 492.000000 | 0.000000 | 492.000000 | 492.000000 |
| decode_first_fair2 | elapsed_seconds | 3 | 4.706843 | 0.157689 | 4.604772 | 4.888461 |
| decode_first_fair2 | output_token_throughput | 3 | 104.605477 | 3.439575 | 100.645166 | 106.845683 |
| decode_first_fair2 | ttft_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first_fair2 | ttft_mean_s (ms) | 3 | 477.203505 | 21.308701 | 463.914021 | 501.781538 |
| decode_first_fair2 | ttft_p50_s (ms) | 3 | 514.837778 | 23.503281 | 497.122320 | 541.500735 |
| decode_first_fair2 | ttft_p95_s (ms) | 3 | 939.660205 | 35.648329 | 916.192976 | 980.681535 |
| decode_first_fair2 | itl_samples | 3 | 480.000000 | 0.000000 | 480.000000 | 480.000000 |
| decode_first_fair2 | itl_mean_s (ms) | 3 | 56.829697 | 1.155507 | 55.755162 | 58.051966 |
| decode_first_fair2 | itl_p50_s (ms) | 3 | 28.928764 | 0.542224 | 28.581258 | 29.553556 |
| decode_first_fair2 | itl_p95_s (ms) | 3 | 61.248537 | 2.540380 | 59.726426 | 64.181208 |
| decode_first_fair2 | request_latency_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first_fair2 | request_latency_mean_s (ms) | 3 | 2750.411696 | 65.999251 | 2696.140938 | 2823.881887 |
| decode_first_fair2 | request_latency_p50_s (ms) | 3 | 2725.383415 | 68.470918 | 2667.015166 | 2800.753330 |
| decode_first_fair2 | request_latency_p95_s (ms) | 3 | 4046.585369 | 116.694618 | 3963.875784 | 4180.064954 |
| decode_first_fair2 | admission_delay_samples | 3 | 12.000000 | 0.000000 | 12.000000 | 12.000000 |
| decode_first_fair2 | admission_delay_mean_s (ms) | 3 | 8.891580 | 0.428196 | 8.420048 | 9.256168 |
| decode_first_fair2 | admission_delay_p50_s (ms) | 3 | 6.114320 | 0.993058 | 5.082748 | 7.063762 |
| decode_first_fair2 | admission_delay_p95_s (ms) | 3 | 23.309654 | 1.199567 | 22.287566 | 24.630304 |
| decode_first_fair2 | torch_compile_frames_during_measurement | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decode_first_fair2 | itl_p99_s (ms) | 3 | 1307.170771 | 26.864286 | 1279.174567 | 1332.738197 |
| decode_first_fair2 | itl_max_s (ms) | 3 | 2064.349197 | 41.191075 | 2028.435106 | 2109.312505 |
| decode_first_fair2 | itl_over_100ms | 3 | 8.000000 | 0.000000 | 8.000000 | 8.000000 |
| decode_first_fair2 | itl_over_200ms | 3 | 8.000000 | 0.000000 | 8.000000 | 8.000000 |
| decode_first_fair2 | decode_mean_batch_size | 3 | 3.243243 | 0.000000 | 3.243243 | 3.243243 |
| decode_first_fair2 | decode_step_mean_s (ms) | 3 | 28.710196 | 0.964801 | 28.113356 | 29.823280 |
| decode_first_fair2 | decode_batch_1_rounds | 3 | 33.000000 | 0.000000 | 33.000000 | 33.000000 |
| decode_first_fair2 | decode_batch_1_step_mean_s (ms) | 3 | 27.413793 | 2.408635 | 25.863262 | 30.188661 |
| decode_first_fair2 | decode_batch_2_rounds | 3 | 5.000000 | 0.000000 | 5.000000 | 5.000000 |
| decode_first_fair2 | decode_batch_2_step_mean_s (ms) | 3 | 28.881001 | 1.408983 | 27.547549 | 30.354984 |
| decode_first_fair2 | decode_batch_3_rounds | 3 | 3.000000 | 0.000000 | 3.000000 | 3.000000 |
| decode_first_fair2 | decode_batch_3_step_mean_s (ms) | 3 | 28.860291 | 1.180687 | 27.500734 | 29.627953 |
| decode_first_fair2 | decode_batch_4_rounds | 3 | 107.000000 | 0.000000 | 107.000000 | 107.000000 |
| decode_first_fair2 | decode_batch_4_step_mean_s (ms) | 3 | 29.097831 | 0.580619 | 28.639866 | 29.750864 |

Prefill-first 三次吞吐量范围为 100.800–107.937 tok/s；fair2 为 100.645–106.846 tok/s。两者区间重叠，不能根据各自均值给出稳定的快慢排序。

## 完成数、Token 与资源核验

| 策略 | repeat | 完成/请求 | 输出 Token | ITL 样本 | Prefill/Decode | 抢占 | 空闲/总 KV Block | 引用计数和 | 计时编译帧 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill_first | 0 | 12/12 | 492 | 480 | 14/146 | 0 | 105/105 | 0 | 0 |
| decode_first | 0 | 12/12 | 492 | 480 | 14/356 | 0 | 105/105 | 0 | 0 |
| decode_first_fair2 | 0 | 12/12 | 492 | 480 | 14/148 | 0 | 105/105 | 0 | 0 |
| decode_first | 1 | 12/12 | 492 | 480 | 14/356 | 0 | 105/105 | 0 | 0 |
| decode_first_fair2 | 1 | 12/12 | 492 | 480 | 14/148 | 0 | 105/105 | 0 | 0 |
| prefill_first | 1 | 12/12 | 492 | 480 | 14/146 | 0 | 105/105 | 0 | 0 |
| decode_first_fair2 | 2 | 12/12 | 492 | 480 | 14/148 | 0 | 105/105 | 0 | 0 |
| prefill_first | 2 | 12/12 | 492 | 480 | 14/146 | 0 | 105/105 | 0 | 0 |
| decode_first | 2 | 12/12 | 492 | 480 | 14/356 | 0 | 105/105 | 0 | 0 |

## 文件、复现命令与限制

原始实验保留在 `results`。重复实验的汇总和逐请求/逐轮 CSV 位于 `results_reliability`；每个 `runs/<policy>_repeatN/` 还有独立 summary.json、summary.csv、raw_requests.csv、raw_steps.csv，旁边的 JSON 保存原始请求、逐轮记录、GPU/软件元数据和资源快照。

```bash
.venv/bin/python -m benchmarks.mixed_load_bench --workload-from results/workload.json --repeats 3 --order-mode rotate --output-dir results_reliability
.venv/bin/python -m benchmarks.reliability --baseline results --repeats results_reliability --report benchmarks/benchmark_analysis_zh.md
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

限制：单台笔记本 GPU、12 请求、一个混合工作负载、每策略只有三次重复；顺序轮换只能平衡位置，不能隔离后台负载、GPU 温度/频率或功耗变化。GPU 遥测只有窗口前后快照，不是连续频率轨迹，不能据此断言漂移已消除或把变化归因于某个因素。同步单线程仅在 step 间隙注入请求，存在可测的接纳延迟；没有模拟异步生产服务、网络或客户端时延。轻量追踪也有 CPU 开销，旧单次与新增追踪重复不是完全相同的仪器条件。默认 ignore_eos 固定长度、原始文本截取，不代表所有真实聊天负载；没有制造 KV 高压，零抢占不能验证高压性能。

本复核确认测量定义及记录的内部一致性，区分了长尾 ITL、批大小与轮数；不宣称某种调度算法普遍更快。

## 验证记录

全部 49 项单元测试通过（保留原有 42 项，新增 7 项可靠性测试）。9 次 GPU 运行的请求 CSV、逐轮 CSV、JSON 汇总、Token/轮次对应关系、输出长度和 KV 资源快照均已校验。原始 results/ 的 CSV/JSON 校验仍通过；git diff --check 与新增文件空白检查通过。未新增依赖，未修改 Scheduler、ModelRunner 或 CUDA Kernel，未执行 Git Commit。
