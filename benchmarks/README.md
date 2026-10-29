# 长短请求混合负载 Benchmark

第三阶段整理后的精确工作负载与汇总表保存在
[baselines/stage3](baselines/stage3/README.md)，分析见
[benchmark_analysis_zh.md](benchmark_analysis_zh.md)。本地 `results/`、`results_*/`
保留完整原始追踪，但不进入 Git；不要用 `git add -f` 上传这些目录。

从项目根目录运行，使用已有虚拟环境，不安装依赖：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.mixed_load_bench \
  --model /你的本地路径/Qwen3-0.6B \
  --num-requests 12 --seed 0 --output-dir results
.venv/bin/python -m benchmarks.mixed_load_bench --verify-only results
```

也可直接运行 `benchmarks/mixed_load_bench.py`。省略 `--model` 时，仅在本地唯一
Qwen3-0.6B 模型目录可确定时自动选择，不下载模型。输出目录必须为空，防止覆盖
已有实验。第二次运行应指定新的 `--output-dir`。

三种策略依次运行，每次使用独立子进程、独立模型实例和相同工作负载：

| policy 标签 | scheduling_policy | max_consecutive_decode_steps |
| --- | --- | --- |
| prefill_first | prefill_first | 0 |
| decode_first | decode_first | 0 |
| decode_first_fair2 | decode_first | 2 |

默认参数：`tensor_parallel_size=1`、`enforce_eager=True`、`max_model_len=512`、
`max_num_batched_tokens=256`、`max_num_seqs=4`、`gpu_memory_utilization=0.65`。
温度为 0.6，`ignore_eos=True`，输出长度由 `max_tokens` 控制。

## 工作负载和重复实验

默认 12 个请求，四种组合均出现：短/长 Prompt × 短/长输出。
短 Prompt 为 64–128 Token，长 Prompt 为 256–384 Token；短输出 16–32 Token，
长输出 48–64 Token。真实文本经过模型 tokenizer 编码，在 Token 边界截取，
把相同的精确 Token ID 提交给三种策略。每个请求校验
`prompt_tokens + max_tokens <= max_model_len`。

前 4 个请求计划在 0 秒到达，其余每隔 0.02 秒到达。可以调整：

```bash
.venv/bin/python -m benchmarks.mixed_load_bench \
  --model /你的本地路径/Qwen3-0.6B \
  --num-requests 16 --seed 42 --initial-burst 4 --arrival-interval 0.05 \
  --repeats 3 --output-dir results_repeat3
```

`--initial-burst 0` 表示第一个请求也等待一个到达间隔；
`--arrival-interval 0` 表示所有请求同时到达。`--policies decode_first_fair2`
可以只运行一个策略。

每次重复使用相同工作负载，采样随机种子为 `seed + repeat`。同一重复中的三个
策略使用相同采样种子，但调度批次会改变随机数消耗顺序，因此不承诺输出 Token
内容逐个相同。工作负载 Token ID、输出长度和计划到达时间可复现；GPU 实测时间
仍会受到频率、温度和其他负载影响。每次重复单独报告，不混合不同重复的分位数。

## 时间戳和指标

所有原始时间均为 `time.perf_counter()` 相对于正式计时起点的秒数。

- `scheduled_arrival`：计划到达时间。
- `admission_time`：真正执行 `scheduler.add()` 入队后的时间。
- `token_times`：每次真实 `Sequence.append_token()` 返回后的时间列表。
- `first_token_time`：`token_times[0]`；没有输出时为空。
- `completion_time`：`postprocess()` 完成状态转换和 KV 释放后的时间。
- TTFT：`first_token_time - admission_time`。
- ITL：同一请求相邻 `token_times` 之差，汇总所有请求的间隔样本。
- 完整请求延迟：`completion_time - admission_time`，仅统计完成请求。
- 接纳延迟：`admission_time - scheduled_arrival`，单独报告。
- 输出吞吐量：观测到的总输出 Token 数除以正式测试耗时，包含等待未来请求的空闲时间。

P50/P95 使用线性插值：排序后的位置为 `(n-1)*q`，在相邻两个样本间插值，
即 Hyndman–Fan type 7。没有样本时为 JSON `null` / CSV 空单元格；一个样本时
分位数就是该样本。`*_samples` 明确记录样本量。ITL 按间隔加权，不是先算每请求
均值后再平均；单 Token 请求没有 ITL 样本。

Benchmark 临时包装各请求实例的 `append_token()`，在原方法成功返回后记录 Token；
未生成输出的 Chunked Prefill 不会产生 TTFT/ITL 样本。`ModelRunner.run()` 的
`.tolist()` 已将采样结果取回 CPU，所以该时间是主机观察到的 Token 可用时间，
不是 CUDA Event 时间，也不是客户端网络接收时间。

调度轮数只统计成功返回的非空批次，抢占次数只统计成功执行的 `preempt()`。
包装会原样转发参数和返回值，并在退出时恢复，不修改调度器或 ModelRunner 源码。

模型加载、引擎初始化、显式暖机和结果序列化位于正式计时窗口之外。
暖机覆盖短/长 Prompt、Chunked Prefill 和 Decode 批大小 4、3、2、1。
每个正式窗口还检查当前 PyTorch 版本的 `torch._dynamo` 编译帧计数；如果计时期间
触发 `torch.compile`，将该运行标为失败，而不是将其用于性能对比。
`torch_compile_frames_during_measurement=0` 是有效运行的必要条件。

## 输出与一致性校验

- `results/workload.json`：精确工作负载、SHA-256、种子、引擎配置和源码 commit。
- `results/raw_requests.csv`：每策略、每次重复、每请求的完整时间戳及指标。
  `token_times` 和 `itl_s` 是 CSV 单元格中的 JSON 数组，单位为秒。
- `results/summary.json`、`results/summary.csv`：同一份平坦汇总指标、计数与样本量。
- `results/comparison.md`：三种策略对比表，时间列换算成毫秒。
- `results/runs/*.json`、`*.log`：独立进程的原始结果、软件/GPU 信息和异常日志。

每个子进程完成后，导出函数从 `raw_requests.csv` 的 Token 时间重算指标，核对
原始字段、请求数、重复 ID 及两种 summary 格式。也可用 `--verify-only` 独立复核，
这一步无需导入 torch 或访问 GPU。

## 范围和异常处理

这是同步、单线程、开环计划到达的引擎评测。只有在 `step()` 返回后才能接纳
运行期间到达的请求，所以会产生边界接纳延迟。它没有模拟生产级异步在线服务、
HTTP 排队或网络传输，不能把这里的 TTFT 当作端到端服务 TTFT。
`max_num_seqs` 沿用现有调度器语义，限制一次批次的请求数，不是总入队上限。

`--timeout-seconds`（默认 120）在步间检查正式测试超时，
`--max-steps`（默认 10000）限制调度步数。单个同步 GPU 调用不能被步间检查打断，
所以父进程另外用 `--process-timeout`（默认 300）限制包含初始化/暖机的整个进程。

异常会保留 traceback 和已观测到的部分结果，以非零退出码报告。失败运行在对比表
中不展示性能数值，不能将部分输出当作成功运行。如果 CUDA 不可访问，会明确
报告失败；没有 CPU 模拟性能或填充结果的回退路径。小规模负载不一定发生抢占，
实际计数为 0 时如实记录，不能据此声称验证了高 KV 压力下的性能。

## 可靠性复核和逐轮追踪

复用既有工作负载并平衡三种策略的执行位置：

```bash
.venv/bin/python -m benchmarks.mixed_load_bench \
  --workload-from results/workload.json \
  --repeats 3 --order-mode rotate --output-dir results_reliability
.venv/bin/python -m benchmarks.reliability \
  --baseline results --repeats results_reliability \
  --report benchmarks/benchmark_analysis_zh.md
```

`--workload-from` 读取精确 Token ID，并校验 SHA-256。引擎参数、种子、暖机和到达
设置必须与来源一致；来源使用非默认参数时，命令行也需传入对应参数。
`rotate` 的三轮顺序为 A/B/C、B/C/A、C/A/B；默认 `fixed` 保留原固定顺序。

新增 `raw_steps.csv` 记录每个真实 `engine.step()` 的阶段、批大小、请求 ID、
实际追加的输出 Token 序号、起止时间与主机耗时。没有改变 Scheduler 或
ModelRunner。每个 `runs/<policy>_repeatN/` 单独保存该次实验的请求/逐轮 CSV、
summary CSV/JSON 和元数据，旁边的同名 JSON 保留完整原始数据。
GPU 温度、SM/显存频率、功耗和利用率只在正式窗口前后读取，不在 Token 路径上
调用 `nvidia-smi`。遥测不可用时记录明确错误，不将缺失值填成零。

分析器校验每个输出 Token 与逐轮 trace 一一对应、Decode 输出数等于批大小、
汇总轮数匹配，以及全部请求与 KV Block 已释放。输出 `analysis.json`、
`cross_run_statistics.csv`、`decode_batch_statistics.csv`、`baseline_top_itl.csv`
及中文分析报告。跨运行统计对每次运行的指标求均值、样本标准差（ddof=1）、
最小/最大值；不混合原始样本重算分位数。旧实验没有逐轮 trace，分析器不会
伪造它的精确批大小分布或轮次耗时。

## 第四阶段：running 内部 Decode 队列策略

Config 新增 `decode_queue_policy="head_first"`（默认）或 `"round_robin"`。
它独立于 `scheduling_policy` 和 `max_consecutive_decode_steps`：前者控制 running
内部下一轮的优先顺序，后两者仍控制 Prefill/Decode 跨阶段优先级。

原三策略 CLI 默认值保持不变。四组队列对照实验显式指定：

```bash
.venv/bin/python -m benchmarks.mixed_load_bench \
  --workload-from benchmarks/baselines/stage3/workload.json \
  --policies prefill_first prefill_first_rr decode_first_fair2 decode_first_fair2_rr \
  --repeats 3 --order-mode rotate --output-dir results_stage4
.venv/bin/python -m benchmarks.decode_queue_analysis \
  --baseline results --results results_stage4
```

两个 `_rr` 标签启用 round_robin，其余标签保留 head_first；实际配置写入每次
summary 和请求 CSV。实验和暖机复用第三阶段实现，结果目录仍被 Git 忽略。

分析报告见 `decode_queue_analysis_zh.md`；`first_to_second.csv` 单独保存每请求的
首→第二 Token 间隔（每次 12 个样本），与全体 ITL（每次 480 个样本）分开统计。
`queue_run_metrics.csv`、`queue_cross_run_statistics.csv` 和 `queue_analysis.json`
保存逐次与跨运行指标。源码核心 Diff 另存本地结果目录的 `source.diff`。
4 组但只有 3 次重复时，简单 rotate 不能完全平衡 4 个执行位置，报告会保留这一限制。
