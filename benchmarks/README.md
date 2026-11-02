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

## 第五阶段第一步：Prefill 成本 Profiling

```bash
.venv/bin/python -m benchmarks.prefill_profile --output-dir results_prefill_profile
.venv/bin/python -m benchmarks.prefill_profile --verify-only results_prefill_profile
```

独立工具覆盖 q=16/32/64/128/256、c=0/64/128/256、batch_size=1/2/4。
q 为整轮实际执行的 Token 总数；同质批次每请求 q_i=q/batch_size，c 为每请求
前缀长度，同时导出 q_each/c_each/c_total。默认每组合3次暖机、10次测量。
max_model_len=1024，用于覆盖512个Prompt Token加一个输出Token；其余引擎模式为
TP=1、eager=True、显存比例0.65。

KV分配和真实前缀计算位于计时外：c>0 时调用原Runner执行前缀，再由原postprocess
推进缓存进度；之后原Scheduler选择目标轮次，不修改调度算法。输出一个Token后
立即完成请求，检查KV引用计数和释放。该微基准不等于完整在线请求延迟。

step_wall_ms 是主机 engine.step 总时间；runner_wall_ms、prepare_wall_ms 和
model_submit_wall_ms 提供主机侧分段。CUDA Event包围run_model的完整前向+logits，
记录为cuda_model_stream_ms；它排除输入准备、采样与后处理，但可能包含提交间隙，
不能称为纯活跃Kernel时间，也不能从主机总时间中相减得到纯CPU耗时。
`--no-cuda-events` 可关闭Event，此时字段为空，不填造设备时间。

结果写入Git忽略的本地目录：raw_prefill.csv、summary.csv、profile.json、
manifest.json、runs/batch*.json和日志。报告为prefill_profile_analysis_zh.md，包含
q×c关系表、每组合均值/P50/P95及测量限制；当前没有验证任何预测成本模型。
`--q-values`、`--c-values`、`--batch-sizes`、`--warmup-rounds`、`--repeats`
可控制小规模复现；上下文长度不满足或q无法均分的组合会明确列入skipped计划。

## 第五阶段第二步：Timeline 与进程留出验证

```bash
.venv/bin/python -m benchmarks.prefill_timeline --output-dir results_prefill_timeline
.venv/bin/python -m benchmarks.prefill_timeline --verify results_prefill_timeline
.venv/bin/python -m unittest discover -s tests -v
```

默认使用第一步的 seed=0、相同真实 Token Prompt 与引擎参数，覆盖任务指定的7种
组合。6个独立进程各组合5次暖机、15次正式采样；每轮随机组合顺序。事先固定偶数
进程训练、奇数进程留出，按时间交替执行。最后另启1个进程采集 torch.profiler，
不把 Profiler 数据输入预测模型。可用 `--processes`、`--repeats`、`--warmup-rounds`
调整规模，`--profile-only` / `--measure-only` 分开运行，`--case-ids` 选择指定子集。
每进程设超时，失败保留日志并报错；输出目录必须为空，以免覆盖已有结果。

NVML通过标准库ctypes只读采集SM时钟、温度、功耗；不支持的字段保留为空并报告
错误码。正式计时外采集前后快照，不是连续传感器采样。Step Wall与CUDA Event
沿用第一步口径：Event是前向+logits的流时间跨度，包含主机提交造成的间隙，
不等于活跃Kernel总耗时。

`results_prefill_timeline/`（Git忽略）保存 manifest、源码快照、逐进程JSON/日志、
raw_measurements.csv、process_summary.csv、case_summary.csv、timeline.json、
timeline_summary.csv、逐组合CPU算子/Kernel/算子类别/流间隙CSV，以及
traces/*.trace.json.gz。解压trace后可用现有Chrome Trace/Perfetto查看器查看。
CUDA关联使用correlation ID，CPU与GPU镜像annotation分开处理，避免重复计数。

`prefill_cost_model.py`仅使用标准库：A为训练样本中位数常数（适合MAE的基线），
B为`intercept + slope*q/256`最小二乘模型；均在独立进程留出集评价。误差逐样本
保存于holdout_predictions.csv，MAE/P95及误差分桶保存于holdout_metrics.csv和
cost_models.json。还有只使用训练进程的留一进程交叉验证，避免训练集拟合冒充
预测能力。当前7种组合中c与分页KV标记共线，不拟合无法区分这些因素的复杂模型。

实测解释及限制见 [prefill_timeline_analysis_zh.md](prefill_timeline_analysis_zh.md)。
Profiler和Python范围标记会引入额外开销，trace中的空隙只能描述该次可见的活动，
不能当作未插桩执行的GPU空闲率或据此宣称成本模型有效。

## 第六阶段第一步：KV Cache 原生容量高压测试

```bash
.venv/bin/python -m benchmarks.kv_pressure_bench --output-dir results_kv_pressure
.venv/bin/python -m benchmarks.kv_pressure_bench --verify results_kv_pressure
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

工具不限制或改写KV容量。原ModelRunner在初始化时按显存测量计算并分配完整KV池；
工具核对实际张量、Runner配置和BlockManager的容量完全相等，并要求所有对比运行
容量一致。`num_kvcache_blocks`输入会被原Runner覆盖，不能用于固定容量；本工具
不采用修改free_block_ids的方式制造压力，也不修改Scheduler/ModelRunner源码。

默认低/中/高档为16/48/80个同时到达的请求；同一seed生成同一80请求序列，各档
取前缀，同一档各策略使用完全相同的Prompt Token IDs。Prompt交替248/504 Token，
输出16 Token，跨越256 Token页边界；请求1/3有一个刻意共享的完整256 Token前缀。
每个请求独立完成最多需要2或3块，运行前检查其资源和上下文约束。
压力标签是施加的负载等级，是否发生抢占由实际调度和占用决定。

比较prefill_first/head_first、prefill_first/round_robin、decode_first/fair2/round_robin。
默认3次重复，27个独立进程；每组两轮相同暖机，轮换策略与压力等级顺序。
引擎参数为TP=1、eager=True、max_model_len=768、Token预算256、max_num_seqs=4、
显存比例0.65。暖机后仅清除空闲Block的Prefix索引/Token元数据，保持真实容量、
free列表和引用计数不变，防止暖机Prefix命中影响正式负载。

沿用EngineObserver在真实append_token之后记录时间、完成后记录完成时间，
导出TTFT/ITL/吞吐等原指标，并增加逐请求抢占、实际重算、成功分配时的Prefix
复用、缓存覆盖和峰值占用。重算量只计算抢占后本轮执行区间与历史有效KV前缀的
交集，普通Prefill和尚未计算过的输出Token不计入。恢复时Prefix命中避开了哪些
Token计算也单独保存；只计算成功allocate命中，不把can_allocate探测重复累计。

recompute_affected_step_wall_s包含混合批次中的新计算、调度与同步；
recompute_only_step_wall_s仅汇总整批Token都属于重算的轮次，两者都不等于反事实
的边际延迟。marginal_preemption_overhead_s保留为空，不按Token比例摊分耗时。
step_weighted_block_utilization按每步调度后占用乘整步耗时加权，是采样近似，
不是逐次分配/释放事件积分得到的精确时间占用率。

结果保存于Git忽略的results_kv_pressure：manifest及源码快照、runs逐进程JSON和
日志、总summary.csv、pressure_runs.json、cross_run_statistics.csv；各压力子目录
包含summary.csv/JSON、raw_requests.csv、steps.csv、preemption_events.csv、
allocation_events.csv。异常保留日志/部分结果并报错，不计入成功性能统计。
超时、总步数与无新输出/新KV进度限制分别由--timeout-seconds、--max-steps、
--no-progress-steps控制。分析见[kv_pressure_analysis_zh.md](kv_pressure_analysis_zh.md)。

## 第六阶段第二步：KV首次接纳余量保护

Config新增`kv_admission_policy="none"|"reserve"`（默认none）与
`kv_reserve_blocks=0`（严格非负整数，拒绝bool/浮点数等）。none或reserve=0
保持旧调度轨迹。reserve只在新请求首次分配前检查：

`free_blocks - (seq.num_blocks - 活跃共享前缀块数) >= kv_reserve_blocks`

空闲缓存命中仍消耗free槽位，不能减去全部Prefix命中块。该检查不修改BlockManager，
也不是Prefix收益评分或缓存命中率优化。保留FIFO和原RR；已持块Chunked Prefill、
抢占恢复和running为空的前进路径不被余量阻塞。Decode/恢复仍可用掉余量，
所以它不保证全程free>=reserve，也不保证零抢占。物理容量不足仍明确报错。

```bash
.venv/bin/python -m benchmarks.kv_pressure_bench \
  --workload-from results_kv_pressure/manifest.json --expected-blocks 105 \
  --policies prefill_first B0 B4 B8 B16 decode_first_fair2_rr \
  --repeats 3 --output-dir results_kv_admission_20261102
.venv/bin/python -m benchmarks.kv_pressure_bench --verify results_kv_admission_20261102
.venv/bin/python -m benchmarks.kv_admission_analysis --results results_kv_admission_20261102
```

原默认三策略仍可运行；B0/B4/B8/B16均为prefill_first+round_robin，余量分别0/4/8/16。
A/C参考策略重新独立实测。精确复用上阶段Token工作负载，三压力×六配置×三重复，
不覆盖原目录；物理容量不等于105时明确失败，不裁剪free列表。

新结果保存策略配置、Config/Scheduler等源码快照，以及逐请求首次Block分配时间、
reserve拒绝尝试数和episode等待时长。reserve_deferral_events.csv记录每次返回False；
reserve_wait_events.csv将首次拒绝到首次实际分配合并一次。这个关联等待包含后续
FIFO/容量等待，不能称为reserve的纯边际延迟；sum可重叠，不等于测试Wall Time。
没有拒绝时P50/P95为空。原重算、Prefix、资源及逐Token指标继续核验。

离线分析工具输出全部阈值、标准差/min/max、相对B0变化、请求平均延迟的可加分解
（首次分配前等待+首次分配至首Token+首Token至完成），以检查等待是否转移。
详见[kv_admission_analysis_zh.md](kv_admission_analysis_zh.md)。保持none默认，不自动
加入动态余量、成本模型或其他调度优化。
