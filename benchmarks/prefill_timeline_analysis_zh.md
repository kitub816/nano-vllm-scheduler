# Prefill Timeline 与离线成本模型验证（第五阶段第二步）

2026-10-29（Asia/Shanghai），基础提交 `51d3bc4`，RTX 4060 Laptop 8GB，
本地 Qwen3-0.6B。此阶段没有修改 Scheduler、ModelRunner 或 CUDA Kernel，
没有安装依赖，没有 Commit/Push。

## 结论

当前数据不支持引入 Cost-aware Scheduler。此前 q=64 明显比 q=16/256 慢的峰值
未稳定复现；小幅逆序仍然存在。单轮耗时大体受较大的固定主机提交开销影响，
底层 GEMM 路径也随形状切换，不能将耗时简单视为 q/c 的光滑单调函数。

在预先固定的独立进程留出集上，q 线性模型的 Step Wall MAE 为 **2.752ms**，
常数基线为 **2.410ms**。q 模型虽然降低汇总 P95 绝对误差，但在三个留出进程上
MAE 都更差，不能宣称它具有可靠预测优势。没有强行拟合复杂模型。

这解释了可见机制与不确定性，不能唯一追溯上次某个45ms样本的因果来源：上次没有
对应逐轮CPU/CUDA trace或连续频率记录。本次还没有隔离CPU频率、WSL主机调度与
其他系统负载，不能直接归因为温控、某一个算子或测量Bug。

## 源码执行路径与测量范围

1. `LLMEngine.add_request()`创建真实Sequence并入waiting；`step()`依次调用
   Scheduler.schedule、ModelRunner.run、Scheduler.postprocess。原调度接口不变。
2. `prepare_prefill()`构造Token/position、累积序列长度、slot mapping并复制到GPU；
   当已计算前缀导致k长度大于q长度时构造block_tables，Attention使用分页KV路径。
   c=0时直接使用当前轮QKV；两个分支均写入真实KV Cache。
3. `run_model()`执行完整模型前向和compute_logits；Prefill走eager路径。
   enforce_eager并不关闭局部RMSNorm、RoPE等已有torch.compile装饰。
4. `run()`随后采样并tolist，GPU输出复制回主机；postprocess推进缓存计数，
   仅完整Prefill后append_token。工具要求目标轮每请求真实输出1个Token，随后
   全部完成并释放KV。未完成的Chunked Prefill不作为目标测量。

复用第一步的prepare_case与TimingProbe：先用真实Runner计算c个前缀Token，
经原postprocess更新缓存，再让原Scheduler调度目标轮。KV分配、前缀计算、
测量前cuda.synchronize在计时外；输入准备、采样和后处理属于Step Wall。
它是受控目标轮次微基准，不是从零开始完成整个请求的成本。

- **q**：整轮实际计算的Token总数；同质批次每请求q_i=q/batch_size。
- **c**：每请求已完成前缀Token数；同时保留c_each、c_total和真实block_tables。
- **Step Wall Time**：同一time.perf_counter单调时钟下engine.step首尾差值，
  结束时采样结果已在CPU，且已执行postprocess。
- **CUDA Event Time**：run_model前后同一CUDA流的时间跨度，覆盖前向+logits；
  排除输入准备、采样和postprocess，但包含主机提交不及时造成的流间隙。
  读取前验证结束Event已完成。不能叫纯活跃GPU时间，也不能用Wall减Event得到
  纯CPU时间。
- **Profiler Kernel时间**：从Chrome trace按CUDA correlation关联到前向内的
  CPU launch，再取GPU Kernel区间。CPU范围可能早于Kernel完成，不能仅用CPU
  起止时间裁剪GPU事件。Kineto的gpu_user_annotation镜像不当作额外CPU范围。
- **Kernel间隙**：同流相邻Kernel区间之间的可见空隙；同时扣除trace里可见的
  memcpy/memset活动，另存observed_gpu_activity_free_ms。不是整块GPU的全局
  空闲率，也不包含首Kernel之前、末Kernel之后的间隙。
- **CPU算子**：导出self与inclusive时间。Attention/Linear/RMSNorm采用临时
  record_function范围与对应Kernel归类；归类到Attention的时间包含KV写入。
  Python调用、TorchDynamo查找和GPU执行会重叠，inclusive时间不能跨层相加。

## 实验设计与元数据

模型快照 `c1899de289a04d12100db370d81485cdf75e47ca`，bfloat16，TP=1，
enforce_eager=True，max_model_len=1024，max_num_seqs=4，
gpu_memory_utilization=0.65，KV Block大小256，容量105块。
初始max_num_batched_tokens=256，目标轮预算设为该组合q，不修改调度算法。
长度上限与第一步一致，允许c=256/q=256加一个输出Token。

torch=2.7.1+cu126，CUDA=12.6，transformers=4.51.3，triton=3.3.1，
flash-attn=2.8.3.post2，Python=3.12.16。没有更换软件版本。

7组合×6独立测量进程×15正式重复=**630个正式样本**；每组合90个。
每进程先对每组合暖机5次，共210个初始暖机样本，编译不计入正式结果。
正式采样每轮用seed+process_id随机排列组合。使用seed=0，保存每个Prompt的真实
Token IDs，已逐组合核对与第一步保存的Prompt完全一致。

进程按0→1→2→3→4→5依次启动、独立加载模型并退出，不并发争用GPU。
预先固定0/2/4训练、1/3/5留出，每边315样本，执行时间交替分配角色。
在此之外第6号独立进程采集7份Profiler trace，先各组合暖机5次、再在插桩下
追加2次暖机，每组合记录1个目标前向；同时保留邻近的未开Profiler单次control。
训练/留出分割写在运行前manifest中；Profiler与control都不用于成本模型训练。

分位数统一为type 7：排序后在(n-1)*p位置线性插值。跨进程标准差为6个进程均值
的样本标准差（n-1），不是将90个样本当作90个独立进程。
所有正式样本和trace中编译计数均为0；初始化、模型加载、前缀准备、暖机均不计时。

## 未开启Profiler的耗时

下表均为ms；均值为每组合90次正式样本的均值（每进程等量，因此也等于6个进程
均值的均值）。P50/P95合并90次采样，标准差仅针对6个进程均值。

| 组合 | n | Wall mean | Wall P50 | Wall P95 | 进程均值SD | Event mean | Event P95 |
|---|---|---|---|---|---|---|---|
| b1_q16_c0 | 90 | 34.613 | 33.623 | 42.160 | 1.149 | 33.953 | 41.423 |
| b1_q64_c0 | 90 | 34.920 | 34.104 | 42.040 | 1.055 | 34.202 | 41.409 |
| b1_q256_c0 | 90 | 35.244 | 34.365 | 42.314 | 1.649 | 34.528 | 41.627 |
| b1_q16_c256 | 90 | 34.950 | 33.630 | 43.002 | 1.724 | 34.337 | 42.470 |
| b1_q256_c256 | 90 | 34.911 | 34.096 | 41.486 | 1.434 | 34.262 | 40.903 |
| b4_q64_c0 | 90 | 34.384 | 33.567 | 40.255 | 0.832 | 33.678 | 39.545 |
| b4_q256_c0 | 90 | 35.055 | 34.102 | 41.157 | 1.674 | 34.313 | 40.392 |

b1/c0逐进程均值如下（ms）；q64同时大于q16和q256只发生在进程3，即1/6进程，且差值很小。

| 进程 | 用途 | q16 | q64 | q256 |
|---|---|---|---|---|
| 0 | 训练 | 33.874 | 34.252 | 34.706 |
| 1 | 留出 | 36.645 | 35.814 | 35.510 |
| 2 | 训练 | 34.795 | 35.853 | 37.144 |
| 3 | 留出 | 33.249 | 33.684 | 33.301 |
| 4 | 训练 | 34.433 | 35.939 | 37.103 |
| 5 | 留出 | 34.681 | 33.980 | 33.698 |

第一步对应均值为32.856、45.082、31.831ms（当时每组合仅10次、顺序按组合聚集）。
本次合并均值34.613、34.920、35.244ms，原来的明显q64峰值没有稳定复现。
仍有进程内逆序，不能说已经得到严格单调规律，也不能凭这次未复现就认定上次是Bug。

整个正式集Step Wall范围29.249–55.271ms，均值34.868ms；Event均值34.182ms；
主机model_submit均值32.841ms，输入准备均值0.330ms。主机提交前向所耗时间很大，
Event并未独立测出一个可直接用于计算量建模的纯GPU区间。

## 时钟、温度和功耗

用ctypes调用当前已有NVML库，只读获取SM时钟、温度、功耗，不修改频率/功率设置。
每次目标采样前后各取一次，共1260个传感器快照。前快照位于prepare_case之前，
因此包围的是准备工作加目标轮，并非恰好每个Kernel的瞬时状态；字段是驱动传感器
读数，不保证在每个请求内刷新。没有将NVML调用放进Step计时。

| 传感器 | 有效快照 | min | mean | max |
|---|---|---|---|---|
| sm_clock_mhz | 1260 | 2055.000 | 2448.095 | 2610.000 |
| temperature_c | 1260 | 50.000 | 51.958 | 54.000 |
| power_w | 1260 | 18.067 | 33.704 | 49.212 |

| 进程 | Wall均值ms | SM时钟均值MHz（前快照） | 温度均值°C（前快照） | 功耗均值W（前快照） |
|---|---|---|---|---|
| 0 | 34.010 | 2474.857 | 51.076 | 33.958 |
| 1 | 35.939 | 2529.714 | 51.524 | 35.271 |
| 2 | 35.847 | 2478.286 | 52.362 | 34.032 |
| 3 | 33.507 | 2359.571 | 51.724 | 31.633 |
| 4 | 36.055 | 2560.857 | 52.933 | 36.034 |
| 5 | 33.849 | 2288.857 | 52.143 | 31.243 |

温度为50–54°C，没有仅凭这些端点快照就证明热降频的证据。进程3/5的平均SM时钟
低于进程1/4但Wall也更短，不能用单一GPU时钟解释主机侧整轮延迟。时钟/功耗具有
动态变化，但没有锁频对照、连续传感器或CPU调度数据，相关性不能确认为因果。

## Profiler：结构证据与采集开销

以下每组合只有1份Profiler目标trace。其耗时不合并进前面的原始性能统计。
control为本进程邻近的1次未开Profiler采样，仅用于展示量级，不能用来精确扣除
开销。额外Python范围标记本身也有成本，下面的增量包含Profiler及这些范围标记。

| 组合 | control Wall | Profiler Wall | 模型CPU范围 | Event跨度 | Kernel次数 | Kernel活跃并集 | Kernel跨度 | 无可见GPU活动 |
|---|---|---|---|---|---|---|---|---|
| b1_q16_c0 | 34.925 | 53.401 | 49.525 | 50.632 | 370 | 6.006 | 50.395 | 44.389 |
| b1_q64_c0 | 32.774 | 52.396 | 50.289 | 51.418 | 426 | 6.285 | 51.233 | 44.948 |
| b1_q256_c0 | 35.519 | 54.211 | 50.810 | 51.927 | 370 | 11.078 | 51.729 | 40.615 |
| b1_q16_c256 | 36.269 | 53.519 | 49.938 | 50.924 | 370 | 5.981 | 50.695 | 44.713 |
| b1_q256_c256 | 34.422 | 55.833 | 52.605 | 54.103 | 370 | 18.310 | 53.870 | 35.516 |
| b4_q64_c0 | 35.317 | 54.235 | 50.929 | 52.095 | 426 | 6.419 | 51.976 | 45.556 |
| b4_q256_c0 | 33.639 | 53.371 | 50.653 | 51.826 | 370 | 14.001 | 51.605 | 37.557 |

Profiler Wall为52.396–55.833ms，相比邻近control多17.250–21.410ms。
Kernel活跃并集仅5.981–18.310ms，而Kernel首尾跨度50.395–53.870ms，
说明本次trace存在大量可见间隙。由于Profiler放大主机开销，不能把这些差值或比例
当成正常运行中的空闲率。未开Profiler时model_submit与Event也接近整轮时间，
两种证据共同支持“主机发射/框架开销很显著”，仍不构成对历史长样本的唯一解释。

每份trace的模型Kernel均在stream 7，未关联Kernel为0。整step还有6次额外Kernel
用于模型范围之外的采样等工作，按前向边界排除，没有重复计入前向Kernel次数。
同流相邻Kernel间隙统计如下，单位ms；仅表示这次trace可见的间隙，不代表硬件调度
延迟或其他GPU进程完全无工作。

| 组合 | 间隙n | Kernel间隙均值 | P95 | max | CPU launch间隔均值 | P95 |
|---|---|---|---|---|---|---|
| b1_q16_c0 | 369 | 0.120 | 0.217 | 0.679 | 0.120 | 0.208 |
| b1_q64_c0 | 425 | 0.106 | 0.227 | 0.455 | 0.104 | 0.208 |
| b1_q256_c0 | 369 | 0.110 | 0.221 | 0.580 | 0.123 | 0.215 |
| b1_q16_c256 | 369 | 0.121 | 0.231 | 0.364 | 0.120 | 0.212 |
| b1_q256_c256 | 369 | 0.096 | 0.228 | 0.948 | 0.128 | 0.215 |
| b4_q64_c0 | 425 | 0.107 | 0.234 | 0.418 | 0.105 | 0.196 |
| b4_q256_c0 | 369 | 0.102 | 0.207 | 0.376 | 0.123 | 0.205 |

算子类别的CPU范围inclusive耗时 / 对应CUDA Kernel时间和（ms）：
CPU是主机执行与提交范围，不是Kernel完成时间；GPU是Kernel持续时间和，
本次单流近似等于并集，但一般多流可能重叠，不能随意相加解释Wall。

| 组合 | Attention CPU/GPU | Linear CPU/GPU | RMSNorm CPU/GPU | LMHead CPU/GPU |
|---|---|---|---|---|
| b1_q16_c0 | 7.074 / 0.196 | 8.552 / 4.203 | 18.573 / 0.189 | 0.213 / 1.267 |
| b1_q64_c0 | 7.527 / 0.167 | 9.085 / 4.522 | 18.037 / 0.173 | 0.224 / 1.271 |
| b1_q256_c0 | 7.580 / 0.603 | 9.546 / 8.433 | 17.903 / 0.345 | 0.272 / 1.322 |
| b1_q16_c256 | 7.406 / 0.419 | 8.765 / 4.047 | 17.826 / 0.143 | 0.292 / 1.262 |
| b1_q256_c256 | 7.519 / 2.282 | 10.786 / 13.264 | 18.200 / 0.538 | 0.236 / 1.643 |
| b4_q64_c0 | 7.371 / 0.258 | 9.389 / 4.518 | 18.028 / 0.173 | 0.328 / 1.318 |
| b4_q256_c0 | 7.666 / 0.432 | 9.275 / 11.306 | 17.968 / 0.462 | 0.277 / 1.313 |

可复核的结构差异：

- q=64时前向426次Kernel，q=16/256时370次。q64的Linear类别为168次，
  其他为112次，多出的56次名称为`cublasLt::splitKreduce_kernel`。
  q16使用CUTLASS WMMA GEMM；q64使用Ampere GEMM加split-K reduction；
  q256使用另一组Ampere GEMM。这证明形状触发真实算法/Kernel组合切换，
  不能按Token数假设相同固定Kernel路径。它并不能独自证明上次q64多出的13ms
  全部来自split-K；本次q64并未显著更慢。
- b4/q64与b1/q64具有相同总q和GEMM Kernel路径，但Attention处理序列数量不同。
  仅用batch_size忽略每请求q_i也会丢失语义。
- c>0使Attention通过block_tables读分页KV，c=0直接使用当前K/V；这改变路径，
  但不是在c=0附近保持同一公式的平滑增加。q16/c256与q16/c0的Kernel数均370，
  并不意味着逐Kernel成本相同。
- 每个前向有113次RMSNorm Kernel，CPU范围约18ms而CUDA总和仅0.14–0.54ms。
  每份trace还有170次TorchDynamo Cache Lookup，CPU self约4.83–5.12ms；
  这里是缓存查找，不是正式轮次重新编译（编译计数已验证为0）。
  这些小算子和框架查找说明固定提交开销值得重视；数值仍受Profiler影响。
- Linear与LMHead构成活跃Kernel时间的大头；q和c增加会改变活跃工作量，但部分
  计算可填入原有发射间隙，使整轮Wall对计算量的变化不敏感。这个解释与trace
  一致，尚未通过去除框架开销或硬件计数器的对照实验确认为完整因果模型。

## 离线模型与严格进程留出

A：训练315个样本的中位数常数。中位数是训练MAE最优的常数，避免选一个偏弱
的常数基线。B：`intercept + slope*q/256`，标准库最小二乘拟合，预测下限0。
没有使用进程ID、温度或留出样本拟合系数，也没有按单个样本随机拆分同一进程。
Wall与CUDA Event分别拟合，两者均不是活跃Kernel时间预测模型。

| 目标 | 模型 | intercept ms | slope ms/256 Token | 训练进程 | 训练n |
|---|---|---|---|---|---|
| step_wall_ms | constant | 34.489060 | 0.000000 | 0,2,4 | 315 |
| step_wall_ms | q_linear | 34.991103 | 0.604257 | 0,2,4 | 315 |
| cuda_model_stream_ms | constant | 33.757023 | 0.000000 | 0,2,4 | 315 |
| cuda_model_stream_ms | q_linear | 34.324288 | 0.543162 | 0,2,4 | 315 |

留出进程1/3/5合计315个样本；误差=预测−实际，以下绝对误差单位ms：

| 目标 | 模型 | MAE | P50 AE | P95 AE | max AE | 偏差 |
|---|---|---|---|---|---|---|
| step_wall_ms | constant | 2.410 | 1.746 | 6.829 | 20.782 | 0.057 |
| step_wall_ms | q_linear | 2.752 | 2.380 | 5.870 | 19.675 | 0.872 |
| cuda_model_stream_ms | constant | 2.355 | 1.733 | 6.908 | 20.780 | -0.001 |
| cuda_model_stream_ms | q_linear | 2.706 | 2.348 | 5.850 | 19.669 | 0.848 |

| 目标 | 模型 | |误差|<1ms | [1,2) | [2,5) | [5,10) | ≥10ms |
|---|---|---|---|---|---|---|
| step_wall_ms | constant | 82 | 91 | 115 | 22 | 5 |
| step_wall_ms | q_linear | 53 | 74 | 160 | 24 | 4 |
| cuda_model_stream_ms | constant | 89 | 88 | 111 | 22 | 5 |
| cuda_model_stream_ms | q_linear | 50 | 81 | 159 | 21 | 4 |

| 留出进程 | Wall 常数MAE | Wall q模型MAE | 常数P95 AE | q模型P95 AE |
|---|---|---|---|---|
| 1 | 3.199 | 3.264 | 9.694 | 8.701 |
| 3 | 1.873 | 2.347 | 3.922 | 4.372 |
| 5 | 2.159 | 2.643 | 4.819 | 5.267 |

训练内部留一进程验证也保留于cost_models.json：Wall的常数/q模型MAE分别为
进程0：2.185/2.648，进程2：2.875/2.890，进程4：2.731/2.406ms。
没有一致的q模型优势。汇总留出P95改善主要出现在有较长尾的进程1；进程3和5
P95反而更差。中位数基线与最小二乘的损失目标不同，这不是对所有q模型的否定，
但本次预设简单模型没有展示可靠收益，也不能只挑选更好的单一指标宣称有效。

不拟合C的原因：本设计c只有0/256，c/256与分页KV路径标记完全共线；batch=4
只覆盖c=0，且q64/q256的数据范围窄。复杂模型无法独立识别c、路径和batch交互。
即便训练集拟合更好也很可能只记忆7个组合。当前常数已覆盖主要固定开销，
q系数只有约0.60ms/256 Token，而进程/样本波动大于该量级。

下一步若继续研究，先固定/记录CPU与GPU运行状态，扩展相同分页路径下的c网格、
不同q_i与batch的完整交叉，并跨日期、跨独立进程重新预注册留出验证。可先研究
Kernel数量/算法选择和主机提交成本的分段描述；只有稳健优于强常数基线且误差
小于实际调度决策差异，才有理由讨论成本感知调度。当前建议保持调度器不变。

## 原始文件、完整性与运行命令

正式数据位于Git忽略的 `results_prefill_timeline/`（约7.7MB）：

- manifest.json：运行前固定参数、分割、随机种子、7组合、基础commit、源码哈希。
- prefill_timeline.py / prefill_profile.py / prefill_cost_model.py：启动时源码快照。
- measurement_0..5.json / .log：每进程环境、Prompt IDs、暖机、105个正式样本、
  NVML前后快照、实际q/c、KV状态与失败信息。
- raw_measurements.csv / timeline.json：630条原始正式数据与聚合；
  process_summary.csv、case_summary.csv保存逐进程和跨进程均值/分位数/标准差。
- profiler_6.json / .log、timeline_summary.csv、traces/*.trace.json.gz：
  7份真实trace及解析；每组合另有operators.csv、kernels.csv、categories.csv、
  streams.csv，保留完整CPU算子和Kernel名称/持续时间/发射时间。
- cost_models.json、holdout_metrics.csv、holdout_process_metrics.csv、
  holdout_predictions.csv：模型系数、训练内部验证、留出误差分桶和1260条预测
  （315样本×2目标×2模型），可以重算所有误差。

工具会拒绝重复/缺失样本、测量中编译、错误q/c/输出数、计时不一致、未释放的KV；
复核时逐字段比较CSV与JSON，重算summary和模型，重新解析7份trace并与汇总比较。
已核对7组合Prompt IDs与第一步逐Token一致。正式目标轮共输出1170个Token，
即完成1170个单输出请求；最大同时使用4/105个KV Block，每轮used/ref_count回0，
所有7个worker退出前105/105块可用。这里不是长输出在线负载，不能替代混合负载
Benchmark的TTFT/ITL评估。

```bash
.venv/bin/python -m benchmarks.prefill_timeline --output-dir results_prefill_timeline
.venv/bin/python -m benchmarks.prefill_timeline --verify results_prefill_timeline
.venv/bin/python -m unittest discover -s tests -v
git diff --check
git status --short
```

试采使用独立results_prefill_timeline_pilot*目录，未混入正式样本；早期试采因
Kineto将同名CPU annotation镜像到GPU而触发范围数量断言，保留失败日志后修正
解析为只读取CPU user_annotation，并用单元测试覆盖镜像情形。另一次参数路径
JSON序列化错误发生在启动GPU worker之前。均没有隐藏失败或用假数据替代。

原有74项测试完整保留，新增12项纯Python测试，合计**86项通过**。
测试覆盖correlation归属、GPU工作超出CPU范围、镜像不重复计数、跨流重叠、
可见memcpy、CUDA trace缺失、模型进程隔离、留出数据不影响拟合、误差分桶、
非法/退化模型输入，以及缺失/重复/泄漏/编译/计时异常检查。

## 限制

单GPU、单模型、单次日期窗口、7种受控组合，只有3个留出进程；同一进程的15次
重复不是15个独立系统状态。原数据和本次交错测量的顺序、暖机次数不同，
留出集仍是相同7种组合，只验证跨进程预测，没有验证未见过的q/c组合或外推能力。
不能仅凭均值变化确立某一个原因。没有CPU频率/线程调度、内存带宽、SM占用率等
硬件计数器；端点SM时钟/温度/功耗无法复原step内部变化。Profiler每组合仅1份，
显著增加开销；Kernel时间和类别耗时也受频率、并发与采集扰动，不能视为无扰动
性能。预先构造真实前缀与预分配KV排除了资源竞争、排队、接纳延迟及在线请求到达。
没有修改算法验证收益，没有证明q/c成本模型可泛化，更不能宣称一种调度普遍更快。
