# 第六阶段第四步：Bounded Bypass 最小候选策略实测

日期：2026-11-02（Asia/Shanghai）。基础提交：`6a79269814b971c27f5f55971f0ea3627c6b5b3c`。

## 结论

保留为**默认关闭的实验开关**，不建议改为默认策略。本次三次独立重复中均出现两次
真实越过，短请求66的首次Block分配等待平均减少7.732秒，证明局部FIFO绕过可以生效。
但80请求整体首次分配等待均值只减少10.2毫秒，且三次变化为-91.2/-21.4/+82.0毫秒，
收益不稳定。大请求等待均值和P95、TTFT P95、ITL P95/P99均变差；吞吐均值降低1.83%，
抢占从21次增至22次，重算从8,981增至9,238 Token。不能据此宣称整体优化成功。

## 源码语义与最小改动

- `Config.kv_bounded_bypass: bool = False`：独立开关，严格接受bool，拒绝整数、浮点数、
  字符串等。需要同时启用`kv_admission_policy="reserve"`且余量大于0才可能发生越过。
- `Scheduler._schedule_prefill()`：保留队首的物理容量和Token预算检查。只有队首通过
  `can_allocate()`、Token预算检查，再被`_can_admit_prefill()`拒绝时，才检查第二项。
  物理容量失败、单纯Token预算不足不触发绕过。
- `_bypass_candidate()`：只看`waiting[1]`，不向后扫描。队首与候选都必须未首次接纳，
  没有Block且没有输出；有Block的Chunked Prefill和已有输出的抢占恢复请求均受保护。
  候选重新通过真实`can_allocate()`、reserve和剩余Token预算检查。
- `_promote_bypass()`：候选的真实`allocate()`成功后才提升队列优先级、记录被越过队首的
  `seq_id`。候选若需分块，保留在队首直至Chunked Prefill完成；完整Prefill则立即按原路径
  从waiting转到running。原队首仍保留在waiting，首次接纳前无法再次被越过。
- `_bypassed_before_admission`仅存储已被越过、尚未首次分配的请求；请求首次真实分配时删除
  自身记录。失败的候选检查不消耗额度，也不更改队列或KV引用。有限负载完成后集合为空。

真实Block成本沿用上阶段：`seq.num_blocks - 活跃共享Prefix块数`。活跃共享只增加ref_count；
空闲缓存命中仍从free列表移除，不能视为零物理成本。整个Prompt首次就分配所需Block，
分块只是分轮计算。Decode在`len(seq) % 256 == 1`时需要扩页；原尾部受害者选择、释放、
恢复Prefill及可能的缓存复用路径均未改动。

`BlockManager`、`Sequence`、`ModelRunner`和CUDA Kernel没有本阶段修改。
`_schedule_decode()`、`preempt()`、`postprocess()`、跨阶段fair2计数与RR队列顺序保持原实现。
running为空时原reserve前进豁免仍在；物理资源确实无法满足时继续明确抛出RuntimeError。
未增加超时抢跑、动态余量、缓存收益评分或其他调度策略。

Benchmark改动仅增加`B8_bypass`身份、传递开关、记录成功`_promote_bypass()`事件及一致性校验。
事件包含真实队列前后顺序、队首/候选ID、分配事件索引、物理Block成本、分配后free和reserve。
分析脚本`kv_bypass_analysis.py`读取并核验六次实测，输出分组等待、跨次统计和逐请求配对差值。

## 测试与兼容性

先增加CPU测试，观察到配置尚不存在时13项失败，再实现开关。正式GPU前全部135项通过。
随后新增3项纯Python分析算术测试，以及1项缓存命中与候选Token预算交互测试，最终
**139项全部通过**；原有119项全部保留。

新增覆盖默认关闭完整轨迹、一次越过额度、大请求有限负载排空、只检查第二项、失败检查
不耗额度、持块Chunk和候选Chunk续算、抢占恢复两侧保护、活跃/空闲缓存物理成本、物理不足
明确报错、Token预算/max_num_seqs、真实抢占恢复、RR/fair2顺序、事件重复检测和引用归零。
分析测试采用明确的合成CPU数据，不作为性能结果。

另直接加载基础HEAD的Scheduler类，与当前类比较四种配置的完整80请求CPU轨迹：
默认关闭的Prefill-first+RR+B8、默认关闭的Decode-first+fair2+RR+B8、开关开启但reserve=0、
开关开启但admission=none。逐轮队列、Token预算、Block IDs和引用计数的轨迹hash完全相同。
此项为逻辑兼容性检查，不是GPU性能测量；记录于`fifo_compatibility_check.json`。

## 实验设置与计时

- RTX 4060 Laptop GPU 8GB，Qwen3-0.6B，本地模型路径见manifest。
- Python 3.12.16，PyTorch 2.7.1+cu126，CUDA 12.6；没有安装/升级依赖。
- 原生物理容量105 Block，每块256 Token；KV张量shape为`[2,28,105,256,8,128]`。
  没有裁剪free列表或伪造容量，物理张量、Config及BlockManager容量一致。
- 完整复用`results_kv_admission_20261102/manifest.json`的high工作负载：80请求，
  40个248-Token短Prompt、40个504-Token长Prompt，全部生成16 Token、ignore_eos=True，
  同一对请求共享一个完整Prefix块。Token IDs和输出长度未重新生成。
- 两组均Prefill-first+Round-robin、reserve=8、max_model_len=768、
  max_num_batched_tokens=256、max_num_seqs=4、TP=1、enforce_eager=True、显存利用率0.65。
- 每次独立进程，先相同的两轮暖机（248/504分块Prefill及批量Decode），清理空闲Prefix
  元数据并同步，再正式计时。加载、初始化和暖机不计入；正式窗口编译frame数全部为0。
- 每对相同repeat使用相同seed=`0+repeat`；执行顺序为
  `FIFO0, Bypass0, Bypass1, FIFO1, FIFO2, Bypass2`。三对轮换减轻顺序偏差，仍非完全平衡设计。
- 每次80个TTFT/延迟样本、1,200个相邻输出ITL样本；长请求等待每次40个样本。
  跨次统计是3个每次指标的均值、样本标准差(ddof=1)、min/max，未合并样本再算P95。
  分位数统一为type 7线性插值`(n-1)*p`。

时间戳全部来自同一`perf_counter()`起点。实际入队为真实add之后，首次Block分配为真实
allocate之后，Token完成在GPU采样结果转为主机Token IDs、真实append成功后记录，完成在
postprocess释放后记录。Chunked Prefill未输出时不记录TTFT。首次分配等待从实际入队计算，
不是从计划到达算，也不是分配后的首Token完成时间。TTFT、ITL、完成延迟和吞吐仍沿用
已有观察器，包含Python/同步/观察开销，不能称为纯GPU内核耗时。

## 实测对比：三次均值 ± 样本标准差

时间单位均为秒；吞吐为输出tok/s，Block均值是步耗时加权近似。

| 指标 | B8 FIFO | B8 Bounded Bypass |
|---|---:|---:|
| TTFT Mean | 3.341835 ± 0.024939 | 3.332644 ± 0.071781 |
| TTFT P50 | 1.891194 ± 0.045551 | 1.925071 ± 0.056848 |
| TTFT P95 | 11.355167 ± 0.051424 | 11.644552 ± 0.209562 |
| ITL P95 | 2.117304 ± 0.016560 | 2.231616 ± 0.061252 |
| ITL P99 | 2.929078 ± 0.022085 | 3.036293 ± 0.082195 |
| ITL Max | 3.954643 ± 0.054969 | 4.087741 ± 0.042615 |
| 首次Block分配等待 Mean | 3.294947 ± 0.024496 | 3.284748 ± 0.070462 |
| 首次Block分配等待 P95 | 11.288903 ± 0.046280 | 11.611789 ± 0.208750 |
| 首次Block分配等待 Max | 11.505751 ± 0.050392 | 11.829226 ± 0.211670 |
| 大请求首次分配等待 Mean | 3.408265 ± 0.024224 | 3.504089 ± 0.073632 |
| 大请求首次分配等待 P95 | 11.291821 ± 0.046234 | 11.671796 ± 0.209431 |
| 大请求首次分配等待 Max | 11.505751 ± 0.050392 | 11.829226 ± 0.211670 |
| 小请求首次分配等待 Mean | 3.181629 ± 0.024788 | 3.065407 ± 0.067293 |
| 完成延迟 P95 | 13.950975 ± 0.132145 | 14.213141 ± 0.244195 |
| Output tok/s | 91.565929 ± 0.868493 | 89.889564 ± 1.539566 |
| 抢占次数 | 21 ± 0 | 22 ± 0 |
| 重算Token | 8,981 ± 0 | 9,238 ± 0 |
| 实际越过次数 | 0 ± 0 | 2 ± 0 |
| KV峰值Block | 105 ± 0 | 105 ± 0 |
| KV平均占用Block（步加权） | 80.337168 ± 0.192324 | 80.841769 ± 0.110054 |
| Decode平均Batch Size | 3.996610 ± 0 | 3.993220 ± 0 |
| 请求完成 / 输出Token | 80 / 1,280 | 80 / 1,280 |

全部指标的每次值、均值、SD、min/max保存在`bypass_analysis/cross_run_statistics.csv`。
以下显式保留全部六次关键值，避免只展示最好重复：

| repeat | 策略 | 首次等待Mean | 首次等待P95 | 大请求等待Mean | TTFT P95 | ITL P95 | ITL P99 | 延迟P95 | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | FIFO | 3.323148 | 11.282433 | 3.436077 | 11.342145 | 2.110108 | 2.947483 | 13.893600 | 91.931376 |
| 0 | Bypass | 3.231936 | 11.456774 | 3.448862 | 11.489298 | 2.194469 | 2.977643 | 14.028119 | 91.062261 |
| 1 | FIFO | 3.278956 | 11.246198 | 3.391773 | 11.311506 | 2.105560 | 2.935163 | 13.857215 | 92.191981 |
| 1 | Bypass | 3.257552 | 11.529439 | 3.475719 | 11.561437 | 2.198066 | 3.000994 | 14.121379 | 90.460302 |
| 2 | FIFO | 3.282737 | 11.338078 | 3.396945 | 11.411850 | 2.136245 | 2.904588 | 14.102109 | 90.574431 |
| 2 | Bypass | 3.364755 | 11.849154 | 3.587686 | 11.882923 | 2.302313 | 3.130241 | 14.489925 | 88.146128 |

跨次均值变化：整体首次等待Mean -0.31%，P95 +2.86%；大请求等待Mean +2.81%、
P95 +3.37%；TTFT Mean -0.28%、P95 +2.55%；ITL P95 +5.40%、P99 +3.66%；
吞吐 -1.83%。这些是本次观测的算术比较，不是统计显著性或普遍因果结论。

## 真实越过、受益与公平性

三次Bypass轨迹的事件ID和步骤一致：

| step（从0开始） | 被越过队首 | 候选 | free检查前 | 队首真实新增Block | 候选真实新增Block | 分配后free |
|---|---:|---:|---:|---:|---:|---:|
| 96 | 65（长） | 66（短） | 9 | 2 | 1 | 8 |
| 381 | 75（长） | 76（短） | 9 | 2 | 1 | 8 |

两次均为物理容量足够、但队首`9-2<8`，候选`9-1>=8`；不是绕过物理容量失败。
每个被越过请求恰好一次，后续即使又有可放入的小请求也不能继续越过该队首。

逐请求配对的三次均值如下，差值=Bypass-FIFO：

| request_id | 角色 | FIFO首次等待 | Bypass首次等待 | 差值 |
|---|---|---:|---:|---:|
| 65 | 被越过长请求 | 10.784020 | 11.135677 | +0.351657 |
| 66 | 提前接纳短请求 | 10.846041 | 3.114257 | -7.731784 |
| 75 | 被越过长请求 | 11.285488 | 11.668514 | +0.383025 |
| 76 | 被提升短请求 | 11.353777 | 11.608804 | +0.255026 |

76相对其本次队首更早接纳，但与原FIFO的实际等待相比反而更久；局部改变反馈到整个
执行轨迹后，不保证每个被提升请求都比FIFO快。旧离线分析的133个机会是独立局部探针，
不能当作133次可累积收益或推导这次真实事件应有三个候选。

全部六次完成率100%，每次80请求、1,280输出，未发现有限测试中的未完成/饥饿请求。
每次等待超过10秒的大请求均为8个。此阈值仅作描述，不把长等待等同于饥饿。
额度上限防止同一未接纳请求被无限重复绕过；它不限制其他资源等待、已有跨阶段策略的
等待，也不构成持续在线新请求下的严格Wall Time上界。

## 抢占、重算及瓶颈

FIFO三次均175轮Prefill、295轮Decode、21次抢占；Bypass三次均177轮Prefill、295轮Decode、
22次抢占。Decode批次仍接近满载，不能用“填大Decode Batch”解释收益。
reserve拒绝尝试从135变140，涉及请求从3变4；这与成功越过次数不同，尝试还包括候选检查。

FIFO丢失有效KV共8,981 Token、恢复缓存命中0；Bypass丢失9,494 Token、恢复Prefix命中256，
实际重算为`9,494-256=9,238`。全Prefix复用Token从256增至512，但更多复用没有抵消
资源竞争带来的额外丢失。额外重算257 Token（+2.86%）和一次抢占并非所有Prefill都算重算。
恢复Prefill的其余21/22 Token是尚未喂回模型的新输出，不计入重算。

两组都达到105 Block峰值，平均占用略升；资源完整性逐步核验，末尾全部105 Block空闲、
ref_count_sum=0、waiting/running均空。Prefix哈希可保留不代表资源泄漏。
提前接纳66消耗一个Block并改变RR参与者和后续扩页/受害者序列，观察到更重的竞争链。
这些计数支持“绕过并未解除本负载的容量瓶颈”；没有单独测量反事实纯抢占边际耗时，
不能把总耗时差精确归因于那一个新增抢占或257重算Token。

## 复现、原始数据与完整性

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.kv_pressure_bench \
  --workload-from results_kv_admission_20261102/manifest.json \
  --expected-blocks 105 --pressures high --policies B8 B8_bypass \
  --repeats 3 --output-dir results_kv_bypass_20261102
.venv/bin/python -m benchmarks.kv_pressure_bench --verify results_kv_bypass_20261102
.venv/bin/python -m benchmarks.kv_bypass_analysis --results results_kv_bypass_20261102
.venv/bin/python -m benchmarks.kv_bypass_analysis --results results_kv_bypass_20261102 --verify
git diff --check
```

输出目录必须为空，复跑请换新目录。原始数据保存在项目根目录`results_kv_bypass_20261102/`：

- `manifest.json`：完整80请求Token IDs、配置、来源、源码hash、实验顺序规则；
  `runs/`：六份独立进程JSON及完整日志；源代码快照保留实际测量版本。
- `high/raw_requests.csv`：逐请求真实入队/分配/Token/完成时间和重算信息；
  `steps.csv`、`allocation_events.csv`、`preemption_events.csv`、
  `reserve_deferral_events.csv`、`reserve_wait_events.csv`、`bypass_events.csv`。
- `summary.csv`、`pressure_runs.json`、`cross_run_statistics.csv`：原始逐次汇总与跨次统计；
  `bypass_analysis/`：全体/大小请求分组、240条逐请求配对、逐指标配对差值和CSV/JSON校验。
- `input_integrity.json`：已有155个原始文件及80请求工作负载未更改、实际源码与快照匹配；
  `source_diff.patch`、CPU测试日志和兼容性检查保留在本地结果目录。

结果目录沿用现有Git忽略规则保存本地，不提交模型、wheel、.venv或大量原始结果。
未执行Commit/Push。既有stage6.3分析及用户已有其他修改保持保留。

## 限制与下一步

只有一块GPU、一个模型、一套固定同时到达的80请求负载和reserve=8；只做用户要求的高压B8。
三次不足以覆盖温度/功耗/时钟变化和尾分布，不能外推到其他长度、真实在线流量或不同容量。
计时前后温度45–51°C，SM时钟快照1635–2355MHz，显存时钟7001/8001MHz；快照不是全程
采样，轮换顺序也无法完全排除硬件漂移。第三次Bypass更慢需如实保留，不能挑前两次宣称成功。

同步单线程事件循环只能在step间隙注入请求；本工作负载计划到达均为0，不是生产异步在线服务。
平均KV占用使用每步调度后用量乘整步Wall Time的近似，不是Block每次分配/释放的连续时间积分；
还不覆盖step间隙。观察器的Python开销计入两组正式时间。

当前证据适合保留开关用于后续可控实验，**不足以支持切换默认策略或继续叠加复杂动态策略**。
若后续继续研究，应先扩大独立重复/负载覆盖、分解扩页竞争和大请求等待，再判断是否值得
结合接纳余量或受害者选择优化；本阶段不自动实现这些额外改动。
