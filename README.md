# Long-Video / Video-LLM Papers Summary — NeurIPS 2025 · ICLR 2026 · CVPR 2026 · AAAI 2026

19 篇长视频理解 / Video-LLM 论文的中文方法总结，每篇含：
- **方法**（用具体例子讲清楚 pipeline 流程）
- **架构图**（直接引用作者项目主页 / arxiv / GitHub README 的原图 URL，归属保留在源站）
- **创新点**（相对 SOTA 新在哪里）

最后附按"方法类型"横切的五条主线小结。

> 关于图片：本仓库不分发任何论文图片二进制，全部通过 `<img src=>` 引用作者**官方项目主页 / arxiv HTML / GitHub README** 的原始 URL。每张图下方标注了来源页面与 Figure 编号，版权归各论文作者所有。文字总结为本仓库作者的原创学术评论。

> Summaries are original commentary written for academic review purposes. All figures are loaded via direct URL reference from the authors' official project pages / arxiv HTML pages / GitHub READMEs; no binaries are redistributed in this repository. Per-figure attribution links to original sources are provided.

---

## NeurIPS 2025

### 1. Scaling RL to Long Videos (LongVILA-R1)
**方法**：NVIDIA 提出长视频推理的全栈训练框架，包含数据集 LongVideo-Reason（104K QA）、两阶段训练（CoT-SFT + RL）和 MR-SP 并行系统。举例：给一段足球比赛长视频问"为什么这粒进球被判无效？"，系统先用 NVILA-8B 抽视频 embedding 缓存进 vLLM 引擎，CoT-SFT 阶段先让模型学会写"先看越位线 → 再看传球瞬间 → 再判断"的思维链；进入 RL 阶段后，按 MR-SP 把视觉序列在多 GPU 间切片并行编码、LLM prefill 复用缓存，rollout 出多条推理路径用规则奖励打分更新模型。

<img src="https://arxiv.org/html/2507.07966v1/x5.png" alt="LongVILA-R1 training pipeline" width="100%"/>

> *Figure source*: [arXiv 2507.07966 Figure 5](https://arxiv.org/abs/2507.07966) — © original authors. CoT-SFT 阶段 + MR-SP RL 阶段两阶段训练流程。

**创新点**：1) 首个面向 hour-long 视频的 RL 训练栈，单 8×A100 节点支持 3600 帧 / 256K token；2) MR-SP 引入序列并行 + 视频 embedding 缓存，长视频 RL 训练提速 2.1×；3) 配套 104K reasoning 标注数据，VideoMME 带字幕 71.1%。

### 2. PerceptionLM
**方法**：Meta FAIR 推出的全开源图像/视频理解 VLM（<8B LLM decoder + Perception Encoder），重点在数据引擎而非新架构。举例：给一段 30 秒做菜视频生成细粒度描述，先用 scene detector 切片，再抽关键帧让 Llama 3 写 frame-level caption，初版 PLM 写 video-level caption，最后用 LLM 把帧描述 + 已有元数据融成连贯的细节级 caption；时空定位类问题则由 2.8M 人工标注的细粒度 video QA / grounded caption 数据训练得到。

<img src="https://raw.githubusercontent.com/facebookresearch/perception_models/main/apps/plm/docs/plm_main_fig.png" alt="PerceptionLM architecture" width="100%"/>

> *Figure source*: [facebookresearch/perception_models GitHub README](https://github.com/facebookresearch/perception_models) — © Meta FAIR. Perception Encoder + LLM decoder 主架构及训练数据 pipeline。

**创新点**：1) 完全可复现、不依赖闭源教师模型蒸馏的开放训练食谱；2) 系统识别并补齐细粒度视频理解的数据缺口，发布 2.8M 人工标注 video QA + 时空 grounded 数据；3) 同步发布 Perception Encoder，组合成端到端可复用栈。

### 3. MR. Video
**方法**：把 MapReduce 思路套到长视频 QA 的 agent 框架。举例：问"主角在电影里和谁吵了几次架？"，Map 阶段把长视频切成短段，VLM（Gemini-2.0-Flash）独立给每段写 caption 并行处理；Reduce 阶段用 LLM（GPT-4o）把各段里的不同人名/物体名归一化。第二轮 MapReduce：针对用户问题对每个短段抽取相关信息（Map），再聚合推断出总答案（Reduce）。整个流程不需训练，纯 agent 编排。

<img src="https://raw.githubusercontent.com/ziqipang/MR-Video/main/assets/overview.png" alt="MR. Video MapReduce framework" width="100%"/>

> *Figure source*: [ziqipang/MR-Video GitHub README](https://github.com/ziqipang/MR-Video) — © original authors. Map 短段并行密集理解 + Reduce 全局推理 agent pipeline。

**创新点**：1) 用 MapReduce 取代 sequence-to-sequence VLM 和顺序选关键帧的 video agent，可并行扩展、不受 context 长度限制；2) 显式提出跨段角色/物体名一致化 + 问题意图分析；3) LVBench 60.8%，比此前 VLM 和 video agent 高 10% 以上。

### 4. VideoLucy
**方法**：受电影《Lucy》启发，做"从粗到细"的层级记忆回溯。举例：问"主角刚才把钥匙放哪了？"，VideoLucy 用 Qwen2.5-VL-7B 先以最粗粒度扫一遍生成顶层记忆；DeepSeek-R1 作为 agent 判断信息不足，则把时间窗收缩到与"钥匙"相关的片段，下一层用更密采样、更细 caption 抓取连续帧的时间上下文；如此迭代深入。

<img src="https://videolucy.github.io/static/images/Fig1.png" alt="VideoLucy pipeline" width="100%"/>

> *Figure source*: [videolucy.github.io Figure 1](https://videolucy.github.io/) — © original authors. Deep memory 多级 backtracking 框架。

**创新点**：1) 提出层级记忆 + 迭代 backtracking 机制，专门解决 sparse sampling 丢关键信息和单帧建模缺时间上下文两大痛点；2) 时间尺度与细节粒度随深度动态联动；3) 发布 EgoMem 长视频基准。

### 5. When Thinking Drifts (Video-VER)
**方法**：先诊断再开药方。诊断阶段在 MVBench 20 个子任务上系统性证明：让 video 模型"边想边答"反而降准确率，因为 CoT token 训练时从未被监督，模型容易生成视觉无依据的"幻觉思维链"。药方阶段提出 Visual Evidence Reward（VER），用 GRPO 强化学习：模型生成的 CoT 若显式引用并落地到视觉证据则给奖励。例：问"接下来会发生什么"，模型必须先输出"在 12s 处看到杯子在桌沿、手伸过去"再推理出"杯子会掉"，才能拿分。

<img src="https://arxiv.org/html/2510.06077v1/figs/visual_evidence_generation.png" alt="Video-VER VER" width="100%"/>

> *Figure source*: [arXiv 2510.06077](https://arxiv.org/abs/2510.06077) — © original authors. Visual evidence 生成 + Visual Evidence Reward 训练流程。

**创新点**：1) 首次系统性识别并用贝叶斯框架解释视频 CoT 的"思维漂移"现象；2) 提出 VER 作为对 CoT token 的显式视觉证据监督信号；3) 在 10 个视频理解 benchmark 上稳居 top。

### 6. Improve Temporal Reasoning via Video Contrastive Decoding
**方法**：训练-free 的解码端策略。以"一个人先拿起杯子再放下"这种连续动作为例：原始视频送入 VideoLLM 得到 logits p(正向)；同时把关键帧的时间顺序打乱/扭曲（破坏时间一致性），再送入同一模型得到 logits p(扭曲)，这个分支会输出"时间不敏感"的错误回答（比如把动作顺序说反）。最后在 token 层做对比解码 p(正向) − α·p(扭曲)，把依赖 language/image prior 的错误分布"减掉"。

> ⚠️ 该论文目前仅有 [OpenReview PDF](https://openreview.net/forum?id=2nIAtsUC27)，未发布 arxiv HTML / 项目主页 / GitHub，因此暂时拿不到可外链的原图。建议直接打开 PDF 看 Figure 1 / Figure 2。

**创新点**：1) 首次从 language prior 与 image prior 失败的视角解释 VideoLLM 时间推理短板；2) 不训练、不加数据、不动模型权重，仅在解码阶段做一次扭曲分支前传即可即插即用；3) 用"扭曲时间一致性"作为负样本构造，对比对象比此前 frame-drop 类方法更针对时间因果。

### 7. SAMA
**方法**：把"指代理解 + 分割 grounding + 多轮对话"放进同一模型。例：用户在一段球赛视频第 5 秒框出一个球员问"他刚才传球给谁了?"，SAMA 先用时空上下文聚合器把被框区域跨帧的视觉特征汇聚成 referent token，LLM 据此生成回答；下一轮"把 7 号在整段视频里画出来"，LLM 输出 grounding token 触发内置的 SAM 分支，在所有相关帧上输出 mask。

<img src="https://raw.githubusercontent.com/sunye23/SAMA/main/resources/sama_teaser.png" alt="SAMA overview" width="100%"/>

> *Figure source*: [sunye23/SAMA GitHub README](https://github.com/sunye23/SAMA) — © original authors. SAMA-239K 数据 + 时空上下文聚合器 + SAM 联合的 referential grounded video chat pipeline。

**创新点**：1) 构建 SAMA-239K（15K 视频、17 万 referential grounded QA），首次把 referring/grounding/多轮 chat 联合监督；2) 用统一 spatio-temporal context aggregator + SAM 解耦"理解"与"像素级 grounding"；3) 提出 SAMA-Bench 填补 benchmark 空白。

---

## ICLR 2026

### 8. ScaleLong（benchmark）
**方法**：核心做法是"同一段长视频里同时塞入跨四个时间尺度的问题"。例：一段 90 分钟的烹饪纪录片，clip 级问"第 12 分钟厨师左手拿的是什么?"（秒级），shot 级问"切洋葱这段镜头持续多久?"（十秒级），event 级问"准备主菜共用了哪些食材?"（分钟级），story 级问"整集主题想表达什么家庭情感?"（小时级）。269 个平均 86 分钟的视频，5 大类 36 子类。

<img src="https://raw.githubusercontent.com/multimodal-art-projection/ScaleLong/main/imgs/LongVideoBench.png" alt="ScaleLong benchmark" width="100%"/>

> *Figure source*: [multimodal-art-projection/ScaleLong GitHub README](https://github.com/multimodal-art-projection/ScaleLong) — © original authors. Clip/Shot/Event/Story 四层时间尺度问答任务构造。

**创新点**：1) 首个 within-content 多时间尺度评测；2) 揭示 23 个 MLLM 普遍呈 U 型曲线，最短和最长尺度好、中间 event 级最差；3) 系统性消融表明视觉 token 容量在所有尺度上都正相关。

### 9. VideoReasonBench（benchmark）
**方法**：刻意构造"非看视频做不出来"的推理题。代表例子：滑动数字华容道视频——开头展示初始 3×3 数字盘面，中间播放一连串滑动操作（latent state 仅部分可见），三档难度问题：L1 回忆"第 7 步把哪格滑到了哪儿"；L2 推断"第 15 步结束时盘面是什么样"；L3 预测"再做这套操作 5 步后会是什么样"。

<img src="https://raw.githubusercontent.com/llyx97/video_reason_bench/main/assets/overview.png" alt="VideoReasonBench overview" width="100%"/>

> *Figure source*: [llyx97/video_reason_bench GitHub README](https://github.com/llyx97/video_reason_bench) — © original authors. 6 类视频（Number/Circle/Cup/File/Card/Chip）+ 3 级推理任务（Recall/Infer/Predict）。

**创新点**：1) 首个 vision-centric 视频推理 benchmark；2) 实证了 long-CoT/thinking budget 在视频域的价值：Gemini-2.5-Flash 要消耗 7000+ token 才达 27.4%；3) 三层难度阶梯对 test-time scaling 研究友好。

### 10. Seeing, Listening, Remembering, and Reasoning（M3-Agent）
**方法**：双进程 agent。记忆进程：持续接收视频+音频流，在线抽取实体（人脸、声纹、物体），构建 entity-centric 多模态记忆图——节点是实体，episodic 记忆挂"什么时候在哪发生了什么"，semantic 记忆挂"这个人叫张三、爱喝美式"。控制进程：用户问"张三昨天把外套放哪了?"agent 用 DAPO（RL 训练）做多轮迭代检索。

<img src="https://m3-agent.github.io/static/images/illustration.png" alt="M3-Agent architecture" width="100%"/>

> *Figure source*: [m3-agent.github.io](https://m3-agent.github.io/) — © original authors. 实时视听输入 → entity-centric episodic+semantic 长期记忆构建 → 多轮迭代推理检索。

**创新点**：1) 把跨模态身份一致性显式建模为 entity-centric 多模态图节点；2) episodic + semantic 双层长期记忆，去掉 semantic 掉 13–19%；3) 用 DAPO 强化学习训练多轮检索-推理策略，M3-Bench-robot/web、VideoMME-long 分别 +8.2/+7.7/+5.3。

### 11. Map the Flow（机制可解释性）
**方法**：用机制可解释性工具去"解剖" VideoLLM 在 VideoQA 中的内部信息流。例子：给 LLaVA-NeXT-7B-Video-FT 喂视频 + 时序问题，作者逐层探测 attention edges——发现早到中间层先做 cross-frame 互动，中间层把视频表征对齐到包含"先/后/之前"等时序概念的语言 embedding，中后层才生成答案。在前一半层屏蔽 cross-frame attention 模型立刻给出错；保留约 42% 关键 edges 即可保持原始性能。

<img src="https://map-the-flow.github.io/static/images/teaser.jpg" alt="Map the Flow teaser" width="100%"/>

> *Figure source*: [map-the-flow.github.io](https://map-the-flow.github.io/) — © original authors. (a) 早-中层 cross-frame 交互 → 时序词对齐 → 信息流向最后 token → 答案生成；(b) Attention Knockout；(c) 各层答案概率曲线。

**创新点**：1) 首个系统性刻画 VideoLLM 时序推理"三阶段信息通路"；2) 提出可量化的因果干预实验范式；3) 用稀疏化结果说明现有 VideoLLM 大量 attention 是冗余的。

### 12. Let's Split Up
**方法**：提出新任务 "category splitting"——把已训练好的视频分类器中一个粗类拆成多个细类，不重训、不要新标签。例子：原分类器只有 "open" 类，想拆成 "open cupboard / open by pushing / open quickly / open halfway"。方法零样本地利用视频分类器内部 latent 的组合结构（动作 = 对象 × 方式 × 速度 × 结果），把粗类 head 的权重沿这些语义方向编辑成多个细分子 head。

<img src="https://kaitingliu.github.io/Category-Splitting/static/images/zero_shot_v3.jpg" alt="Let's Split Up zero-shot" width="100%"/>

> *Figure source*: [kaitingliu.github.io/Category-Splitting](https://kaitingliu.github.io/Category-Splitting/) — © original authors. 组合式权重编辑：粗类权重 + modifier 向量 → 多个细分子类权重。

**创新点**：1) 提出 category splitting 新任务设置；2) 零样本编辑方法不需要任何新视频样本，明显优于 CLIP 类 baseline；3) 证明 zero-shot edits 是 low-shot 微调的好初始化。

---

## CVPR 2026

### 13. WorldMM
**方法**：把长视频（小时到天级）当作 agent 的外部记忆库。例子：给一段一周长的监控视频问"周三下午那个人为什么搬走了红箱子"。Stage 1 multimodal memory construction：分段构建三类记忆——episodic（多时间尺度事件索引）、semantic（知识图谱式抽象）、visual（保留 raw 视觉细节）。Stage 2 adaptive retrieval：retrieval agent 迭代选 memory 源和时间粒度。Stage 3 response generation。

<img src="https://worldmm.github.io/assets/fig/method.webp" alt="WorldMM method" width="100%"/>

> *Figure source*: [worldmm.github.io](https://worldmm.github.io/) — © original authors. Episodic + semantic + visual 三类记忆 + 自适应多模态检索 agent。

**创新点**：1) 三种记忆共存，弥补前作纯文本摘要导致视觉细节丢失的问题；2) 多时间尺度索引 + 自适应迭代检索，打破固定时间窗口的局限；3) 在五个 hour- 到 week-long QA benchmark 上平均提升 8.4%。

### 14. StreamReady
**方法**：解决 streaming VQA 中"何时回答"的问题——问题可能先于证据出现，太早答是幻觉、太晚答失去实时价值。例子：直播流里用户问"那辆红车撞上了吗"，证据画面要 20 秒后才出现。模型边收帧边运行一个轻量 readiness module，每步输出 Answer Readiness Score (ARS)；ARS 低则继续观察并保持沉默，达到阈值（证据窗口出现）立刻输出答案；ARS 用非对称损失训练——早答惩罚重于晚答。

<img src="https://arxiv.org/html/2603.08620v1/images/cvpr26-framework_last.jpg" alt="StreamReady framework" width="100%"/>

> *Figure source*: [arXiv 2603.08620 Figure 2](https://arxiv.org/abs/2603.08620) — © original authors. Visual memory tree + 短期/长期分支推理 + 可学习 `<RDY>` token & readiness head 决定何时回答。

**创新点**：1) 首次把"何时答" formalize 为 ARS 这个 timing-aware 指标；2) readiness gating 模块作为轻量 add-on 把 temporal reasoning 与 on-time answering 统一在一个框架；3) ProReady-QA 提供带证据窗口和 proactive 多轮的新评测。

### 15. TimeLens
**方法**：从"数据质量"和"时间戳编码"两条线重做视频时间定位 (VTG)。例子：训练时给 Qwen2.5-VL-7B 喂视频 + query "找到主角第一次出现的时间段"。数据侧用 Gemini-2.5-Pro 自动重标 Charades-STA / ActivityNet 等老数据得到 TimeLens-100K；评测则手工精修出 TimeLens-Bench。模型侧采用 interleaved textual timestamp encoding——直接把时间戳作为 text token 与帧 token 交错。训练用 thinking-free RLVR（IoU 作为 reward）。

<img src="https://timelens-arc-lab.github.io/static/images/teaser_v2-1.png" alt="TimeLens teaser" width="100%"/>

> *Figure source*: [timelens-arc-lab.github.io](https://timelens-arc-lab.github.io/) — © original authors. 数据质量轴（TimeLens-Bench / TimeLens-100K）+ 算法设计轴（interleaved textual timestamp + 训练 recipe）。

**创新点**：1) 揭示并修复了主流 VTG benchmark 的标注质量问题；2) 系统对比并提出 interleaved textual timestamp encoding 替代位置嵌入路线；3) thinking-free RLVR 配方让 TimeLens-3B 反超 Qwen2.5-VL-7B，开源模型整体超过 GPT-5 / Gemini-2.5-Flash。

### 16. VideoAuto-R1
**方法**：训练时采用 "Thinking Once, Answering Twice" 范式——模型先对视频问题给出初始答案，然后做一次 CoT 推理，最后输出复核答案，两次答案都用可验证奖励监督。推理时用规则化早退策略：模型先输出初始答案，再计算其 token 的长度归一化平均 log 概率作为置信度，若超过阈值就直接结束、跳过 CoT；只有难题才真的展开思考链。

<img src="https://ivul-kaust.github.io/projects/videoauto-r1/static/images/method.png" alt="VideoAuto-R1 method" width="100%"/>

> *Figure source*: [ivul-kaust.github.io/projects/videoauto-r1](https://ivul-kaust.github.io/projects/videoauto-r1/) — © original authors. "答-想-再答" 单模型训练范式 + 推理时基于置信度的早退机制。

**创新点**：1) 首次系统指出 RL 训练的 video 模型上 direct answering 常常持平甚至超过 CoT；2) 单一模型同时学"答-想-再答"，无需 mode-switch token；3) 在 video QA / grounding 达 SOTA 同时平均回复长度压缩约 3.3 倍（149→44 tokens）。

### 17. TrackMAE
**方法**：在标准 masked video modeling 之上显式注入运动轨迹监督。流程：输入视频→用现成 point tracker（如 CoTracker）稀疏跟踪若干点→根据轨迹设计 motion-aware tube masking（运动剧烈区域优先 mask）→ViT 编码可见 patch→解码器同时重建被 mask 的像素/特征，和被 mask 点的未来轨迹位置。例如打篮球片段，模型不仅要补回被遮的球员 patch，还要预测篮球点的运动轨迹。

<img src="https://arxiv.org/html/2603.27268v1/x2.png" alt="TrackMAE overview" width="100%"/>

> *Figure source*: [arXiv 2603.27268 Figure 2](https://arxiv.org/abs/2603.27268) — © original authors. 双分支 mask-and-predict：下分支 ViT encoder + spatial decoder 重建特征，上分支 CoTracker3 提取轨迹 + motion decoder 预测轨迹。

**创新点**：1) 把 motion 从 MVM 的隐式信号升级成显式重建目标；2) 用轨迹反向指导 mask 策略，让 mask 集中在真正"动"的区域；3) 在六个下游数据集上一致优于 VideoMAE / MotionMAE 等 SOTA 自监督基线。

---

## AAAI 2026

### 18. IPFormer-VideoLLM
**方法**：针对多镜头场景下 VideoLLM 出现 instance identity 遗忘和短暂出场人物被淹没的问题。流程：输入多镜头视频→sliding window 切片→每片过 visual encoder 得到 frame token→用检测器/跟踪器得到每个 instance 的 bbox→bbox 区域内特征池化聚合得到 instance-level 特征向量（Instance Prompts，IP）→IP Token Generator 把 IP 注入 Q-Former 风格 connector 的 query→送入 LLM 生成答案。

<img src="https://arxiv.org/html/2506.21116v1/x5.png" alt="IPFormer pipeline" width="100%"/>

> *Figure source*: [arXiv 2506.21116 Figure 5](https://arxiv.org/abs/2506.21116) — © original authors. Sliding-window 采样 → video encoder → frame-level (cls+avg pool) + instance-level (IP Token Generator) → 拼接注入 learnable query → attention 融合 → MLP → LLM。

**创新点**：1) 首次把 instance-level 特征作为 visual prompt 显式注入 connector；2) 提出 MultiClip-Bench 填补多镜头 dense caption + QA 标注空缺；3) 在镜头切换、实例追溯指标上显著超越 Video-LLaVA。

### 19. APVR
**方法**：训练免费（plug-and-play）的小时级长视频理解框架，做帧-token 双粒度自适应检索。流程：输入 1 小时视频和问题（例如"主角在厨房第一次拿起刀是几点几分？"）→Pivot Frame Retrieval (PFR) 先做 query expansion，再用迭代 spatio-semantic 置信度打分挑出候选帧，并用 temporal diffusion 保证时间连贯、adaptive resampling 精炼帧选择→选出的帧送进 MLLM 后，Pivot Token Retrieval (PTR) 用 query-aware 多层注意力打分，按 dynamic chunk-wise selection + head-wise soft voting 在 KV cache 里只保留最相关的视觉 token。

<img src="https://arxiv.org/html/2506.04953v1/extracted/6515668/fig2_framework.jpg" alt="APVR framework" width="100%"/>

> *Figure source*: [arXiv 2506.04953 Figure 1](https://arxiv.org/abs/2506.04953) — © original authors. Training-free 框架，PFR 与 PTR 两个 plug-and-play 组件嵌入 MLLM。

**创新点**：1) 同一框架内同时做帧级和 token 级两层检索；2) 完全 training-free，可即插即用嵌入现有 MLLM，绕过 memory wall 与注意力二次复杂度；3) 在 LongVideoBench 与 VideoMME 上不仅超越 training-free baseline，也压过部分需要训练的方法，达 SOTA。

---

## 整体规律小结

按"方法类型"横切，19 篇大致分成五条主线：

1. **训练新范式（RL / CoT / 自监督）**：LongVILA-R1、Video-VER、VideoAuto-R1、TrackMAE。共同信号：纯 text-CoT 经验搬不动视频，要么换奖励、要么换数据、要么显式建模 motion。

2. **Agent / 记忆系统（长视频 / 流视频）**：MR. Video、VideoLucy、M3-Agent、WorldMM、StreamReady。从静态长视频走向"持续观察 + 多模态记忆 + 主动决策"。

3. **训练-free 即插即用增强**：Video Contrastive Decoding、APVR。在不动模型的前提下榨现有 VideoLLM。

4. **细粒度 / 多镜头 / grounding**：SAMA、IPFormer、Let's Split Up、TimeLens、PerceptionLM。都在啃"具体到 instance / timestamp / 子类"的细颗粒度问题。

5. **Benchmark / 可解释性**：ScaleLong、VideoReasonBench、Map the Flow。评测维度从"长不长"走向"看得细不细 / 推得对不对 / 内部怎么算的"。

---

## Citation List

| # | Paper | Venue | Link |
|---|---|---|---|
| 1 | Scaling RL to Long Videos (LongVILA-R1) | NeurIPS 2025 | [arXiv 2507.07966](https://arxiv.org/abs/2507.07966) · [project](https://research.nvidia.com/labs/eai/publication/longrl/) |
| 2 | PerceptionLM | NeurIPS 2025 Spotlight | [arXiv 2504.13180](https://arxiv.org/abs/2504.13180) · [code](https://github.com/facebookresearch/perception_models) |
| 3 | MR. Video | NeurIPS 2025 | [arXiv 2504.16082](https://arxiv.org/abs/2504.16082) · [code](https://github.com/ziqipang/MR-Video) |
| 4 | VideoLucy | NeurIPS 2025 | [arXiv 2510.12422](https://arxiv.org/abs/2510.12422) · [project](https://videolucy.github.io/) |
| 5 | When Thinking Drifts (Video-VER) | NeurIPS 2025 | [arXiv 2510.06077](https://arxiv.org/abs/2510.06077) · [project](https://vision.cs.utexas.edu/projects/video-ver/) |
| 6 | Improve Temporal Reasoning via Video Contrastive Decoding | NeurIPS 2025 | [OpenReview](https://openreview.net/forum?id=2nIAtsUC27) |
| 7 | SAMA | NeurIPS 2025 | [arXiv 2505.18812](https://arxiv.org/abs/2505.18812) · [code](https://github.com/sunye23/SAMA) |
| 8 | ScaleLong | ICLR 2026 | [arXiv 2505.23922](https://arxiv.org/abs/2505.23922) · [code](https://github.com/multimodal-art-projection/ScaleLong) |
| 9 | VideoReasonBench | ICLR 2026 | [arXiv 2505.23359](https://arxiv.org/abs/2505.23359) · [code](https://github.com/llyx97/video_reason_bench) |
| 10 | Seeing, Listening, Remembering, and Reasoning (M3-Agent) | ICLR 2026 | [arXiv 2508.09736](https://arxiv.org/abs/2508.09736) · [project](https://m3-agent.github.io/) |
| 11 | Map the Flow | ICLR 2026 | [arXiv 2510.13251](https://arxiv.org/abs/2510.13251) · [project](https://map-the-flow.github.io/) |
| 12 | Let's Split Up | ICLR 2026 | [project](https://kaitingliu.github.io/Category-Splitting/) |
| 13 | WorldMM | CVPR 2026 Highlight | [arXiv 2512.02425](https://arxiv.org/abs/2512.02425) · [project](https://worldmm.github.io/) |
| 14 | StreamReady | CVPR 2026 | [arXiv 2603.08620](https://arxiv.org/abs/2603.08620) |
| 15 | TimeLens | CVPR 2026 | [project](https://timelens-arc-lab.github.io/) · [code](https://github.com/TencentARC/TimeLens) |
| 16 | VideoAuto-R1 | CVPR 2026 | [project](https://ivul-kaust.github.io/projects/videoauto-r1/) |
| 17 | TrackMAE | CVPR 2026 | [arXiv 2603.27268](https://arxiv.org/abs/2603.27268) |
| 18 | IPFormer-VideoLLM | AAAI 2026 | [arXiv 2506.21116](https://arxiv.org/abs/2506.21116) |
| 19 | APVR | AAAI 2026 | [arXiv 2506.04953](https://arxiv.org/abs/2506.04953) |

---

## License

- **文字总结 / Summaries (text)**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Free to share and adapt with attribution to this repository.
- **图片 / Figures**: 本仓库不包含图片二进制，通过 URL 引用作者官方页面。版权归原论文作者所有。Figures are *not redistributed* in this repository; each is referenced via direct URL to the authors' original publication page. All figure rights remain with original authors.

If you are an author and would prefer your figure not be embedded via external reference here, please open an issue and the reference will be removed.
