# Long-Video / Video-LLM Papers — 深度方法解析 · NeurIPS 2025 · ICLR 2026 · CVPR 2026 · AAAI 2026

19 篇长视频理解 / Video-LLM 论文/项目的中文**深度方法解析**。与一般综述不同，每篇都尽量**逐个深入源码**，描述真实方法过程——以**一条真实训练/使用数据样本**走完整个训练或方法流程，并给出 `file:line` 级引用。

每篇文档（`papers/NN-*.md`）固定包含：
1. **源码可得性 / 结构** — 官方仓库、克隆情况、关键目录；
2. **数据来源与真实格式** — 真实样本字段（仓库内 JSON/jsonl 摘录）；
3. **完整方法 / 训练流程** — 含公式与 `file:line`；
4. **一条真实数据的全过程** — 单条样本从输入→模型→输出/奖励的真实流转；
5. **模型 / 组件**；
6. **创新点**。

对应源码镜像在 `src/NN-*/`（已剥离 `.git`、删除 >40M 大文件与可重新下载的第三方数据；无公开源码的项目保留空目录 + `NO_SOURCE.md` 说明）。带配图的早期综述版见 [`OVERVIEW.md`](OVERVIEW.md)。

> 文字为本仓库作者面向学术评审的原创解析；源码镜像版权归各原作者所有，许可证随仓库保留。无公开源码者基于论文正文/公式撰写并已在文首标注。

---

## 索引

| # | 项目 | 会议 | 类型 | 源码 | 说明文档 | 源码镜像 |
|---|---|---|---|---|---|---|
| 01 | **LongVILA-R1** — 长视频推理 RL 全栈训练 | NeurIPS 2025 | 训练/RL | ✅ | [papers/01](papers/01-longvila-r1.md) | [src/01](src/01-longvila-r1/) |
| 02 | **PerceptionLM** — 全开源图像/视频 VLM + 数据引擎 | NeurIPS 2025 | 模型/数据 | ✅ | [papers/02](papers/02-perceptionlm.md) | [src/02](src/02-perceptionlm/) |
| 03 | **MR. Video** — MapReduce agent 长视频 QA | NeurIPS 2025 | Agent | ✅ | [papers/03](papers/03-mr-video.md) | [src/03](src/03-mr-video/) |
| 04 | **VideoLucy** — 层级记忆深度记忆代理 | NeurIPS 2025 | Agent/记忆 | ✅ | [papers/04](papers/04-videolucy.md) | [src/04](src/04-videolucy/) |
| 05 | **Video-VER** — 视频事件推理 | NeurIPS 2025 | 方法 | ❌ 无源码 | [papers/05](papers/05-video-ver.md) | [src/05](src/05-video-ver/) |
| 06 | **Video Contrastive Decoding** — 对比解码提升时序推理 | NeurIPS 2025 | 解码 | ❌ 无源码 | [papers/06](papers/06-video-contrastive-decoding.md) | [src/06](src/06-video-contrastive-decoding/) |
| 07 | **SAMA** — 指代理解+视觉接地+多轮对话 | NeurIPS 2025 | 模型/数据 | ✅ | [papers/07](papers/07-sama.md) | [src/07](src/07-sama/) |
| 08 | **ScaleLong** — 同视频多时间尺度基准 | ICLR 2026 | Benchmark | ✅ | [papers/08](papers/08-scalelong.md) | [src/08](src/08-scalelong/) |
| 09 | **VideoReasonBench** — latent-state 视频推理基准 | ICLR 2026 | Benchmark | ✅ | [papers/09](papers/09-videoreasonbench.md) | [src/09](src/09-videoreasonbench/) |
| 10 | **M3-Agent** — 实体中心多模态长时记忆图 + DAPO | ICLR 2026 | Agent/记忆/RL | ✅ | [papers/10](papers/10-m3-agent.md) | [src/10](src/10-m3-agent/) |
| 11 | **Map the Flow** — VideoLLM 时序推理信息流机制解剖 | ICLR 2026 | 可解释性 | ✅ | [papers/11](papers/11-map-the-flow.md) | [src/11](src/11-map-the-flow/) |
| 12 | **Let's Split Up** — 零样本分类器权重编辑做细粒度理解 | ICLR 2026 | 方法/数据 | ✅ | [papers/12](papers/12-category-splitting.md) | [src/12](src/12-category-splitting/) |
| 13 | **WorldMM** — 多模态三类记忆 + 自适应检索 agent | CVPR 2026 Highlight | Agent/记忆 | ✅ | [papers/13](papers/13-worldmm.md) | [src/13](src/13-worldmm/) |
| 14 | **StreamReady** — 流式 VQA 学习"何时作答" | CVPR 2026 | 方法/Benchmark | ❌ 无源码 | [papers/14](papers/14-streamready.md) | [src/14](src/14-streamready/) |
| 15 | **TimeLens** — VTG 数据质量 + 交错时间戳 + thinking-free RLVR | CVPR 2026 | 训练/RL/数据 | ✅ | [papers/15](papers/15-timelens.md) | [src/15](src/15-timelens/) |
| 16 | **VideoAuto-R1** — Thinking Once, Answering Twice 自适应推理 RL | CVPR 2026 | 训练/RL | ✅ | [papers/16](papers/16-videoauto-r1.md) | [src/16](src/16-videoauto-r1/) |
| 17 | **TrackMAE** — 运动感知视频掩码自编码器 | CVPR 2026 | 自监督 | ⚠️ 占位 | [papers/17](papers/17-trackmae.md) | [src/17](src/17-trackmae/) |
| 18 | **IPFormer-VideoLLM** — 多镜头场景实例提示视频理解 | — | 模型/数据 | ❌ 无源码 | [papers/18](papers/18-ipformer-videollm.md) | [src/18](src/18-ipformer-videollm/) |
| 19 | **APVR** — 免训练小时级长视频双级检索 | AAAI 2026 | Training-free | ❌ 无源码 | [papers/19](papers/19-apvr.md) | [src/19](src/19-apvr/) |

源码列：✅ 官方源码已镜像；⚠️ 官方仓库仅占位（实现未发布）；❌ 无公开源码（说明文档基于论文/公式）。

---

## 按方法类型横切的几条主线

- **强化学习训练（RLVR / GRPO / DAPO）**：LongVILA-R1（MR-SP 序列并行 + embedding 缓存）、TimeLens（thinking-free、IoU reward、beta=0 无 reference）、VideoAuto-R1（双 boxed 双奖励 + 置信度早退）、M3-Agent（DAPO 多轮检索，reward=GPT-4o 判对错）。共性是把"答对"或"IoU"做成可验证奖励、组内归一优势。
- **记忆 / Agent 长视频**：MR. Video（MapReduce）、VideoLucy（层级记忆）、M3-Agent（实体中心记忆图）、WorldMM（episodic/semantic/visual 三类记忆 + 自适应检索 agent）。趋势是从单一文本记忆走向多模态、多时间粒度、可迭代检索。
- **免训练 / 解码侧**：Video Contrastive Decoding（时间扭曲负分支对比解码）、APVR（帧级 + token 级双级检索，即插即用）。无需重训即增强时序推理或扩展可处理帧数。
- **机制理解与数据质量**：Map the Flow（attention knockout 解剖时序信息流）、TimeLens / Let's Split Up（揭示并修复标注质量、零样本权重编辑）。把"为什么有效""数据对不对"作为一等问题。
- **基准（Benchmark）**：ScaleLong（同视频多时间尺度、U 型曲线）、VideoReasonBench（latent-state + 部分可见 + 可执行性仿真判分）、StreamReady/ProReady-QA（带证据窗的主动式流式 QA）。
