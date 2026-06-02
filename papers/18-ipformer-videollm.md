# 18 · IPFormer-VideoLLM（多镜头场景的实例提示视频理解）

> [arXiv 2506.21116](https://arxiv.org/abs/2506.21116) · 全称 *IPFormer-VideoLLM: Enhancing Multi-modal Video Understanding for Multi-shot Scenes*（Yujia Liang 等）

> **无公开源码**：GitHub 检索无同名/相关仓库，论文仅写 "The code and model will be available"，PapersWithCode 404。本文**全部基于论文 arXiv 2506.21116v2**，引用以 `论文行号`（抓取 HTML 纯文本）形式给出并附公式。

---

## 1. 源码可得性

无 repo（未公开）。可得资源：arXiv abs/HTML v2/PDF。`/home/user/lvp-src/ipformer` 未创建。

---

## 2. 数据来源与真实格式

**自建 MultiClip-Bench**（首个多镜头 multi-shot 视频数据集，含训练集 + 人工校验测试集）：
- **视频来源**（行58）：Kinetics-710、VideoChatGPT、VideoChat、YouCook2、NExT-QA、WebVid、EgoQA，过滤后约 6.7k 视频。
- **多镜头粗到精筛选管线**：先按时长粗筛（保留 **≤2 分钟**的短视频），再用 **OC-SORT** 做人物 ID 跟踪、保留 **ID switch >5** 的真·多镜头视频，最后用 **Gemini-1.5-Pro-flash** 按双标准（是否多镜头 + 是否含人物身份变化）精筛。
- **描述型标注**（行56-58）：每条含三要素——**keyframe description（带 person ID）**、**character information（人物特征/动作）**、**dense caption**；共 23k 视频-文本对。管线：PySceneDetect 选关键帧 → LLaVA-1.5 初描述 → Gemini 结合原视频精修（识别 person ID 变化与行为切换）→ Gemini 产 dense caption 与 character info。
- **QA 格式**（行59,65）：GPT-4 每视频生成 10 个 QA → 精修保留 4-6 个 → 加 3 个错误选项转**多选题**。四类问题：**consistency（一致性/重识别）、short-frame（短时段事件）、unexpected content、others**。训练集 **45.5k**、测试集 **2.75k**（无重叠）。
- **外部训练数据**（行400-401）：对齐阶段 LAION-CC-SBU 558K + Valley 702K；指令阶段 LLaVA-v1.5 665K + Video-ChatGPT 100K + VideoChat2 326K + Perception/STAR 27K + 自建 MultiClip 68K。

---

## 3. 完整方法 / 训练流程

### 整体（行60-70）
建立在 **Video-LLaVA** 之上：视觉编码器 → 视觉对齐模块（IP Token Generator + 注意力 + MLP）→ LLM。因编码器固定帧数限制，用**滑动窗口采样**：视频均匀抽 `F` 帧，按 8 帧无重叠滑窗得 `T=⌊F/8⌋` 个 slice，各 slice 独立过编码器与对齐模块，最后拼接送 LLM。

### 帧级信息 FT（行78）
编码器输出 `R∈ℝ^{8×N×D}`，`N=256+1`（256 image token + 1 cls）。对 256 image token avg-pool 得 `GF∈ℝ^{1×D}`。`C` 与 `GF` 各重复 `X=5` 次拼接：
> **FT = (C×X, GF×X) ∈ ℝ^{(X×2)×D}**（每帧 10 个 frame token）

### IP Token Generator（行81-84）— 核心
1. **检测**：category-agnostic 检测器（Deformable DETR，参数来自 Groma），分类头换成**二分类器**对 proposal 按定位质量打分，NMS 后每帧保留 `M(<10)` 个框，不足补零。
2. **抽实例特征**：对各帧 feature map 做 global RoI pooling。
3. **跨帧聚类（迭代贪心阈值聚类，论文行84）**：把 slice 内 8 帧检测到的所有实例特征跨帧归并到"同一物理实例"。论文描述的是基于余弦相似度的迭代分组，可还原为：
   - **逐轮播种**：从尚未分组的实例特征中取第一个作为新组的种子；
   - **吸纳相似项**：计算其余未分组特征与种子的**余弦相似度**，把所有 `sim > 0.9` 者并入该组（同一人/物在相邻帧的特征高度相似，故跨帧聚到一组）；
   - **多轮直至收敛**：剩余未分组特征重复"播种→吸纳"，直到所有实例都归入某组；离群（与任何种子都不够相似）的实例自成独立组；
   - **组内聚合**：每组对其成员特征**按 channel 求均值**，得到该实例的 **Instance Prompt**（一个 D 维向量代表"这一物理实例在本 slice 的统一表征"）。
   - 每 slice 实例 token 上限 `V=80`，不足补零。该阈值聚类的意义：把"同一人跨帧/跨镜头的多次检测"压成**一个** prompt（解决身份遗忘），同时让每个实例无论出现频次高低都只占一份额度——**平衡实例数量、避免高频实例淹没稀疏实例**，利于 short-frame 理解。
   > 注：本文无源码，上述迭代步骤是对论文"cosine 相似度阈值 0.9 迭代分组"文字描述的合理还原；精确的种子选择顺序/收敛判据论文未逐行给出，不臆造额外细节。

### 拼接 + 注入 query + cross-attention（行76-80）
受 Conditional-DETR 启发，把帧级与实例级信息作为 anchors **加到 learnable query**：实例 token `IP∈ℝ^{V×D}`，与 8 帧 FT 拼接得 `VT∈ℝ^{8×(X×2)×D + V×D}`（8×10 frame + 80 instance = 160 token/slice）。`VT` 加到 learnable queries 引导其在 **cross-attention**（query 作 Q，编码器视觉特征作 K/V）中聚合视频特征，输出经 **MLP** 投影对齐文本空间，再与文本拼接送 LLM。

### Token 压缩（行461-464）
每 slice 输出 **160** token，对比 full-projection 8×256=2048，降到基线 **<10%**；其中**第一（对齐）阶段训练时间减约 75%**，整体训练时间约减半。

### 训练（行399-401）
两阶段（沿用 Video-LLaVA）：① 模态对齐预训练，仅训视觉对齐模块（AdamW lr=1e-3，batch 256，1 epoch，8 帧@224）；② 指令微调，冻结视觉编码器、微调全部 LLM（batch 128，采样帧 8→16）。**损失**：标准自回归语言建模损失（next-token CE，对答案 token），检测器二分类用于打分定位质量。

---

## 4. 一条真实数据的全过程

以多镜头视频 short-frame QA 为例：
1. **抽帧/滑窗**：均匀抽 `F=16` 帧 → 8 帧滑窗切 `T=2` 个 slice。
2. **编码**：每 slice 8 帧@224 进 LanguageBind(OpenCLIP-L/14) → `R∈ℝ^{8×257×D}`。
3. **帧级 FT**：256 image token avg-pool 得 GF；C 与 GF 各 ×5 → 每帧 10 个 frame token。
4. **实例级 IP**：检测器每帧出 <10 框 → RoI pooling 取实例特征 → slice 内余弦 0.9 聚类（同一人跨帧/跨镜头归一组）→ channel 均值得 Instance Prompt，补零到 80。显式聚合"跨场景同一人物"，解决身份遗忘。
5. **注入 + 压缩**：`VT`(8×10+80=160) 加到 learnable query，cross-attention 聚合 → MLP 投影。每 slice 160 token，2 slice 约 320。
6. **LLM**：两段压缩视觉特征 + 文本问题/选项 token 送 Vicuna-7B。
7. **输出**：LLM 自回归生成多选答案。实例提示保留短时出现人物身份，能正确做人物重识别与短帧事件作答。

---

## 5. 模型 / 组件（行397-398）

- **Base VideoLLM**：Video-LLaVA（基线，相同训练设置）。
- **视觉编码器**：LanguageBind（初始化自 **OpenCLIP-L/14**，224×224）。
- **LLM**：**Vicuna-7B v1.5**；tokenizer 来自 LLaMA。
- **视觉压缩/注意力模块**：参考 BLIP-2 / VideoChat2，用预训练 **BERT-base** 实现注意力压缩；另有两层 MLP(GeLU) full projection。
- **检测器（IP Token Generator）**：**Deformable DETR**（category-agnostic），参数取自 **Groma** Region Proposer，分类头改二分类，后处理 NMS。

---

## 6. 创新点

1. **MultiClip-Bench 数据集与自动标注引擎**：首个多镜头视频数据集，提出 keyframe description(带 person ID) + character information + dense caption 三要素标注，PySceneDetect+LLaVA-1.5+Gemini+GPT-4 全自动管线，专攻跨镜头人物一致性与短帧事件。
2. **Instance Prompt + IPFormer 连接器**：检测器+RoI pooling+跨帧余弦聚类显式抽取并聚合实例级特征，作为 anchor 注入 learnable query 引导 cross-attention，**显式保留地**把实例身份信息带过场景切换（论文措辞为 "reduces key feature loss"，非"无损"），解决"实例身份遗忘"与"短时人物被淹没"。
3. **高效注意力压缩**：以 anchor-guided Q-Former 取代 full projection，token 压到基线 <10%（每 slice 160），第一阶段训练时间减约 75%（整体约减半），性能反超 full-projection。

> 核对：各数值（X=5、V=80、M<10、阈值0.9、N=256+1、160 token/slice）取自论文行78-84；两阶段超参取自行399-401；组件取自行397-398。无源码，file:line 为 HTML 纯文本行号。
