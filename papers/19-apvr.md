# 19 · APVR（Adaptive Pivot Visual-token Retrieval，免训练小时级长视频理解）

> AAAI 2026 · [arXiv 2506.04953](https://arxiv.org/abs/2506.04953)（v3, 2025-11-15 已被 AAAI 2026 接收）· 东南大学 + 中兴 ZTE，Hong Gao、Yiming Bao 等

> **无公开源码**：论文全文及附录无 GitHub/项目主页链接，GitHub API 检索官方实现 `total_count=0`。本文**全部基于论文正文与附录（Alg.1、公式 1–22、Prompt 模板）**，引用以论文文本行号给出，**无 file:line 代码引用**。

---

## 1. 源码可得性

无 repo。论文纯文本与 HTML 已抓取。`/home/user/lvp-src/apvr` 未创建。

---

## 2. 数据来源与真实格式

**评测基准**（三个 hour-level，均**不用字幕**）：
- **LongVideoBench (LVB)**：17 类 referred-reasoning 多选题；
- **VideoMME**（w/o sub.）：按时长分 Long(30-60m)/Medium/Short；
- **MLVU**：9 类任务的大规模长视频基准。

**输入格式**：多选题 QA。MLLM 接收 prompt = 问题 + 检索到的 pivot 帧 + 多选项，生成回答。评测框架 **LMMs-Eval**。

**真实样本**（附录 Fig.5）：
> Question: "When does the person in red clothes appear with the dog?"
> Key Objects: person, dog, red clothes; Cue Objects: grassy area, leash, fence
> Rel: (person; attribute; red clothes), (person; spatial; dog)

---

## 3. 完整方法（training-free 两级检索）

输入：视频 ℱ={f_t}, fps=2 抽帧，query Q，迭代数 P，初始步长 ∇。

### 3.1 Pivot Frame Retrieval (PFR) — 帧级（Alg.1）

**(a) 语义扩展**：LLM 把 Q 展开为四类——Objects（关键/线索物体，供 Grounding-DINO 检测）、Descriptions（实体/上位词）、Relations（三元组 spatial/time/attribute/causal）、Semantics。

**(b) 迭代自适应重采样**：步长逐轮收缩 `∇_p=max(1, ∇/p)`；每轮 `AdaptiveResample` 从未访问帧采样。

**(c) 时空置信打分**（每帧，公式1）：
- CLIP 语义分 `s^CLIP_t = softmax(τ·v_t·t_agg)`，τ=100，t_agg 聚合 {Q,Des,Sem}（公式17-19）；
- Grounding-DINO 空间分 `s^GD_t = s^o_t + Σ_r(ω_r·s_r)`，s^o 来自检测 logits softmax(max)，再按满足的关系三元组加权（公式20-22）；
- 融合 `S_t=(1−λ)·s^CLIP_t + λ·s^GD_t`，**λ=0.5**。

**(d) 时间扩散**：分数向邻帧传播 `S_i=max(S_i, S_t/(1+|i−t|)), i∈[t−w,t+w]`（公式2）。

**(e) 候选集** = 高置信集 `H_s=TopK(S⊙U, N/(2∇_p))`（公式3）∪ 高熵不确定集 `H_e={f_t | E_γ(s_t)>μ+0.5σ}`（Shannon 熵，公式4-5）；multinomial 重要性采样 + α 比例随机采样（公式6-8）。

**(f) 输出**：P 轮后 `F_pivot=TopK(S⊙ℱ, K)`，**K=1024**。最优超参：**P=3, ∇=4**。

### 3.2 Pivot Token Retrieval (PTR) — token 级

pivot 帧经 Vision Encoder G(·)+Projector P(·) 生成 `T_vis=P(G(F_pivot))`。

**(a) Query-aware 多层注意力打分**（公式9-10）：`A^l_cross=softmax(q^l_text·(k^l_vis)^T/√d)`，q_text 来自 {Q,Des,Sem}，按 query 维聚合 `a^l=Σ A^l_cross`。

**(b) 自适应 token 选择**（在 KV cache 上，按 token 维分 W 个 chunk）：
- 基础比 `η_w=Σ_w a/max{Σ_i a}`（公式11）；
- 动态比 `ρ_w=min(1.0, √(|{j: a_j>0.01·max(a)}|/L_w))`，阈值 **0.01·max(a)**（公式12）；
- 最终比 `γ_w=ρ_w·η_w`（公式13）；
- **Head-wise soft voting**：`Z_w=TopK((Σ_{j=1}^h softmax(a_{w,j}))⊙T_w, γ_w·L_w)`（公式15），每 head softmax 归一后求和投票；
- `T_pivot=Concat({Z_w})` 更新各层 KV cache（公式16）。最后 T_pivot 喂 MLLM 生成答案。

---

## 4. 一条真实数据全过程（附录样本）

以 "When does the person in red clothes appear with the dog?" 为例：
1. **抽帧**：hour-level 视频 fps=2 密集抽 N 帧。
2. **语义扩展**：LLM 输出 Key Objects=person/dog/red clothes，Cue=grassy area/leash/fence，Rel=(person;attribute;red clothes)、(person;spatial;dog)，Sem="leash often appears with dog"。
3. **PFR 迭代打分（P=3, ∇=4）**：每轮自适应采样帧 → CLIP 算 {Q,Des,Sem} 相似度 s^CLIP，Grounding-DINO 检测 person/dog/red clothes/leash 算 s^GD（含关系三元组加权）→ 融合 `S_t=0.5·s^CLIP+0.5·s^GD` → 时间扩散到邻帧 → 高置信+高熵集重采样下一轮。
4. **选 pivot 帧**：`F_pivot=TopK(S⊙ℱ, 1024)` 取出"红衣人与狗同框"的高分帧。
5. **PTR 选 pivot token**：1024 帧过 Vision Encoder+Projector 得 T_vis；用 {Q,Des,Sem} 作 query 算跨模态注意力 a；KV cache 分 W chunk，按 η_w/ρ_w/γ_w 与 head-wise soft voting，TopK 保留关注"红衣/狗"区域的 token，压缩无关 token，更新 KV cache。
6. **注入 LLM 出答案**：T_pivot + 问题 + 选项 → MLLM 生成正确时间点答案。hour-level 视频 ≤2 分钟完成。

---

## 5. 模型 / 组件

- **Base VideoLLM（三个 baseline）**：Qwen2-VL-7B、Qwen2.5-VL-7B、VideoLLaMA3-7B。
- **检索辅助视觉模型**：CLIP (ViT-B/16) 做语义相似；Grounding-DINO (Swin-T) 做检测+空间推理。
- **LLM query 扩展器**：语义信息展开（Prompt 模板见 Fig.5）。
- **检索机制**：PFR（时空置信打分 + 时间扩散 + 自适应重采样）+ PTR（多层注意力 + 动态分块 + head-wise soft voting 的 KV-cache token 选择）。
- **硬件/框架**：8×A800 80GB，LMMs-Eval。

---

## 6. 创新点

1. **Hour-level 突破 memory wall**：双粒度检索把可处理帧数从 7B 模型基线的 256 帧提升到 **K=1024 帧**，hour-level 视频 ≤2 分钟出答案。
2. **Training-free 即插即用双级检索**：首个同时在帧级 (PFR) 与 token 级 (PTR) 做 query 引导检索的训练无关框架，无需重训即可嵌入现有 MLLM，保留原模型能力。
3. **SOTA（训练无关 + 训练型均超越）**：相对 baseline 提升 LVB +9.5% / VideoMME +4.6% / MLVU +9.7%；Qwen2.5-VL+APVR 在 LVB 达 **64.9%**（超 GPT-4V、Gemini-1.5-Pro）、VideoMME **68.4%**、MLVU **76.1%**。

> 注：APVR 无官方开源代码，以上引用均指论文文本/公式编号，非源码 file:line。
