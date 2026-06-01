# 05 · When Thinking Drifts（Video-VER）

> NeurIPS 2025 · [arXiv 2510.06077](https://arxiv.org/abs/2510.06077) · [项目页](https://vision.cs.utexas.edu/projects/video-ver/)

> **无公开源码**：项目页 "Code" 按钮指向占位链接 `https://github.com/`，未指向真实 repo；GitHub 搜索无对应仓库。本文**全部基于 arXiv 论文 2510.06077**（已抓取全文），无 file:line 可引，统一以「paper.txt:N」与公式作引。

---

## 1. 源码可得性

- 无公开仓库（主页 Code 按钮为占位符）。HuggingFace 无对应模型/数据卡。
- 论文未随附代码或附录脚本。下文基于论文全文还原。

---

## 2. 数据 / 输入格式

**诊断用 MVBench 子任务**（paper.txt:111）：利用 MVBench 结构化分类分析 **20 个子任务**。结论——CoT 在"快速视觉是/否判断或单标签判断"任务上掉点（典型 scene transition detection），在"多跳/计数"任务上才有益（object/action counting）。整体诊断覆盖 10 个 benchmark、3 个开源 MLLM（Qwen2.5-VL-3B/7B、Video-R1-7B）。

**训练数据**（两阶段，paper.txt:190-191）：SFT 冷启动用 `Video-R1-COT-165k`；RL（GRPO+VER）用 `Reversed-in-Time (RTime)` + `Video-R1-260k` 混合。VER 所需的关键字段是**问题相关视觉证据（QD-VE）**，由外部 `Qwen2.5-VL-72B` 离线、用**倒置提示（inverted prompting）**生成：喂入 (question, ground-truth answer)，让 72B 产出"支持该答案的视觉观察列表"（paper.txt:181, 1307）。

**一条代表性样例**（依论文构造）：
```
question: "What will the person most likely do next?"  (next-action 预测，MCQ)
answer:   (ground-truth 选项)
visual_evidence (QD-VE, 由 Qwen2.5-VL-72B 倒置提示生成):
   - "a red ball enters the basket"   # 论文反复用的示例视觉事实
   - <其它最小、可验证、与问题相关的视觉观察>
```
证据被约束为"最小、可验证、问题相关"的事实片段（paper.txt:1305），而非通用 caption。

---

## 3. 完整方法流程

### 诊断阶段（暴露问题）
1. **CoT 降准确率的实证**（paper.txt:101-115）：对比 Direct Answer (DA) 与 CoT。开源模型多数 benchmark 上 CoT < DA（如 Qwen2.5-VL-7B：MVBench DA 63.6 → CoT 59.8）。
2. **Visual Thinking Drift（视觉思维漂移）**（paper.txt:117-154）：许多错误 CoT"逻辑自洽却脱离视频内容"——基于幻觉视觉细节或孤立时间片段推理。next-action 是典型：抓早期事件、忽略更近线索。
3. **贝叶斯解释**（paper.txt:123-152）：生成过程 `p(c_{1:T}, a | q, v) = p(a|c_{1:T},q,v)·∏_t p(c_t|c_{<t},q,v)`。单步 softmax 含"语言先验项 h^T·W_lang"与"视觉似然项 h_v^T·W_vis"，实际 ‖W_lang‖ ≫ ‖W_vis‖；随链长 t 增大，视觉似然被稀释。单步正确率 1−ε 则长度 T 链全对 ≈ 1−Tε：**失败率随链长线性增长**；早期 token 一旦锁定不存在的视觉事实，自回归无回溯，后验塌缩到幻觉。

### 药方阶段（VER + GRPO）
对每个问题 q，策略 π_θ 采样 G=8 个回答 {o_i}。VER 奖励（paper.txt:162-164）：
1. **LLM judge 给二元证据分** e_i∈{0,1}：用 `Llama-3.1-70B-Instruct`（temp 0）判断 o_i 的 CoT 是否"恰当引用了视觉证据 v"。
2. **证据奖励系数** r_e = α（取 **0.3**）若 e_i=1，否则 0。
3. **双门控证据增广奖励**：
```
r_i^evid = r_i + r_e   仅当 (答案正确) 且 (e_i = 1)
r_i^evid = r_i         其它情况
```
即**只有答案对、且推理确实落地到视觉证据时才发放额外奖励**。
4. **组内归一化优势** A_i = (r_i^evid − mean) / std。
5. **clipped GRPO 目标**（paper.txt:168-173）：标准 GRPO，含重要性采样比、ε 截断、β·KL(π_θ‖π_ref)。

实际共 **4 个奖励**（paper.txt:504）：accuracy、visual evidence(α=0.3)、format、length（鼓励 320–512 token）。训练 16 帧、推理 32 帧、8×H200、2000 RL 步。

---

## 4. 一条真实数据的全过程（next-action 预测）

**输入**：视频 v + q="What will the person most likely do next?"。训练数据已附 QD-VE，如 `"a red ball enters the basket"`。

- **Step 1 — 采样 G=8 个带视觉证据的 CoT**：
  - o_3：CoT 明确写"球已接近篮筐、上一帧人正抬手投球" → 选正确选项。
  - o_5：CoT 只复述早期场景、凭语言先验编故事 → 答案错（典型 drift）。
  - o_7：答案碰巧对，但 CoT 泛泛叙述、未引用任何具体视觉事实。
- **Step 2 — VER 判定**（Llama-3.1-70B judge，二元）：o_3 e=1；o_5 e=0；o_7 e=0。
- **Step 3 — GRPO 奖励**（α=0.3，accuracy 设 1/0 示意）：
  - o_3：对 ∧ e=1 → r^evid = 1+0.3 = **1.3**。
  - o_5：错 → **0**。
  - o_7：对但 e=0 → 1+0 = **1.0**（会"说对"但不"看着说"，拿不到证据加成）。
  - 组内归一化：o_3 优势最高，o_7 次之，o_5 为负。
- **Step 4 — 更新**：clipped GRPO 按 A_i 提升 o_3 这类"看着想"的轨迹、压低 o_5，并相对压低 o_7，β·KL 约束不偏离初始模型。

> 设计要点：奖励**有意做成二元**——证据缺失只是"零奖励"而非负惩罚，避免被 teacher 偶发幻觉带偏；推理时 Video-VER **完全独立**，不依赖外部 72B/judge。

---

## 5. 模型 / 组件

- **Base / policy**：`Qwen2.5-VL-7B`，两阶段后训练（SFT→RL）得 **Video-VER**。
- **RL 算法**：GRPO，扩展为 evid-GRPO。
- **视觉证据生成器（teacher，仅离线训练用）**：`Qwen2.5-VL-72B`，倒置提示。
- **LLM judge（奖励信号）**：`Llama-3.1-70B-Instruct`，temp 0，输出 0/1。
- **算力 / 超参**：8×H200；G=8、α=0.3、训练 16 帧 / 推理 32 帧、2000 RL 步。

---

## 6. 创新点

1. **现象与机理**：首次系统提出并命名 **"Visual Thinking Drift"**，用**贝叶斯视角**解释——语言先验权重远大于视觉似然（‖W_lang‖≫‖W_vis‖）、视觉似然随链长被稀释、失败率随链长线性增长。
2. **Visual Evidence Reward + evid-GRPO**：把"推理是否落地到视觉证据"显式做成 RL 奖励，采用 **answer-correct ∧ evidence-grounded 双门控二元加成**，让模型从"think before answering"转向"see while thinking"。
3. **倒置提示生成 QD-VE**：用 (question, GT-answer) 反向让 teacher 产出"支撑已知答案的最小可验证视觉事实"，比通用 caption 更强制视觉 grounding；消融显示 10 个 benchmark 中 9 个优于通用 caption。

> 结果：Video-VER（7B, CoT）在 10 个 benchmark 中 9 个第一，相对 base 平均 +4.0%，最高 +9.0%（VideoHallucer 44.1→53.1）。
