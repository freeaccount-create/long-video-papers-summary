# 06 · Improve Temporal Reasoning via Video Contrastive Decoding

> NeurIPS 2025 · [OpenReview id=2nIAtsUC27](https://openreview.net/forum?id=2nIAtsUC27) · 弗吉尼亚大学 / Adobe Research
> 全称：*Improve Temporal Reasoning in Multimodal Large Language Models via Video Contrastive Decoding*

> **无公开源码**：NeurIPS checklist 第 5 项作者答 `[NA]，"publish when accepted"`；GitHub 搜索作者账号（`daiqingqi`/`donglgcn`）与关键词均无对应仓库；补充材料需登录、仅为 PDF 附录非代码。本文**全部基于论文正文（§3.1-3.2、Eq.2-6）与附录 A/B/C/E**，无 file:line 可引。

---

## 1. 源码可得性

无 repo。已获取 OpenReview 元数据与 PDF（`paper.pdf` / `paper.txt`）。`/home/user/lvp-src/video-cd` 未建立。

---

## 2. 数据 / 输入格式

**评测 benchmark**（§4）：时间专项 **TempCompass**（事件排序、属性变化、方向、速度、动作）、**EventHallusion**；通用 **Video-MME**（Temporal Perception/Reasoning 子项）、**MLVU**（Action Count/Order 等带 * 的时间维度）。

输入对 = (文本 query `x`, 视频 `V`)；视频采样为帧序列（Video-LLaVA 8 帧，LLaVA-Video-7B-Qwen2 32 帧）。任务多为多选题。

**代表性例子**：
- **image-prior 误导例（Fig.1(2)(4)，太阳题）**："What direction is the sun moving in? A. Falling B. Rising C. Staying"。这里盲测（去视频）LLM **答对 B.Rising**（论文称之为 language prior 的 *positive influence*），而 **VideoLLM 反被某中间帧"像日落"的 image prior 误导、错答 A.Falling** —— 典型时间推理失败。
- **流程示意例（Fig.3，小狮子题）**："In what direction is the little lion going? A. Falling down(GT) B. Climbing up ..."。此例仅作 Fig.3 对比解码流程的示意图，论文正文未对它做盲测/image-prior 分析。

---

## 3. 完整方法流程

**核心两分支对比解码公式**（Eq.3 / Eq.6）：
```
p_vtd(y | V, V', x) = softmax[ (1+α)·logit_θ(y | V, x) − α·logit_θ(y | V', x) ]
```
注意是 logit 空间的 `(1+α)·p_orig − α·p_distort`（等价"原分支 + α·(原 − 扭曲)"，放大原与扭曲的差），而非朴素 `p_orig − α·p_distort`；α=0 退化为普通解码。`V'` = 时间扭曲后的视频。

- **α**：固定超参，所有实验 **α=1**（附录 C）。
- **是否每步都做**：是。Eq.6 带 `y_<t`，每个自回归 token 步跑两次前向（原+扭曲）相减。
- **自适应可信约束**（Eq.4-5）：只对原分支高概率 token 施加 CD——仅保留 `p_θ(y_t|V,x) ≥ β·max_w p_θ(w|V,x)` 的集合，集合外置 0，避免惩罚到扭曲分支偶然答对的正确 token。**β=0.2**。

**"扭曲时间一致性"的负分支构造**（§3.2，不是简单打乱）。作者先验证朴素方案（加噪/随机打乱帧序/随机丢帧）对采样位置敏感、不稳定，遂提出**注意力引导的自适应扭曲**四步：
1. **注意力引导的帧/token 重要性**：取 LLM 各中间层 attention，对最后 query 位置在各 head 求平均得 token 重要性 `S_l`，动量跨层累积 `S̃_l = w_m·S̃_{l-1}+(1−w_m)·S_l`（`w_m=0.8`），帧重要性 = 帧内 image token 重要性之和。
2. **Key Frame Fusion**：选 top-`w_fdr` 最重要帧，把它们各自**替换为这些选中帧的均值池化**（抹掉时间线索、保留粗粒度图像上下文），再加 `w_fpr` 权重的高斯噪声。
3. **剩余帧 token 级扭曲**：对其余帧 mask 掉 top-`w_tdr` 最重要的 image token。
4. **运动内容扰乱**：非重叠滑窗（`w_ws=8`）分块下采样（块内取均值），用窗口内同位置块的余弦相似度判定"动态块"（相似度低=运动大），选 top-`w_cfr` 最不相似块做均值池化覆写（仅当该位置在窗口其它帧也为动态块时才跨帧均值覆写，否则保留原值）——抹糊运动、保留静态背景。

四步输出拼成 `V'`。设计哲学：扭曲要**适中**（distortion ratio ≳0.6 会随机化输出、失去引导作用）。TempCompass 超参例：α=1, β=0.2, w_fdr=0.2, w_tdr=0.4, w_ws=8, w_cfr=0.3, w_fpr=0.5, w_m=0.8。

---

## 4. 一条真实数据的全过程（基于 Fig.3 流程示意复述）

取 Fig.3 的"小狮子朝哪移动"示意例（注：这是论文流程图的演示样例，非论文报告的某条实测改正样本；论文正文明确给出的"+Ours 由错改对"文字样例是 Fig.1(2) 的吐司题，LLaVA-Video-7B-Qwen2 由 C 改为 A）：

1. **原视频 → 正分支 logits**：`V`+`x` 送 LLM 得 `logit_θ(y|V,x)`；中间层 attention 喂给扭曲单元定位关键帧。此时因 language/image prior，错误选项 B 已被分配不低分数。
2. **扭曲帧 → 负分支 logits**：经四步得 `V'`（关键帧均值池化+噪声、剩余帧 mask 重要 token、运动块糊化），送同一 LLM 得 `logit_θ(y|V',x)`。时间信息被移除、误导性静态先验被放大 → 错误选项 B 在负分支拿到**显著更高**分数。
3. **token 级相减 → 输出**：每步按 Eq.6 算 `(1+α)·logit(y|V,x) − α·logit(y|V',x)`（α=1），β=0.2 截断到原分支高概率集合。B 因负分支得分高被大幅压低，正确选项 A 相对突出 → 输出正确答案（此为 Fig.3 示意流程；论文实测改正样例见吐司题）。

---

## 5. 模型 / 组件

- **Base VideoLLM**：Video-LLaVA（8 帧）、LLaVA-Video-7B-Qwen2（32 帧）。
- **盲测 LLM**（诊断 language prior）：Qwen2-7B-Instruct、Vicuna-13B-v1.5。
- **视觉编码器**：CLIP。
- **对比基线 CD**：VCD（图像加噪）、SID（注意力剪 token）、TCD（随机丢帧）——本文在时间子项上均显著优于三者。
- **效率**：`torch.cuda.Stream()` 并行两次前向，速度接近原始 VideoLLM。硬件 A6000/A100。

---

## 6. 创新点

1. **新失败诊断视角 —— language prior + "image" prior**：不从架构入手，而从响应行为发现 VideoLLM 时间推理失败既受文本先验（在答错样本中，46.7%(LLaVA-Video-7B-Qwen2)/38.9%(Video-LLaVA) 与盲测 LLM 答案一致）又受单帧静态"image prior"误导。
2. **注意力引导的自适应时间扭曲**：不同于随机加噪/丢帧/打乱，四步流水线在**移除时间线索的同时保留误导性静态先验**，使负分支稳定产出"时间不敏感的错误答案"。
3. **训练无关、模型无关、即插即用的解码侧方法**：无需训练/数据，仅 decoding 阶段用 Eq.6 + 自适应可信约束，即可在时间专项与通用视频两类 benchmark 上一致提升（如 Video-MME Temporal Perception 84.1 vs 61.1）。
