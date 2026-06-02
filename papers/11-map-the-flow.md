# 11 · Map the Flow（VideoLLM 时序推理的信息流机制解剖）

> ICLR 2026 · [arXiv 2510.13251](https://arxiv.org/abs/2510.13251) · [OpenReview QCB0HN61TU](https://openreview.net/forum?id=QCB0HN61TU) · [项目页](https://map-the-flow.github.io/) · 官方代码：[github.com/byminji/map-the-flow](https://github.com/byminji/map-the-flow)

基于**官方源码逐行分析**写成（克隆成功，约 7M，仅代码；模型在 HF `byminji/map-the-flow`）。这是一篇机制可解释性论文：用 attention knockout / logit lens 定位 VideoLLM 时序推理的信息通路。

---

## 1. 源码可得性 / 结构

- `analysis/`：全部解剖工具——`causal_intervention_tools.py`（attention knockout / logit lens 底层）、`information_flow_analysis.py`（三阶段信息通路定位）、`effective_pathway_analysis.py`（有效通路剪枝实验）、`logit_lens_analysis.py`、`gen_prob_analysis.py`、`attention_visualization.py`。
- `scripts/analysis/*.sh`：各模型运行脚本。
- `models/`（internlm2 / internvl / phi3 / pllava 的魔改 modeling）、`tasks/eval/`（数据集加载、conv 模板）、`dataset/`、`videollama3/`。
- `docs/TRAIN.md`、`README.md`、`requirements.txt`。

---

## 2. 数据 / 输入格式

- **被探测模型**：LLaVA-NeXT-7B/13B-Video-FT（Vicuna LLM）、Mini-InternVL-4B-Video-FT、VideoLLaMA3-7B。前两者由作者在 `VideoChat2-IT` 上做 video 指令微调，用来对比"基座 ImageLLM vs. 微调后 VideoLLM"。
- **数据集**：主用 **TVBench**（时序推理多选题），另用 TOMATO、LongVideoBench、Video-MME、VCGBench（开放式）。配置见 `tasks/eval/config_dataset.py`、`README.md:98-102`。
- **输入组织**：视频抽帧为 PIL 列表，`pooling_shape=8-12-12`（T-H-W）→ 每条视频 `8×12×12 = 1152` 个 video token；conv 模板 `eval_mvbench` 拼 prompt。token 序列切成 `system / vision(<image>) / question(含选项) / last` 四段（`information_flow_analysis.py:236-303`）。
- **代表性例子**（注释内真实样例，`information_flow_analysis.py:245-255`）：
  ```
  USER: <image>
  USER: Question: What happened after the person took the food?
  Options: (A) Ate the medicine. (B) Tidied up the blanket.
           (C) Put down the cup/glass/bottle. (D) Took the box.
  Only give the best option. ASSISTANT:Best option:(
  ```
  含时序词 "after"，最后强制模型在 "(" 之后预测单 token 选项字母（A/B/C/D），便于读概率。

---

## 3. 完整方法流程

**核心干预原语 = Attention Knockout**（`causal_intervention_tools.py`）：
- `precompute_attention_masks()`（:6-34）的精确五步：① `torch.tril(ones(N,N))` 建因果下三角基底 mask（:13），允许位看到自身及之前；② 对每个待切断的 `(query_range, key_range)` 对，用高级索引 `attn_mask[q_idx[:,None], s_idx[None,:]] = 0`（:25-29）把这些"query→key" entry 从 1 翻成 0——**精准抠掉指定 token 对的注意力，而保留其余因果连接**；③ `attn_mask.repeat(1,num_heads,1,1)`（:31）把同一 mask 广播到全部注意力头（所有头同等切断）；④ `(1.0 - attn_mask) * torch.finfo(dtype).min`（:32）把"保留=1/切断=0"转成加性 mask：保留处加 0、切断处加 `dtype` 最小值（≈−∞），经 softmax 后该 entry 概率→0；⑤ `opposite=True` 时（:15,:29）整体取反，用于"只保留某通路、切掉其余"的有效通路实验。这种加性 mask 直接加到 attention logits 上，是一种**无需改权重的因果干预**。
- 自回归续写时，把预计算 mask 的**末行裁成 `(1, key_len)`** 复用到后续每个新 token（:111-120），保证生成阶段干预持续生效。
- `_set_block_attn_hooks()`（:103-147）用 `functools.wraps` 包裹每个 `layer.self_attn.forward`，仅对 `layerlist` 指定层注入 mask；自回归时把 mask 末行裁成 `(1, key_len)` 续用（:114-120）。
- `trace_with_attn_block()`（:54-68）跑一次 forward，返回被干预后 **answer / gt token 的 softmax 概率**。冲击量 = `(new-base)*100/base`（`information_flow_analysis.py:384-388`）。

**三阶段信息通路的定义与定位**（`information_flow_analysis.py`，`--target` 选择 + `--window`/`--sweep_range` 逐层滑窗）：

1. **早-中层 cross-frame 交互**（`--target cross-frame`，:306-313）：`find_inter_frame_block_ranges()`（`causal_intervention_tools.py:324-334`）把 1152 video token 按 8 帧分组，构造"帧 i 的 query → 前序所有帧的 key"的阻断对，即切断跨帧注意力。VideoLLM 在早-中层被切后概率大跌，基座 ImageLLM 几乎不变 → 该能力来自 video 指令微调。
2. **时序词对齐 / 视频→语言整合**（`--target vql-to-ql`，:315-321）：定义 `Video↛Question / Video↛Last / Question↛Last / Last↛Last` 四类阻断。`vq-to-true-opt`（:328-332）进一步细分。配合 **Logit Lens**（`logit_lens_analysis.py`）把各层 video token 残差投影到词表（`hs.matmul(E.T)`，`causal_intervention_tools.py:242-248`），统计落在 **temporal bag-of-words**（`eat/open/put/take/up...`，`logit_lens_analysis.py:33-38`）vs **spatial bag-of-words**（`bag/bed/box...`，`logit_lens_analysis.py:24-32`）的频次，证明中层 video token 与"时序概念词"对齐。
3. **答案生成（中-后层）**（`--target question-and-options-to-last`，:323-326 + `gen_prob_analysis.py`）：逐层追踪 last token 上正确/错误选项概率，显示整合完成后概率立即上升。

**保留有效 edges 的实验**（`effective_pathway_analysis.py`）：只**保留**三条通路、其余全切。7B 配置（:260-263）：cross-frame 留 L6-15、Video→Question 留 L6-20、Question→Last 留 L16-25；`last_to_last`/`vision_to_last` 全层切断（:275-288）。逐层组装后统计保留比例 `attn_count_new / attn_count_baseline`（:327-329, 367, 386）。结论：7B 可**压制约 58% 注意力 edges**（≈保留 42%）而保持 VideoQA 精度（`README.md:38`；`effective_pathway_analysis.py:33-35`）。

---

## 4. 一条真实数据的全过程

取上面 "What happened **after** the person took the food?"（TVBench，gt=B "Tidied up the blanket"）：

1. **构建输入**：`processor(text=prompt, images=8帧)` → input_ids；定位 `vision_range`（image_token 起 1152 个）、`question_range`（含 "after" 与四选项）、`last_token`（"(" 位置）（:263-302）。
2. **基线 forward**：`predict_from_input()` 取 last token logits→softmax，得 base 预测 "B"，记 `base_score`（answer prob）与 `base_score_gt`（:201-220）。
3. **逐层 attention 探测**：`window=9` 滑窗扫各层（:349-362），`--target cross-frame` 在 L6-15 切跨帧注意力 → `trace_with_attn_block` → "after/took" 的 video→question 对齐崩塌，正确选项概率相对下降最大。
4. **knockout 干预 → 概率变化**：每层每 block 记 `relative_diff=(new-base)*100/base` 与新预测 token（:373-395）。干预 `Question↛Last`（L16-25）时 last token 拿不到 "after→tidied" 信息，预测从 B 翻为他选；干预早层 cross-frame 则在更早处破坏。
5. **有效通路验证**：`effective-pathway-7b` 只留三段通路、压掉约 58% edges 后重跑，预测仍为 B、精度基本不降 → 证明这三条即"足够通路"。

---

## 5. 模型 / 组件

- **被解剖 VideoLLM**：LLaVA-NeXT-7B/13B-Video-FT、Mini-InternVL-4B-Video-FT、VideoLLaMA3-7B；对照基座 `llava-v1.6-vicuna-7b/13b-hf`、`Mini-InternVL-Chat-4B-V1-5`。
- **解剖工具**（全在 `analysis/`）：Attention Knockout（注意力 forward hook + 加性 mask）、Logit Lens（输入 embedding 矩阵 `E` 投影、`_logit_lens_set_proj_hooks`，`causal_intervention_tools.py:231-265`）、跨帧阻断范围生成器、Attention Map 可视化、生成概率追踪。
- 框架基于 **PLLaVA** 代码库，因果干预借鉴 Google `dissecting_factual_predictions` 与 `cross-modal-information-flow-in-MLLM`（`README.md:180-185`）。

---

## 6. 创新点

1. **首个面向 VideoLLM 时序推理的完整信息流"蓝图"**：把 VideoQA 分解为"早-中层跨帧交互 → 中层视频-语言在时序关键词上整合 → 中-后层答案生成"三阶段，用统一 attention knockout 因果定位每段所在层区间。
2. **揭示跨帧时序能力源于 video 指令微调**：同架构下 VideoLLM 切早-中层跨帧注意力会崩、基座 ImageLLM 不受影响；并用 Logit Lens 证明 video token 中层与"时序概念词表"对齐、空间概念在更早层先于时序概念显现。
3. **有效通路充分性 / 注意力可大幅剪枝**：仅保留三条关键通路、压制约 58% 注意力 edges（7B），VideoQA 性能基本不降，为下游剪枝/泛化提供可解释性指导（`effective_pathway_analysis.py:259-288`）。

> 说明：仓库不含预生成 result JSON，单条样例的概率翻转为依据 pipeline 的合理推演（复现实跑数值需下载 HF 权重+数据集）；58%、各层区间等数值来自代码常量与论文/项目页。
