# 12 · Let's Split Up（零样本分类器编辑做细粒度视频理解）

> ICLR 2026 · [arXiv 2602.16545](https://arxiv.org/abs/2602.16545) · [项目页](https://kaitingliu.github.io/Category-Splitting/) · 官方代码：[github.com/KaitingLiu/Category-Splitting](https://github.com/KaitingLiu/Category-Splitting)

基于**官方源码逐行分析**写成（克隆成功，约 51M，源码 + benchmark 标注齐全；视频与 base ckpt 需另下）。作者 Kaiting Liu, Hazel Doughty（Leiden University）。提出 **category splitting（类别拆分）** 新任务：把已训练分类器的一个粗类**零样本**编辑成多个细粒度子类，同时不损害其余类别。

---

## 1. 源码可得性 / 结构

- `main.py`：入口/编排。
- `algs/editor.py`：基类 `Editor` + 新增 head `NewHead`。
- `algs/ma.py`：**Modifier Alignment（主方法）**；`algs/mr.py`：Modifier Retrieval；`algs/vlm.py`：CLIP baseline；`algs/ft.py`：low-shot 微调。
- `models/mvd.py`：base 视频分类器（MVD/ViT）。
- `datasets/`（`ssv2.py`、`finegym.py` 等）。
- `benchmark/{SSv2-Split,FineGym-Split}/{A,B}/`：`labels.json`、`train/val.csv`、每个粗类下 `equivalent_set/ft_set/unrelated_set.csv`。
- `modifiers/modifiers_in_base_model/*.json`、`modifiers/modifiers_for_new_classes/*.json`：modifier 字典。
- `configs/`、`scripts/`（Table2~4 复现脚本）。

---

## 2. 数据 / 输入格式

- **数据集**：基于 **Something-Something V2 (SSv2)** 与 **FineGym288** 构造 `SSv2-Split`、`FineGym-Split`，各有 A/B 两种 mixed-granularity 划分（部分类别给粗标签、部分保留细标签，模拟真实标注情形）。
- **标签组织**：`benchmark/SSv2-Split/A/labels.json` 共 **119 个 base 类**（`text_label → index`，如 `"Dropping something with spatial relation": 6`）。拆分出的新细类索引从 119 往后排（120、121…），见 CSV：`.../Dropping something with spatial relation/equivalent_set.csv:1` → `107014.mp4,120`（格式 `video_file,label_index`）。
- **每个粗类三个子集**（`main.py:162,180,182`）：`ft_set.csv`（low-shot 微调）、`equivalent_set.csv`（衡量 **generality**，新子类未见样本）、`unrelated_set.csv`（衡量 **locality**，其余类别不受影响）。
- **代表性例子**（`modifiers/modifiers_for_new_classes/SSv2-Split.json`）：粗类 `"Dropping something with spatial relation"` → 5 个细子类 `Dropping something behind/in front of/into/next to/onto something`，modifier 即 `"something behind/in front of/..."`。

---

## 3. 完整方法流程（零样本权重编辑）

**核心思想**：动作 = 对象 × 方式 × 空间关系 × 结果，其组合结构在分类器 head 权重空间近似线性可分。细类权重 ≈ 粗类（原型）权重 + 一个 **modifier 向量**。

**(a) 加新 head**（`editor.py:11-48`, `NewHead`）：原 head 旁并接 `fine_grained_head=nn.Linear(D, num_fine)`，前向拼接 logits `output=cat([head(x), fine_grained_head(x)])`（:43-48）。新权重默认用粗类权重初始化（:30-35）：
```python
w = weight_matrix[coarse_idx : coarse_idx+1].repeat(num_fine, 1)   # W_fine = W_coarse 复制
```

**(b) 从 base model 提取 modifier 向量字典**（`ma.py:48-70` `get_modifier_dict`, `mr.py:32-49`）：对 base model 中本就细标的粗类，求细类权重均值作"粗类原型"，再用每个细类权重减原型得 modifier：
```python
coarse_prototype = mean( W_head[index_list] )          # ma.py:57
modifier_vector  = W_head[fine_idx] - coarse_prototype # ma.py:65 / mr.py:42
```
即 **v_modifier = w_fine − (1/K)Σ w_fine_k**，以 modifier 文本为 key、向量为 value 建字典。

**(c) 编辑出子 head** — 三种零样本策略：
- **MA / Modifier Alignment（主方法）**：训轻量 MLP（`text_embed_dim → 384 → D`，`ma.py:38-46`），用 CLIP 文本嵌入回归 base model 的 modifier 向量（MSE，`ma.py:97-98`），从而**泛化到未见 modifier**。编辑时把目标子类 modifier 文本经 CLIP 编码 → MLP 映射成向量，**加到**对应子 head 权重（`ma.py:132-137`）：
  ```python
  modifier_text_embeds = CLIP.encode_text(modifier_texts)        # ma.py:134
  modifier_vectors     = mapping_function(modifier_text_embeds)  # ma.py:135
  fine_grained_head.weight.data[i] += modifier_vectors[i]        # ma.py:137  → W_sub = W_coarse + v_modifier
  ```
- **MR / Modifier Retrieval（贪心一对一二分匹配，`mr.py:51-71`）**：不是各子类独立取 argmax，而是**全局贪心二分匹配**，保证字典里每个已知 modifier 至多被用一次：
  1. 算双通道相似度矩阵 `sim_modifier = target_modifier_embeds @ modifier_embeds.T`、`sim_fine = target_fine_embeds @ fine_embeds.T`，加权 `sim_total = α·sim_modifier + β·sim_fine`（`α=β=0.5`，:59-61）——既比 modifier 文本（"into"）也比完整细类名（"Dropping … into …"）。
  2. 循环 `num_fine_grained_class` 次（:63-71）：每轮取整个矩阵的**全局最大值** `sim_total.max()`，定位其 `(目标子类 i, 字典条目 j)`（:64-65），把字典第 j 个 modifier 向量加到第 i 个子 head（`fine_grained_head.weight.data[i] += modifier_vectors[j]`，:68）。
  3. 随即把**第 i 行与第 j 列全部置 `-inf`**（:70-71）形成互斥，使该子类与该字典条目都不再参与后续轮——即标准贪心最大权二分匹配，避免多个子类抢同一个 modifier。
- **VLM baseline**：不编辑权重，先 base model 判粗类，再用 CLIP image-text 相似度选细类（`vlm.py:23-26,61-71`）。

**(d) 评测/打分**（`editor.py:121-159`）：视频特征过 `forward_features`（mean-pool 得 `x∈R^D`，`mvd.py:382-407`）→ `NewHead` 输出全部 logits → softmax → **把原粗类索引置 0 并重归一化**（:137-138），多 crop 聚合（`aggregate_groups`，:86-118）取 argmax。Locality = 编辑后/前在 unrelated_set 上准确率之比（:187）。

---

## 4. 一条真实数据的全过程（"Dropping … spatial relation" 拆 5 子类）

以真实粗类 `"Dropping something with spatial relation"`（index=6）为例：

1. **加 head**：`num_fine=5`（`main.py:138`），`fine_grained_head=Linear(384,5)`，5 行全部初始化为粗类 #6 权重 `W_6`（`editor.py:30`）。
2. **取 modifier 文本**：从 `modifiers_for_new_classes/SSv2-Split.json` 读出 5 个 modifier（`main.py:140`, `ma.py:32`）。
3. **建字典 + 训练对齐 MLP**：从 base model 已细标粗类抽 (modifier文本嵌入, modifier向量) 对，训 MLP 拟合（`ma.py:72-130`，CLIP ViT-L/14 文本编码 + MSE + 余弦相似度早停）。
4. **生成 5 个 modifier 向量**：`v_i = MLP(CLIP("something into something"))` 等（`ma.py:134-135`）。
5. **编辑子 head**：`W_sub_i = W_6 + v_i`，i=0..4（`ma.py:137`）。"Dropping something into something" 子 head = 粗类"Dropping"权重 + "into"方向向量。
6. **对一条视频打分**：视频 → `forward_features` → `x` → `NewHead` 输出 119+5 logit → softmax → 粗类 #6 置 0 重归一化（`editor.py:137`）→ 在 5 子类取最大 → 与 GT（如 `120`）比对统计 generality；同时在 unrelated_set 验证其余 118 类预测不变（locality）。

---

## 5. 模型 / 组件

- **Base 视频分类器**：`models/mvd.py` 的 ViT（`vit_small_patch16_224`，embed_dim=384，16 帧、tubelet=2、mean-pooling、无 cls token），即 **MVD (Masked Video Distillation)** 风格 backbone，head 为单层 `nn.Linear(384, num_classes)`（`mvd.py:346`）。base 为 mixed-granularity 训练的 checkpoint。
- **文本/视觉编码器**：**CLIP ViT-L/14**（MA 文本嵌入、MR 检索相似度、VLM baseline 匹配）。
- **对齐模块**：自建轻量 MLP（`text_embed → 384 → D` + GELU，`ma.py:38-46`）。
- **优化**：AdamW + CosineAnnealingLR + EMA 早停（`ma.py:79-130`、`ft.py`）。

---

## 6. 创新点

1. **新任务 category splitting**：把"已训练分类器中某粗类细分为多个子类、同时保持其余类别预测不变"形式化为编辑问题，给出 generality / locality 双指标与 SSv2-Split、FineGym-Split 两个基准（`editor.py:181-188`）。
2. **基于潜空间组合结构的零样本权重编辑**：发现分类 head 权重满足"细类 = 粗类原型 + modifier 向量"近似线性组合（`ma.py:57,65`），无需任何细粒度视频数据即可直接编辑出子 head（`ma.py:137`），完全免重训练。
3. **Modifier Alignment 对齐模块**：CLIP 文本嵌入 → modifier 向量的轻量映射网络（`ma.py:38-46,132-137`），可**泛化到字典中从未出现的 modifier**，显著优于 CLIP-VLM baseline，并可作为 low-shot 微调的优良初始化（`main.py:165-173`）。

> 说明：源码公开完整，file:line 均可核对。粗类示例如 "Dropping something with spatial relation"、"Closing/Opening something" 等，方法管线一致。
