# 13 · WorldMM（多模态记忆 agent 处理超长视频）

> CVPR 2026 Highlight · [arXiv 2512.02425](https://arxiv.org/abs/2512.02425) · [项目页](https://worldmm.github.io/) · 官方代码：[github.com/wgcyeo/WorldMM](https://github.com/wgcyeo/WorldMM)

基于**官方源码逐行分析**写成（克隆成功，约 81M，含完整 Python 源码，未下权重/视频）。作者 KAIST / NTU / DeepAuto.ai。把 hour/week-long 视频当作外部记忆，三类记忆 + 自适应检索 agent。

---

## 1. 源码可得性 / 结构

- `src/worldmm/memory/`：`memory.py`（统一编排 `WorldMemory`）、`episodic/`、`semantic/`、`visual/` 三类记忆。
- `src/worldmm/llm/`：LLM 封装（`openai_gpt.py`/`qwen3vl.py`）+ `templates/`（reasoning、qa、multiscale_filter、各类抽取提示）。
- `src/worldmm/embedding/`：`qwen3_embedding.py`（文本）+ `vlm2vecv2.py`/`VLM2Vec/`（视觉，TIGER-Lab 子模块）。
- `preprocess/`：三类记忆离线构建脚本；`eval/eval.py`(Video-MME)、`eval/eval_egolife.py`(EgoLifeQA)。
- `src/HippoRAG/`：episodic 检索后端（OSU-NLP HippoRAG 内嵌）。
- `script/`、`script/videomme/`：1_setup→2_preprocess→3_build_memory→4_eval 流水线。

---

## 2. 数据 / 输入格式

- **评测 benchmark**：论文称在 **五个 long-video QA benchmark** 上评测、平均较 SOTA 提升 **8.4%**。代码落地两条完整管线：**EgoLifeQA**（week-long 第一人称，`data/EgoLife/`，6 人各多天，每人 30sec 字幕约 6223 条）为主战场；**Video-MME**（`eval/eval.py`）。
- **输入格式**：episodic 基础输入是 30 秒粒度 caption JSON 列表，真实样例（A1_JAKE）：
  ```json
  {"start_time":"11094300","end_time":"11095800","text":"I hold my phone ... \"Okay, then we need a stopwatch.\" ...","date":"DAY1","video_path":"data/EgoLife/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4"}
  ```
  时间编码为整数 `day+HHMMSSFF.zfill(8)`（`episodic/memory.py:30-35` `timestamp_int`）。QA 行含 `question/choice_a..d/answer`，按 video 分组逐题作答（`eval.py:118-129`）。
- **三类记忆表示**：
  - **Episodic**：多时间尺度文本 caption 条目 `CaptionEntry`（`episodic/memory.py:18-41`），粒度 `10sec/30sec/3min/10min/1h`（`memory.py:81`），每粒度一个 HippoRAG 事件图索引。
  - **Semantic**：知识图谱三元组 `SemanticTripleEntry(subject,predicate,object,timestamp)`（`semantic/memory.py:18-39`），按时间戳累积更新，存为 `{timestamp:{consolidated_semantic_triples:[[s,p,o],...]}}`。
  - **Visual**：`VideoClipEntry`（`visual/memory.py:20-43`）= VLM2Vec 视觉 embedding（`.pkl`，关键词语义检索）+ 30s clip 元数据/时间戳（精确定位后按 1fps 抽帧）。

---

## 3. 完整方法流程（三阶段）

**Stage 1 — 多模态记忆构建（离线，`preprocess/build_memory.py`，`script/3_build_memory.sh`）**
- Episodic：`generate_fine_caption_egolife.py` 生成 30sec caption → `multiscale.py:39-117`（windows `180,600,3600`→`3min/10min/1h`，LLM 逐级摘要，`perspective=egocentric`）→ `extract_episodic_triples.py`（OpenIE 事件三元组）。索引：每粒度一个 HippoRAG（`episodic/memory.py:120-127, 186-233`）。
- Semantic：`extract_semantic_triples.py` 抽高层三元组 → `consolidate_semantic_memory.py`（`semantic_consolidation.py:23-93`：embedding 找相似已有三元组，阈值 0.6，LLM 合并/去重，累积更新成 habit/relation 知识）。
- Visual：`extract_visual_features.py` 用 VLM2Vec-V2 对 30s clip 抽特征 → `visual_embeddings.pkl`。
- 多时间尺度索引：检索时各粒度分别取候选（`episodic/memory.py:247-322`，默认 `30sec:10,3min:5,10min:5,1h:3`）。

**Stage 2 — 自适应检索 agent（在线，`memory/memory.py:408-554`）**
- `answer()` 进入 while 循环（`max_rounds=5`, `max_errors=5`，:451）。每轮 reasoning LLM 读 query+choices+round history，按 `memory_reasoning.py` 输出 JSON：`decision=search/answer`；若 search 则选 `memory_type∈{episodic,semantic,visual}` 与 `search_query`（:474-496）。
- 路由检索 `retrieve_from_episodic/semantic/visual`（:274-406），结果累积进 `retrieved_items`+`round_history`，去重（`retrieved_set`）。视觉支持文本关键词或 `DAY X HH:MM:SS - DAY Y ...` 时间范围抽帧（`visual/memory.py:319-333`）。
- 各源机制：episodic = HippoRAG 多粒度候选 + LLM `multiscale_filter` 二次排序（`episodic/memory.py:340-423`）；semantic = embedding 取 top-相似三元组做 personalization，再在实体图上跑 **Personalized PageRank**（damping 0.85）按 subj+obj PPR 分排序（`semantic/memory.py:243-341`）；visual = VLM2Vec 跨模态余弦相似 top-k clip 抽帧（`visual/memory.py:344-409`）。

**Stage 3 — 响应生成（`memory/memory.py:556-593`）**
- agent 判 `answer` 或达上限后退出循环，渲染 `qa`/`qa_egolife` 提示（`templates/qa_egolife.py`），把累积文本三元组/caption + 视觉帧（`_render_retrieved_items_for_qa`，:248-272，文本→text、视觉→image）拼成多模态消息，调 respond LLM 输出单字母选项；`eval.py:193` 判分。

---

## 4. 一条真实数据的全过程（"周三下午那人为什么搬走红箱子"类）

以 EgoLifeQA 多选题为例：
1. **构建（离线）**：该周视频已切 30s/3min/10min/1h caption（如 `[DAY3 14:xx] 某人抱起红色箱子走向门口`），episodic 建多粒度 HippoRAG 事件图；semantic 累积 `(该人, 负责, 搬运杂物)`、`(红箱子, 属于, 储藏室)` 等 habit/relation 三元组；visual 对相关 30s clip 存 VLM2Vec embedding。
2. **检索（agent 迭代，`answer()`）**：
   - Round1：→`search/episodic`，query `"person moves red box afternoon"` → HippoRAG 多粒度召回 + multiscale_filter 选出 DAY3 下午相关 caption（命中搬走红箱子事件，但无动机）。
   - Round2：信息不足→`search/visual`，query 可关键词 `"red box being carried"` 或时间范围 `DAY3 14:30:00 - DAY3 14:31:00` → 抽 1fps 帧补充视觉细节。
   - Round3：→`search/semantic`，query `"red box owner reason"` → PPR 召回 `(红箱子→储藏室)`、`(该人→整理房间习惯)` 给出"为什么"。
   - Round4：reasoning 判 `answer`，跳出循环。
3. **生成**：respond LLM 综合三源证据（文本三元组+caption+帧）→ 输出正确选项字母，`round_history`/`num_rounds` 记入结果（`eval.py:205-206`）。
> 项目页 Case Study 印证：仅 episodic 会漏细粒度物体属性或习惯性行为，agent 动态转 visual/semantic 才答对。

---

## 5. 模型 / 组件

- **Caption/多尺度摘要 LLM**：默认 `gpt-5-mini`（`build_memory.py:181`；支持 `qwen3vl-2b/4b/8b`）。
- **检索 agent / reasoning + 多尺度过滤 + 抽取 LLM**：默认 `gpt-5-mini`（`eval.py:78`）。
- **响应生成 VLM**（需读视觉帧）：默认 `gpt-5`，`fps=1`（`eval.py:79,93`）；亦支持 `qwen3vl-8b`。
- **文本 embedding**：`Qwen/Qwen3-Embedding-4B`（`embedding_wrapper.py:10`）。
- **视觉 embedding**：`VLM2Vec/VLM2Vec-V2.0`（Qwen2VL-2B backbone，`vlm2vecv2.py:23`）。
- **episodic 检索后端**：HippoRAG（每粒度一实例）；semantic 用 igraph Personalized PageRank。

---

## 6. 创新点

1. **多模态多记忆构建**：同时建 episodic（多时间尺度文本事件图）+ semantic（持续合并更新的知识/习惯图谱，`semantic_consolidation.py`）+ visual（VLM2Vec embedding + 时间戳精确抽帧），兼顾文本抽象与视觉细节，克服纯文本 baseline（M3-Agent）丢失视觉证据的问题。
2. **自适应检索 agent**：reasoning agent 按 query 与检索历史**迭代**地选记忆源与时间粒度、自判信息是否充分再停止（`memory.py:408-554`），突破固定时间尺度/单一记忆源（EgoRAG）的限制。
3. **多时间粒度索引 + LLM 二次过滤**：episodic 在 `10sec~1h` 多粒度并行召回、用 multiscale_filter 重排，灵活覆盖变长事件；整体在五个 long-video QA benchmark 上**平均较 SOTA 提升 8.4%**。

> 说明：两条评测管线（EgoLifeQA、Video-MME）在 repo 中确证；论文称"五个 benchmark"，另三个具名因网络限制未取得，以论文表述（5 个、+8.4%）为准。
