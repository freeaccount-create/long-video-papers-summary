# 10 · Seeing, Listening, Remembering, and Reasoning（M3-Agent）

> ICLR 2026 · [arXiv 2508.09736](https://arxiv.org/abs/2508.09736) · [项目页](https://m3-agent.github.io/) · 官方代码：[github.com/bytedance-seed/m3-agent](https://github.com/bytedance-seed/m3-agent)

基于**官方源码逐行分析**写成（克隆成功，Apache-2.0）。

> 训练代码不在本仓：`README.md:190-193` 指向外部 `hyc2026/sft-qwen2.5-omni-thinker`（记忆 SFT）与 `hyc2026/M3-Agent-Training`（控制 RL/DAPO）。本仓含推理/建图/检索代码，DAPO 公式/消融数值引自论文正文。

---

## 1. 源码结构

- `m3_agent/`：`memorization_intermediate_outputs.py`（抽人脸/声纹）、`memorization_memory_graphs.py`（建记忆图）、`control.py`（多轮检索问答+评测）。
- `mmagent/`：`videograph.py`（记忆图结构）、`face_processing.py`、`voice_processing.py`、`memory_processing_qwen.py`（SFT 模型生成 episodic/semantic）、`retrieve.py`（检索）、`prompts.py`（全部 prompt）；`mmagent/src/`（人脸抽取/聚类）。
- `data/annotations/{robot.json, web.json}`：M3-Bench QA 标注。

---

## 2. 数据 / 输入格式

**M3-Bench-robot**（`data/annotations/robot.json`，100 视频）顶层 `{video_id: {video_path, mem_path, qa_list}}`。真实样本（`living_room_06`）：
```json
"living_room_06_Q09": {
  "question": "Where is the yoga mat located?",
  "answer": "Inside the storage room",
  "reasoning": "...请机器人帮忙收拾瑜伽垫到储物间...",
  "timestamp": "13:13",
  "type": ["General Knowledge Extraction"],
  "before_clip": 25
}
```
**M3-Bench-web**（`web.json`，920 个）多一个 `video_url`，QA 无 `before_clip`。`before_clip` 关键：限定只能检索该 clip 及之前的记忆（`control.py:119`、`retrieve.py:132-133`）。

**记忆图节点表示**（`mmagent/videograph.py`，序列化为 `.pkl`）。`VideoGraph` 含 `nodes`、`edges`（双向带权）。`Node.type` 四种（`videograph.py:66`）：
- **实体节点**：`'img'`（人脸，ArcFace embedding + base64 人脸图，`add_img_node` L118）、`'voice'`（声纹，ERes2NetV2 embedding + ASR 文本，`add_voice_node` L138）。
- **文本记忆节点**：`'episodic'` 与 `'semantic'`（`add_text_node` L158-187，存 text embedding + contents + `timestamp=clip_id`）。
- **entity-centric**：文本节点通过边连到 img/voice 实体节点（`memory_processing_qwen.py:194-195` 解析文本里的 `<face_x>/<voice_y>` 连边）；同型文本节点之间不允许连边（`add_edge` L231）。

---

## 3. 完整方法流程

### 记忆进程（memorization）
入口 `m3_agent/memorization_memory_graphs.py`，30 秒切片流式处理（`streaming_process_video` L69-100）。

1. **在线抽实体**：
   - 声纹：`voice_processing.py:process_voices` — Gemini-1.5-pro 做 ASR+说话人分段（`diarize_audio` L136-166），按时间切音频段，过 ERes2NetV2 取 192 维声纹 embedding，`update_videograph` 跨 clip 匹配。
   - 人脸：`face_processing.py:process_faces` — InsightFace `buffalo_l`（RetinaFace 检测 + ArcFace embedding，L25）抽脸，HDBSCAN 聚类（检测分≥0.8、质量分≥20 过滤），`update_videograph` 匹配/新建节点。
2. **跨模态身份一致性**：clip 内新实体与已有同型节点算余弦相似超阈值则合并（`search_img_nodes` 0.3、`search_voice_nodes` 0.6）；跨模态人脸↔声纹绑定靠 MLLM 在 semantic 输出 `Equivalence: <face_x>, <voice_y>`（`prompts.py:314`），`refresh_equivalences`（`videograph.py:417-513`）用**并查集**合并成 `character_i`，`fix_collisions` 保证每个 voice 至多一条 face-voice 映射。
3. **episodic/semantic 双层记忆图构建**：`memory_processing_qwen.py:generate_memories` 把"视频片段 + 带框人脸图(`<face_x>`) + 带时间戳 ASR(`<voice_y>`)"喂 M3-Agent-Memorization（Qwen2.5-Omni-7b SFT），一次产出 `video_descriptions`(episodic) 与 `high_level_conclusions`(semantic, 含 Equivalence)。`process_memories`（L183-254）：episodic 每条都 `insert_memory` 新建节点并连实体边；semantic 解析首个实体的已连 semantic 节点，子集且相似度 >0.85 → `reinforce_node`(+1)，<0 → `weaken_node`，否则新建——实现"语义记忆随时间强化/遗忘"。全部 clip 处理完 `refresh_equivalences()` 后 `pickle.dump`。

### 控制进程（control）— DAPO 训练的多轮检索
入口 `m3_agent/control.py`，vLLM 加载 `M3-Agent-Control`（L141）。多轮循环 `total_round=5`：
- system prompt 要求每轮输出 `Action: [Answer]/[Search]` + `Content:`，可 `<think>` 推理（`enable_thinking=True`）。
- `consumer`（L100-132）解析：`[Answer]`→终止；`[Search]`→调 `retrieve.search` 检索记忆，结果作 `user` 消息 "Searched knowledge: ..." 追加进对话。
- 检索 `retrieve.py:search`（L237-275，底层 `retrieve_from_videograph` L76-136）：query 经 `back_translate`（把 `character_i` 展开成所有 face/voice 变体）→ text-embedding-3-large 编码 → `search_text_nodes` 算 clip 级相似度 → 取 topk clip 的记忆，`translate` 再把实体 ID 回译。两种模式：`mem_wise`（查"character id↔姓名"映射）与 clip-wise（普通事件检索）。最后一轮强制 `[Answer]`。答案用 GPT-4o evaluator 判对错。

  **检索核心 = 两级 max 聚合（精确计算）**：
  1. **query 展开**（`back_translate` L50-73）：一条 query 里若含 `character_i`，按 `character_mappings` 笛卡尔展开成所有 `face_x/voice_y` 变体 query（一个角色对应多张脸/多段声纹时，query 数成倍增长），保证任一身份表述都能命中。展开后的多条 query 各自编码成 `embedding`，构成 `query_embeddings∈R^{Q×E×d}`（Q 条 query、每条 E 个 embedding）。
  2. **第一级 max：embedding→节点分**（`search_text_nodes` `videograph.py:554-612`）：每个 text node 自身存多个 embedding。对全部 `cosine_similarity(query, node)` 算出 `similarities∈R^{Q×N×E'}`（:594-596），再 `np.max(axis=(0,2))`（:608）**沿 query 维与 embedding 维同时取最大**，把每个节点压成一个标量分——即"任一 query 的任一 embedding 与该节点任一 embedding 最相似"的程度（mode='max'）。
  3. **第二级 max：节点→clip 分**（`retrieve.py:110-127`）：按节点 `metadata['timestamp']`(=clip_id) 把节点分归桶，`clip_score = max(scores)` 取该 clip 内最高分节点作 clip 分。
  4. **因果裁剪 + topk**（:130-135）：`before_clip` 限定只保留 `clip_id <= before_clip` 且 `score >= threshold` 的 clip，按分排序取前 topk——这是"只能检索当前及之前记忆"的硬约束实现处。
- **训练**（外部 repo + 论文 §4.4）：control 策略从基座 Qwen3 用 **DAPO** 训练；reward = GPT-4o 评测对=1/错=0（式(1)）；组内 G 条轨迹做 group-normalized advantage `(R_i-mean)/std`（式(2)），只对生成 token 算 loss，DAPO 带 clip-higher 等。论文 Table 7 证实 DAPO 全面优于 GRPO 且收益随规模放大（32b +10.0%/8.0%/9.3%）。

---

## 4. 一条真实数据的全过程（"瑜伽垫放哪了"类，对应"张三把外套放哪了"）

以 `living_room_06_Q09: "Where is the yoga mat located?"`（before_clip=25）：

1. **视听流**：`living_room_06.mp4` 切 30s clips，clip 0~25 逐段流式处理。
2. **实体抽取**：每段 RetinaFace+ArcFace 抽脸→HDBSCAN 聚类→匹配/新建 `img` 节点（`<face_0>`=Lily）；Gemini ASR 分段→ERes2NetV2 声纹→`voice` 节点（`<voice_3>`），跨 clip 用 0.3/0.6 阈值保持 ID 一致。
3. **记忆图节点**：Qwen2.5-Omni 对收瑜伽垫那段生成 episodic（`"<face_0> hands the yoga mat to the robot, asking to put it into the storage room"`，连边 `<face_0>`）+ semantic（`"Equivalence: <face_0>, <voice_3>"`、`"<voice_3>'s name is Lily"`）。`refresh_equivalences` 把 face_0/voice_3 并成 `character_x`。
4. **多轮检索**（限定 before_clip=25）：轮1 `[Search] "Where is the yoga mat placed?"` → `back_translate`→embedding→命中收拾瑜伽垫的 clip，返回 "...put it into the storage room"；若需姓名↔ID 映射，发 `"What is the character id of Lily"`（触发 `mem_wise`）。
5. **答案**：`Action: [Answer] Content: Inside the storage room`，与 GT 一致，GPT-4o 判 True。

---

## 5. 模型 / 组件

| 环节 | 模型/组件 | 出处 |
|---|---|---|
| 人脸检测+embedding | InsightFace `buffalo_l`（RetinaFace + ArcFace） | `face_processing.py:25` |
| 人脸聚类 | HDBSCAN（precomputed cosine） | `src/face_clustering.py:49` |
| ASR + 说话人分段 | Gemini-1.5-pro-002 | `voice_processing.py:148` |
| 声纹 embedding | ERes2NetV2（192d，speakerlab） | `voice_processing.py:35-47` |
| 记忆生成 MLLM | M3-Agent-Memorization = Qwen2.5-Omni-7B SFT | `configs/processing_config.json`、`README.md:184` |
| 文本 embedding | OpenAI text-embedding-3-large | `retrieve.py:96` |
| 控制策略 MLLM | M3-Agent-Control = Qwen3-32b + DAPO（vLLM） | `control.py:32,141` |
| RL 算法 | **DAPO**（reward=GPT-4o 0/1，group-normalized advantage） | 论文 §4.4 |
| 答案评测 | GPT-4o-2024-11-20 | `control.py:34` |

---

## 6. 创新点

1. **Entity-centric 多模态长时记忆图 + 双层记忆**：以人脸/声纹实体节点为锚，episodic（原子事件）与 semantic（高层结论/世界知识）双层文本节点挂在实体上，semantic 随时间 reinforce/weaken。**消融：移除语义记忆使准确率降 17.1%/19.2%/13.1%**（robot/web/VideoMME-long）。
2. **在线跨模态身份一致性**：MLLM 生成 `Equivalence: <face>,<voice>` + 并查集统一为 `character_i`，并用 meta-clip 自动挖掘全局 face-voice 映射。去掉 equivalence 降 2.0%/2.6%/9.1%。
3. **DAPO 强化学习驱动的多轮迭代检索**：把"思考-检索-再思考"建模为 RL（reward=最终答案 GPT-4o 判对），DAPO 优于 GRPO 且收益随规模放大；整体超越 Gemini-1.5-pro+GPT-4o 最强 prompting baseline 8.2%/7.7%/5.3%（robot/web/VideoMME-long，`README.md:15`）。
