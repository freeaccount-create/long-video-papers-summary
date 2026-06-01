# 02 · PerceptionLM (PLM)

> NeurIPS 2025 Spotlight · Meta FAIR · [arXiv 2504.13180](https://arxiv.org/abs/2504.13180)
> 官方代码：[github.com/facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)（同时含 Perception Encoder 与 PLM）

基于**官方源码逐行分析** + HuggingFace `facebook/PLM-Video-Human` 真实样本写成。

> 注意：repo 含训练/数据加载管线与已发布标注 JSONL 格式，但**合成数据"data engine"（场景切片→帧 caption→视频 caption→LLM 融合）的生成脚本未开源**（仅论文 Sec.3 描述），repo 只提供加载这些数据的消费端代码。

---

## 1. 源码结构

- `apps/plm/`：PLM 主体——`train.py`、`transformer.py`(`LMTransformer`)、`tokenizer.py`(`PLMTokenizer`)、`generate.py`、`configs/{stage_1,stage_2,stage_3}/plm_{1b,3b,8b}.yaml`。
- `core/vision_encoder/`：Perception Encoder——`pe.py`(`VisionTransformer`)、`config.py`、`rope.py`(2D RoPE)。
- `core/vision_projector/mlp.py`：connector（`MLPProjector` + `AdaptiveAvgPooling`）。
- `core/data/`：`data_mixer.py`、`preprocessor.py`、`conversation.py`、`data_collators.py`。
- `core/transforms/`：`image_transform.py`、`video_transform.py`、`region_transform.py`。

---

## 2. 数据来源与真实格式

统一 JSONL，核心字段（`core/data/preprocessor.py:38-108`）：`image`/`video`/纯文本 + 可选 `start_time`/`end_time`/`bbox_map`/`bbox`+`width`+`height`；`conversations: [{"from":"human"|"assistant","value":...}]`（`conversation.py:9-18`）。

**2.8M 人工标注 = `facebook/PLM-Video-Human`**（FGQA 2.32M + RCap/RTLoc 各 179K + RDCap 117K）。真实样本：

- **FGQA**（细粒度视频 QA）：
```json
{"qa_id":"130ae268-...","segment_id":"01651739-...",
 "question":"What is the initial state of the cabbage before you begin chopping it?",
 "answer":"cabbage is half cut already and kept on cutting board ...",
 "metadata":{"source_video_id":"-eyDS81FADw","source_dataset":"youcook2",
   "source_start_time":62.0,"source_end_time":77.0,"q_type":"Object State",
   "q_subtype":"initial_end_state","domain":"Cooking and Recipes","is_audited":0}}
```
- **RCap**（grounded 区域视频描述）：
```json
{"uid":0,"video":"sav_017599.mp4","masklet_id":2,"total_frames":73,
 "caption":"A boy enters the frame from the right, he wears glasses and turn back ...",
 "start_frame":30,"end_frame":72}
```
- **RTLoc**：与 RCap 同字段，但 caption 作**输入**、`start/end_frame` 作**输出目标**（RCap 的逆任务）。
- **RDCap**（区域稠密时序描述）：`dense_captions:[{start_frame,end_frame,caption}]`，覆盖全片含主体不可见段。

其余：合成数据 `facebook/PLM-Image-Auto`（SA1B/OpenImages/Obj365/ArxivQA 等）、`facebook/PE-Video`（1M 视频）。

---

## 3. 完整方法 / 训练流程

**架构组装**（`apps/plm/transformer.py`，`LMTransformer(BaseTransformer)`）：`__init__` 同时构造 `tok_embeddings`(line 99)、视觉塔 `self.vision_model = PE_VisionTransformer(...)`(line 122)、connector `self.vision_projector = MLPProjector(args)`(line 123)。connector（`vision_projector/mlp.py:30-62`）= `AdaptiveAvgPooling(pooling_ratio)` → `Linear(width→dim)+GELU+Linear(dim→dim)`，把 PE 的 1536 维投到 LLM 的 4096 维。冻结策略由 `train()`(`transformer.py:128-137`)逐参数设 `requires_grad`。

**视频帧采样**：`core/transforms/video_transform.py:79-130`，`load_video` 按 `sampling_fps` 抽帧，超 `max_frames` 用 `uniform_sample`，支持 `start/end_time` 截取与 `bbox_map` 画框（用于区域任务）。

**三阶段训练**（`configs/stage_{1,2,3}/plm_8b.yaml`）：

| 阶段 | 冻结 | pooling | max_seqlen | tiles/frames | LR | steps | 数据 |
|---|---|---|---|---|---|---|---|
| 1 warmup | LLM+PE 全冻，仅训 projector | 1 | 1280 | 1 tile, 8 帧 | 1e-4 | 8000 | 合成 SA-1B |
| 2 | 全解冻 | 2 | 6144 | 16 tiles, 16 帧 | — | — | 大规模混合 |
| 3 SFT | 全解冻 | 2 | 11520 | 36 tiles, 32 帧 | 1e-5 | 21000 | SFT 混合（含 2.8M 人工标注）|

初始化：LLM 载 `Llama-3.1-8B-Instruct`，视觉塔载 `PE-Core-G14-448`（`stage_1/plm_8b.yaml:73-74`）。

---

## 4. 一条真实数据的全过程（以 §2 的 RCap `sav_017599.mp4` 为例）

1. **混采**：`DatasetMixer.__iter__`（`data_mixer.py:390-408`）按权重选源，逐行取出该 JSON。
2. **构造 conversation**：RCap 已转成"问区域+时段 caption"的对话，区域信息经 `transform["region"]`（`preprocessor.py:45-51`）。
3. **视频解码与抽帧**：`preprocessor.py:85-101` 把 `video`+`start_time/end_time/bbox_map` 交给 `video_transform`，返回张量 `media`，形状 `(T,3,448,448)`。
4. **多模态分词**：`PLMTokenizer.__call__`（`tokenizer.py:209-279`）。每帧视觉 token 数 `(448/14/pooling_ratio)^2`（stage-3 `pooling_ratio=2` ⇒ 每帧 `(32/2)^2=256`）；`conversation.py:72-76` 在文本插入对应数量 `<|image|>`；输出 `text_ids`、`image_pos`、`response_pos`（只标 assistant 回答）。
5. **collate**：`MllmPaddingCollator`（`data_collators.py:67-132`）pad，`label_ids` 仅对 `response_pos` 位置有效（**仅对回答算 loss**）。
6. **前向**：`LMTransformer.forward`（`transformer.py:139-183`）：`h=tok_embeddings(x)` → `h_img=vision_model(images, strip_cls_token=True)` → `h_img=vision_projector(h_img)` → `stitch_images_into_text` 用 `image_pos` 把视觉特征写回 `<|image|>` 槽位 → 过 LLM → `output(norm(h))`。
7. **loss**：`logits=logits[loss_mask]; target=target[loss_mask]` 再 `cross_entropy`（`transformer.py:178-181`）。RTLoc 则相反——caption 入文本、`start/end_frame` 作生成目标。

---

## 5. 模型 / 组件

- **Perception Encoder**：`PE-Core-G14-448`（`image_size=448, patch_size=14, width=1536, layers=50, heads=16`，2D RoPE）。PLM 训练**只取前 47 层**（`stage_*/plm_8b.yaml:43` `layers:47`），`use_cls_token=false`。
- **LLM decoder**：Llama-3 系。PLM-8B：`dim=4096, n_layers=32, n_heads=32, n_kv_heads=8`(GQA), `vocab=128256`, RMSNorm+SwiGLU；另有 1B/3B（Llama-3.2）。
- **Connector**：`MLPProjector` = AdaptiveAvgPool2d(`pooling_ratio=2`，每帧 32×32→16×16=256 token) + 两层 MLP（1536→4096→4096）。

---

## 6. 创新点

1. **PE→LLM 解耦式装配 + 中间层截断**：复用 SOTA 对比视觉塔 PE-Core-G14，截断到第 47 层、去 CLS、`pooling_ratio=2` 自适应池化压缩视觉 token，轻量 2 层 MLP 对齐 LLM，可扩展到 32 帧×36 tile。
2. **最大规模人工时空标注 + 四类新任务**：发布 2.8M 人工标注（FGQA + RCap/RTLoc/RDCap），首创"masklet+时段"驱动的互逆任务与全片稠密时序描述。
3. **统一可复现的多模态训练/数据管线 + 三阶段课程**：单一 `LMTransformer` 承载视觉塔+connector+LLM，统一 JSONL 协议覆盖图/多图/视频/区域/纯文本，三阶段课程（仅训 projector → 全解冻预训练 → SFT，seqlen 1280→6144→11520）全用开源数据。
