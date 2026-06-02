# 07 · SAMA（Referential Grounded Video Chat）

> NeurIPS 2025 · [arXiv 2505.18812](https://arxiv.org/abs/2505.18812) · 官方代码：[github.com/sunye23/SAMA](https://github.com/sunye23/SAMA)

基于**官方源码逐行分析**写成（克隆成功，192 文件）。

---

## 1. 源码结构

- `projects/llava_sam2/models/internvl_sama.py`：**核心模型 `InternVL_Slowfast`**，含时空上下文聚合器（:346-595）。
- `projects/llava_sam2/models/llava_sam2.py`：顶层 `VideoLLaVASAMModel_zero3`，`[SEG]`→SAM 分支与损失。
- `projects/llava_sam2/models/sam2_train.py` / `extension/sam2_base.py`：SAM2 接入。
- `projects/llava_sam2/models/qformer.py`：改造版 BERT Q-Former。
- `projects/llava_sam2/datasets/VideoRegionConv_Dataset.py` / `VideoRegionCaptioning_Dataset.py`：**SAMA-239K 加载器**。
- `annotation/automatic_annotation.py` + `system_message.txt`：数据自动标注流程。

> 注：repo 内**未附 SAMA-239K 示例 json**，真实字段格式从加载器字段访问与标注 prompt 反推。

---

## 2. 数据来源与真实格式

**来源**：SAMA-239K 复用现有 VOS/VIS/grounding 数据集的 mask 标注（mevis/lvvis/ref_youtube_vos/sav/VidSTG），再用 Gemini 类模型自动生成 grounded 对话——把每个目标用彩色框+`<objN>` 标签画在帧上，要求模型输出 description 与 conversation（`annotation/system_message.txt:1-71`）。

**JSON 字段格式**（从 `VideoRegionConv_Dataset.py:262-270` 反推，每条=一个视频）：
```
{ "video": "mevis/.../video_xxx", "height":H, "width":W,
  "frames": [...帧文件名...],
  "anno_map": [obj_id...],          # 指向各 mask_dict.json 里的 RLE key
  "obj_masks":  [[id,...], ...],    # 每轮 QA：问句 <mask> 引用的 referent 目标 id
  "groundtruth":[[[id],...], ...],  # 每轮 QA：答句 [SEG] 要分割的目标 id
  "conversation":[ {"from":"human","value":"...<obj1>..."},
                   {"from":"gpt","value":"<p>the man</p> <obj1> ... <seg>"} ] }
```
mask 本体在外部 `*_mask_dict.json`，按 `anno_map` 取 **COCO RLE**，逐帧 `maskUtils.decode`（`VideoRegionConv_Dataset.py:493-525`）。

**referent / grounding token 标法**：
- **referent token（被指目标）**：原始答句 `<p>描述</p> <objN>`，问句 `<objN>`；加载时目标区域以 `<mask>` 占位（`VideoRegionCaptioning_Dataset.py:522-540`）。
- **grounding token**：答句原始 `<seg>`，加载时 `answer.replace('<seg>','[SEG]')`、`<g_s>→<p>`、`<g_e>→</p>`（`VideoRegionConv_Dataset.py:753`）。`[SEG]` 即触发 SAM 的 grounding token。

---

## 3. 完整方法 / 训练流程

**(a) 被框区域 → referent token（时空上下文聚合器）**，`spatial_temporal_token_generation`（`internvl_sama.py:346-551`）：
1. **Spatial Q-Former**：InternViT 帧特征经 `vlm2spatial_Qformer_proj`→1408 维，与 32 个 query token 做 cross-attn，每帧压成 32 token（:481-495）。
2. **Mask-Pooling 区域特征（双路精确计算）**：
   - *路 A — mask-pooling 语义特征*：`MaskPooling.forward`（`internvl_sama.py:48-64`）。先把 mask 双线性插值到帧特征 `x∈R^{b×c×h×w}` 的分辨率（:51-52），二值化 `mask=(mask>0)`（:60），算每个目标的可见面积 `denorm = mask.sum(dim=(-1,-2)) + 1e-8`（:61）。核心是一行 einsum `"bchw,bqhw->bqc"(x, mask/denorm)`（:62-65）——等价于**对 mask 覆盖像素做面积归一化的加权平均**，把第 q 个目标在该帧的所有特征向量塌成一个 `C` 维区域语义向量 `R^{b×q×c}`。这是"被框区域 → 一个向量"的关键算子，再经 `Qformer_mask_pooling_proj_st` 投影。
   - *路 B — 框形状特征*：把目标框下采样到 112×112，经 `Qformer_mask_proj_st` MLP 编码出几何/形状特征（`:417-435`），与路 A 互补（语义 vs 位置形状）。
   - *注入*：两路经 `<mask>→<vp><mask><pos></vp>` 注入 **Temporal Q-Former** 的 `<mask>`(id 30523, `VISUAL_TOKEN_ID`, 注入路 A mask-pooling 视觉特征)/`<pos>`(id 30524, `MASK_TOKEN_ID`, 注入路 B 112×112 框形状特征) 位置（`qformer.py:72-73, 128-141`，BERT embedding 查表时按这两个 token id 替换成对应区域特征）。
3. **Temporal Q-Former**（2 层、32 query）跨帧聚合，按 `window_size=512` 分窗（:519-544）。
4. **`temporal_context_inject`（精确注意力，:553-595）**：把 temporal Q-Former 聚合的 `temp_embed` 注入回逐帧视觉特征 `vis_embed`，得最终 referent token。具体（以 `num_prompts≤chunk` 分支 :578-594 为例，>chunk 时按 16 个 prompt 分块同算）：
   - **温度 token 沿帧展开**：`temp_embed.unsqueeze(1).expand(-1,image_count,-1,-1)` 把每个目标的 temporal token 复制到该目标的每一帧（:579）。
   - **投影成 Q/K/V**：`query=Qformer_temp_attn_q(vis_embed)`，`key=Qformer_temp_attn_k(temp_embed)`，`value=Qformer_temp_attn_v(temp_embed)`（:582-584）——**原帧视觉特征作 Query，时序聚合 token 作 Key/Value**（与常规 Q-Former 的 query 当 Q 相反，这里让每帧像素"去问"时序上下文）。
   - **缩放点积注意力**：`ctx = softmax( (Q·Kᵀ)/√d ) · V`（:586-588，`d=key.shape[-1]`）。
   - **残差 + 投影**：`ctx = Qformer_temp_proj(ctx) + vis_embed`（:589-590，残差保留原帧信息）。
   - **时间维 mean + 终投影**：`ctx = mean(ctx, dim=1)` 把同一目标跨帧压成 1 个 token（:591），再 `Qformer_final_proj`（:592）得最终 **referent token** `(num_prompts, 1, C)`，splice 进 LLM 输入 embedding 的 `<vp>...</vp>` 位置（:786-857）。

**(b) LLM 输出 grounding token → SAM 出 mask**（`llava_sam2.py:371-416`）：LLM 生成含 `[SEG]` 的答句 → `seg_token_mask = input_ids==seg_token_idx` → 抽最后层 hidden 过 `text_hidden_fcs` MLP 投到 SAM 维度得 `language_embeddings` → SAM2 图像编码 `get_sam2_embeddings`（`sam2_train.py:104`）→ `inject_language_embd` 把它拼到 `sam_prompt_encoder` 的 sparse_embeddings 之后（`sam2_base.py:204-208`）→ `sam_mask_decoder` 输出 mask。

**(c) 训练阶段 / 损失**（`llava_sam2.py:441-460`）：联合损失 = **LLM 自回归 CE + SAM mask CE（weight 2.0）+ DiceLoss（weight 0.5）**。基座 LLM/ViT 冻结（`freeze_llm=True, freeze_visual_encoder=True`），**SAM2 decoder 解冻**（`frozen_sam2_decoder=False`），训练聚合器(Q-Former)+`text_hidden_fcs`+SAM decoder。8×A100-80G。

---

## 4. 一条真实数据的全过程（多轮 referring+grounding 例）

以"框出某球员问他传给谁 / 把某人整段画出来"为例：

1. **框选区域**：问句 `"Who does the player <obj3> pass to?"`。`obj_masks=[[3]]` 取 mevis `mask_dict["3"]` 逐帧 RLE → `decode_mask` 得 `(n_obj, n_frames, H, W)`；同时生成 `prompt_masks`(patch 网格) 与 `prompt_masks_112`(框 112×112)。问句 `<mask>` → `<vp><IMG_CONTEXT><IMG_CONTEXT></vp>`。
2. **referent token**：聚合器对 obj3 区域跨帧 mask-pooling+box112 编码 → spatial→temporal Q-Former → `temporal_context_inject` 得 1 个 referent token `(1, C)`（8B 时 C=4096），splice 进 `<vp>` 两个 `<IMG_CONTEXT>` 位。
3. **LLM 回答**：LLM 读"帧 token + referent token + 文本"，输出 `"<p>the teammate</p> <obj7>, ... [SEG]"`。
4. **grounding token**：对 `[SEG]` 抽 hidden→`text_hidden_fcs`→`language_embeddings`，shape `(num_objs*num_frames, 1, 256)`。
5. **SAM mask**：`g_pixel_values` 经 SAM2 编码，`inject_language_embd` 拼入 sparse embedding，`sam_mask_decoder` 输出 `pred_masks (num_frames, num_objs, h, w)`，与 `groundtruth` RLE GT 算 mask+dice loss。第二轮"把 obj7 整段画出来"复用同一视频特征再走一遍 `[SEG]`→SAM。

---

## 5. 模型 / 组件

- **Base LLM + 视觉编码器**：以 **Sa2VA**（`Sa2VA-1B/4B/8B`）为底座，即 **InternVL2.5**（LLM=InternLM2 或 Qwen2，视觉编码器 InternViT）。`out_channels` 1B=896 / 4B=2048 / 8B=4096。
- **SAM 版本**：**SAM2 Hiera-Large**（`sam2_hiera_l.yaml` / `sam2_hiera_large.pt`，`hidden_dim=256`）。
- **聚合器**：两级 **BERT-Q-Former**——Spatial(32 query, vision_width 1408，InstructBLIP 权重初始化) + Temporal(32 query, 2 层) + MaskPooling + 112×112 box-MLP 双路 + `temporal_context_inject`。

---

## 6. 创新点

1. **统一"指代理解+视觉接地+多轮对话"于单模型**：`<vp><mask></vp>` referent token（输入侧区域指代）与 `[SEG]` grounding token（输出侧触发 SAM2）在同一自回归 LLM 内闭环，支持多轮（`MAX_CONV_LEN=6`）。
2. **可通用的时空上下文聚合器**：mask-pooling 视觉特征 + 112×112 框形状特征双路 → spatial/temporal 双级 Q-Former → 注意力式 `temporal_context_inject`，把任意被框区域压成少量 referent token，与冻结的 InternVL/SAM2 解耦。
3. **SAMA-239K 数据集 + 自动标注流水线**：复用 5 个 VOS/grounding 数据集 mask，借多模态模型按"彩框+`<objN>`标签"自动生成 grounded 多轮 QA，规模化产出 15K 视频（约 23.9 万 QA/对话样本，故名 SAMA-239K；`README.md:42` 仅标 15K videos），并配套 SAMA-Bench（5067 题/522 视频）。
