# DirectorBench — 分钟级视频生成的多智能体诊断式评测框架

> arXiv 2605.30090 · 源码 [github.com/jiaminchen-1031/DirectorBench](https://github.com/jiaminchen-1031/DirectorBench) · 类型：Benchmark / 评测框架（LangGraph 多智能体 DAG）
>
> 本文以仓库内**第一条真实样本** `DB_001`（动作类「屋顶追逐战」，67s，中文）走完整条评测流水线，并给出 `file:line` 级引用。所有引用均指向克隆到本地 `lvp-src/directorbench/` 的真实源码。

---

## 1. 这个评测要回答什么问题

主流视频生成评测（VBench 等）大多是「单镜头 + 短视频 + 标量分」的范式：给一段 2–4s 的视频，跑若干个全局指标（运动平滑度、美学、文图一致），取平均。DirectorBench 处理的是一个不同量级的对象——**分钟级、多镜头、带剧本/对白/BGM 的"成片"**，并把评测本身组织成一个**像剧组复盘一样的诊断流程**：

- 不是只给一个总分，而是产出 **bottleneck（瓶颈短板）+ 可执行的修改建议 + 叙事化诊断报告**；
- 不是一把大模型梭哈，而是拆成 **5 个专科智能体**（剧本/视频/音频/稳定性/跨模态），每个智能体内部再拆成一组**带 Likert 评分锚点的 checkpoint**；
- 不是固定权重，而是用 **用户画像（profile）** 把"这个用户在乎什么"注入到加权方式里——故事派、视觉派、音画同步强迫症会得到不同的总分。

整套东西用 **LangGraph** 编排成一个有依赖关系的 DAG（`directorbench/graph.py`），而不是顺序脚本。

---

## 2. 源码结构

克隆后关键目录（`lvp-src/directorbench/`）：

```
directorbench/
  main.py                 # 入口：evaluate_video() / CLI / 批量
  graph.py                # LangGraph DAG 定义（5 智能体 + 编排 + 诊断）
  schemas.py              # GraphState / EvalResult / UserProfile / ContentProfile
  checkpoints.py          # CHECKPOINTS 注册表：每个子指标 = 一组 CheckpointDef
  preprocessing.py        # Phase 0：分镜检测/音频分离/ASR/转场度量
  config.py               # EvalConfig（阈值、grade 边界、LLM 配置）
  report.py               # ReportWriter：JSONL 追加 + 控制台摘要
  agents/
    base.py               # ★ checkpoint 评测引擎（所有专科 agent 的父类）
    script_agent.py       # 剧本维度
    video_agent.py        # ★ 视频维度（含 OpenCV 算法证据层）
    audio_agent.py        # 音频维度
    stability_agent.py    # 生成稳定性维度
    crossmodal_agent.py   # 跨模态对齐（barrier 后）
    diagnosis.py          # ★ 诊断综合器：维度分→总分→grade→瓶颈
data/
  metadata/dataset_original.jsonl   # ★ 75 条真实评测样本（含 DB_001）
  profiles.jsonl                    # 5 个用户画像
```

带 ★ 的四个文件是评测逻辑的核心，下面的真实数据全过程会反复回到它们。

---

## 3. DAG 拓扑：评测如何被调度

`build_eval_graph()` 用 LangGraph 的 `StateGraph` 把整个评测拼成四个阶段（`graph.py:34-146`）：

```
                 ┌── script_eval ──┐
 orchestrator ──┼── video_eval ───┼── crossmodal_eval ── diagnosis ── END
   (Phase 0)     ├── audio_eval ───┤      (Phase 2)        (Phase 3)
                 └── stability_eval┘
                      (Phase 1, 并行)
```

- **入口 → 编排器**：`workflow.set_entry_point("orchestrator")`（`graph.py:121`）。
- **Phase 0 → Phase 1 扇出**：编排器对四个专科 agent 各连一条边（`graph.py:125-128`），LangGraph 对互相无依赖的节点**自动并行**。
- **Phase 1 → Phase 2 barrier**：四条边全部汇入 `crossmodal_eval`（`graph.py:131-134`）——跨模态对齐必须等所有单模态结果就绪。
- **Phase 2 → Phase 3 → END**：`crossmodal → diagnosis → END`（`graph.py:137-140`）。

`orchestrator_node` 调用 `Preprocessor.run()` 做预处理（分镜、ASR、转场度量），把结果写进共享的 `GraphState.preprocessing` 并把工具调用记录传播到 `tool_context`，供下游 agent 看到哪些工具成功/失败（`graph.py:56-76`）。

入口函数 `evaluate_video()` 负责：建图 → `create_initial_state()` 造初始状态 → `graph.invoke()` 跑完 → 取 `final_state["diagnosis"]` → `ReportWriter.append()` 落 JSONL（`main.py:110-157`）。CLI 还会按 grade 决定退出码：`D/F` 退 1（`main.py:368-370`）。

---

## 4. 真实样本：DB_001「屋顶追逐战」

`data/metadata/dataset_original.jsonl` 的**第一条**记录就是我们要追踪的样本（节选其真实字段）：

```jsonc
{
  "meta_id": "action_006", "sample_id": "DB_001",
  "duration_sec": 67.0, "video_type": "动作类",
  "main_instruction": "屋顶追逐战（多角度快速切换+跳跃动作）",
  "modality_details": {
    "text": { "story_arc": {
        "act1_setup":      "英雄发现目标",
        "act2_conflict":   "屋顶追逐",
        "act3_resolution": "成功制服" },
      "script": [ { "shot_id": 1, "duration": 18,
                    "dialogue": "别跑！", "narration": "追逐开始" } ],
      "tone_requirements": "intense_exciting" },
    "visual": {
      "shots": [ { "shot_id": 1, "action": "屋顶跳跃",
                   "camera_movement": "tracking", "lighting": "high_contrast" } ],
      "camera_requirements": ["tracking", "whip_pan"],
      "consistency_requirements": ["spatial_layout", "momentum"] },
    "audio": { "dialogue": true, "lip_sync": true,
               "bgm_style": "strong_pulse", "sound_effects": ["footsteps","impact"],
               "tone_control": "fast_rhythm", "multi_language": "zh" }
  },
  "language": "zh", "variant_type": "original"
}
```

这条 metadata 是评测的"出题卡"：它声明了**这段视频本应做到什么**（三幕结构、tracking/whip_pan 运镜、强脉冲 BGM、脚步/撞击音效、中文对白+唇形同步）。被评测的对象则是某个生成模型针对 `main_instruction` 产出的 67s 成片 `video.mp4`。评测要做的，就是逐项核对成片与这张出题卡的差距。

### 4.1 选用户画像

我们用 `data/profiles.jsonl` 的 **Profile 1「Story-First」**（叙事优先）来评这条动作样本。它的关键字段：`priority_weights.text_story_arc = 0.55`（剧本权重最高），`hard_constraints` 含 `strong_three_act_arc`、`causal_logic`。CLI 上通过 `--profile-id 1` 加载（`main.py:334-338` → `_load_profile_by_id` `main.py:219-253`）。

`create_initial_state()` 检测到 dict 里有 `"personalization"` 键，就走 `UserProfile.from_profile_dict()` 解析（`graph.py:182-185`），否则当扁平 dict 处理。

---

## 5. Phase 0：编排器把成片"拆解成证据"

`orchestrator_node` 调 `Preprocessor.run()`（`graph.py:60-65`），对 DB_001 的 67s 成片做：

1. **分镜检测**（PySceneDetect）：把 67s 切成若干 shot，得到每个 shot 的起止时间——这是后面"跨镜头一致性"判断的基础。DB_001 的出题卡只给了 shot_id 1（18s），但成片实际会有更多镜头，分镜检测给出真实的镜头边界。
2. **音频抽取 + 分离**（ffmpeg + AudioShake）：把人声/BGM/音效分轨，供音频 agent 单独评 `bgm_consistency`、`narration_reasonableness`。
3. **ASR**（Azure Whisper）：转写对白，拿到 speaker 段落和说到的人名——后面构建 ContentProfile 时用正则从 ASR 文本里抽人名。
4. **转场度量**（OpenCV）：对每个镜头边界算 SSIM / 直方图差 / 光流，作为转场质量的"算法证据层"。

预处理产物（`shots`、`asr_segments`、逐转场度量）写进 `GraphState.preprocessing`，并把工具成功/失败记录塞进 `tool_context`（`graph.py:67-76`）。下游每个专科 agent 都能读到这份共享证据，避免重复解码视频。

---

## 6. checkpoint 评测引擎：每一分都怎么打出来

这是整个框架最值得细读的部分，全部在 `agents/base.py`。所有专科 agent 继承同一个引擎，区别只在"用哪一组 checkpoint"。

### 6.1 ContentProfile 门控（"这条样本该不该评这个 checkpoint"）

不是所有 checkpoint 都对所有视频适用。引擎先用 VLM + ASR 构建一个 `ContentProfile`（`_build_content_profile`，`base.py:229`），判断这段视频**有没有角色、有没有手持物体、有没有场景切换、有没有对白**等布尔属性。

每个 `CheckpointDef` 带一个 `applicable_when` 门控字典。例如 `char_face_consistency` 的门控是 `{"has_characters": True}`（`checkpoints.py:32`），`object_permanence` 是 `{"has_held_objects": True}`（`checkpoints.py:60`），`temporal_logic` 是 `{"has_scene_changes": True}`（`checkpoints.py:102`）。

`_filter_applicable()` 据此过滤（`base.py:383-391`）：

```python
return [cp for cp in checkpoints
        if not cp.applicable_when or profile.matches(cp.applicable_when)]
```

> **DB_001 的实际门控结果**：成片有英雄+目标两个角色 → `has_characters=True`，`char_face_consistency`、`char_clothing_consistency` **激活**；追逐戏里英雄基本没手持道具 → `has_held_objects=False`，`object_permanence` **被跳过**（不计入分母）；全片都在屋顶、无昼夜变化 → `has_scene_changes=False`，`temporal_logic` **被跳过**。这意味着 DB_001 的 `temporal_coherence` 维度实际只在 `char_face_consistency / char_clothing / background_consistency / scale_proportion / motion_continuity` 上打分。

### 6.2 单 checkpoint 评测：防御式、防自洽崩坏

`_evaluate_single_checkpoint()`（`base.py:769`）对一个 checkpoint 构造一段**"缺陷优先 / 怀疑论"**的提示词：要求 VLM 先主动找毛病，再给分，避免"看起来还行就给高分"的乐观偏差。它做了三件关键防护：

1. **id 不匹配重试**：VLM 返回的 `checkpoint_id` 与请求的不符时重试，防止模型答串题。
2. **factual_override**：对像 `duration_completeness` 这种可以用事实直接判定的 checkpoint（成片时长 vs 要求时长），用预处理拿到的真实时长**直接覆盖** VLM 的主观判断（`_factual_override_for_checkpoint`，`base.py:735`）。
3. **reasoning↔score 一致性检查**：`_check_reasoning_score_consistency()`（`base.py:476`）用正/负面措辞启发式 + 句级否定判断（`_is_negated_in_sentence`，`base.py:593`）检测"嘴上说一堆问题、分却给很高"的矛盾，触发重评。

### 6.3 分值归一化：Likert 与 Binary 两套尺子

每个 `CheckpointDef` 有 `checkpoint_type`。归一化在两处对称出现（`base.py:1057-1062` 单条、`base.py:1270-1275` 批量）：

```python
if cp.checkpoint_type == BINARY:
    normalised = float(raw_val)        # 0/1 → 0.0 / 1.0
elif cp.checkpoint_type == LIKERT:
    normalised = (raw_val - 1) / 4.0   # 1..5 → 0.0 / 0.25 / 0.5 / 0.75 / 1.0
```

Likert 的 1–5 配有锚点（`RubricAnchor`）。以 `char_face_consistency` 为例（`checkpoints.py:36-42`）：5=Perfect 像素级一致 / 3=Noticeable 比例或肤色漂移、身份存疑 / 1=Broken 前后镜头像两个人。`_build_rubric_text()`（`base.py:394`）把这些锚点拼进提示词，让 VLM 对着"评分标准"打分而非凭感觉。

### 6.4 批量 vs 逐条

`_BATCH_THRESHOLD = 2`（`base.py:1076`）。≤2 个 checkpoint 时尝试一次合并调用（`_evaluate_checkpoints_batch_inner`，`base.py:1113`），>2 个直接逐条评（`base.py:1096-1100`）——因为合并太多 checkpoint 会让 VLM 注意力涣散、互相干扰。

### 6.5 子指标聚合：置信度加权

`_aggregate_checkpoint_score()`（`base.py:1304`）把一个子指标（如 `temporal_coherence`）下所有**激活的** checkpoint 的归一化分，按 `weight` 加权平均，并**在激活集合上重新归一化**（被门控跳过的 checkpoint 的权重不进分母）。这样 DB_001 跳过 `object_permanence`、`temporal_logic` 后，剩余 checkpoint 的权重会自动放大补足。

---

## 7. 视频 agent 的"算法证据层"（DB_001 最吃重的一维）

`agents/video_agent.py` 是 5 个 agent 里唯一带**纯算法证据**的——它不只让 VLM 看图，还用 OpenCV 算出客观数字去**反驳/校正** VLM。对动作类的 DB_001 尤其关键，因为快速切换最容易出现"换人/穿帮/瞬移"。

视频 agent 评四个子指标：`_eval_user_demand`、`_eval_temporal_coherence`、`_eval_lighting_consistency`、`_eval_transition_quality`（`video_agent.py:81/346/495/559`）。

### 7.1 视觉证据计算

`_compute_visual_evidence()`（`video_agent.py:157`）在抽样缩略图上算：相邻帧 **pairwise SSIM** 及其均值 `avg_ssim`、**直方图卡方距离**、**像素差**、**Haar 人脸计数**（`video_agent.py:164-253`）。这些数字被翻译成自然语言塞进 VLM 提示词，例如把 SSIM 解释为档位（>0.85 几乎不变 / >0.6 中等变化 / 更低 = 剧变，`video_agent.py:290-303`），并明确提示"同场景相邻帧 SSIM 过低强烈暗示穿帮"（`video_agent.py:329`）。

### 7.2 算法–VLM 分歧再评（DB_001 的真实触发点）

`_eval_temporal_coherence()`（`video_agent.py:346`）里有一段关键逻辑：当**算法说差、VLM 却说好**时强制复评（`video_agent.py:407-440`）：

```python
if avg_ssim < 0.3 and r.raw_value >= 3:
    # VLM 给了 3+，但 avg_ssim<0.3 说明相邻帧极度不同
    # → 重新构造提示，告知"SSIM<0.3 时 3+ 分不成立"，逼 VLM 重评
```

> **DB_001 走到这里的真实情形**：屋顶追逐多角度快速切换，相邻缩略图本身差异就大，OpenCV 很可能算出 `avg_ssim` 偏低。引擎需要**区分**两种低 SSIM——(a) 正常的快速运镜/转场，(b) 真正的换人穿帮。于是：若 VLM 在 `char_face_consistency` 上给了 4 分但 `avg_ssim<0.3`，引擎不会盲信，而是带着"SSIM<0.3，3+ 分需要证据"的提示让 VLM 复看人脸区域（`video_agent.py:421-440`）。这正是该样本把"运镜快"和"生成崩"解耦开的机制。

### 7.3 转场质量两层检测

`_eval_transition_quality()`（`video_agent.py:559`）：

- **Layer 1 算法**：对每个镜头边界算合成分 `composite = 0.5*ssim + 0.25*hist_score + 0.25*flow_score`（`video_agent.py:600`），其中光流分 `flow_score = max(0, 1 - flow/30)`（`video_agent.py:598`）。`composite < 0.6` 的边界被**标记为可疑**（`video_agent.py:604`）。
- **Layer 2 VLM**：只对被标记的边界让 VLM 分类——是"同场景内的硬切穿帮"还是"合理的场景切换"。最终惩罚 `penalty = verified_bad * 0.08`（`video_agent.py:680`）：每个被 VLM 确认的坏转场扣 0.08。

DB_001 的出题卡明确要 `whip_pan`（甩镜）和 `tracking`，这类运镜在 Layer 1 会因高光流被标记，但 Layer 2 应判为"合理"，从而**不扣分**——两层设计就是为了不冤枉高动态运镜。

---

## 8. Phase 2–3：从子指标到一份诊断报告

### 8.1 跨模态对齐（barrier 之后）

`crossmodal_eval` 等四个单模态 agent 全部完成后才跑（`graph.py:131-134`），评 `text_video_consistency`（剧本说的镜头有没有拍出来）和 `video_audio_consistency`（画面动作和音效/对白时间戳对不对）。对 DB_001 就是核对："别跑！"这句对白有没有对上口型、`footsteps/impact` 音效有没有踩在跳跃/落地的画面上。

### 8.2 维度分：置信度加权平均

`DiagnosisSynthesizer.synthesize()`（`diagnosis.py`）先 `_compute_dimension_scores()`（`diagnosis.py:114`）把每个 agent 的多条 `EvalResult` 按 **(score, confidence)** 做置信度加权平均：

```python
scores[dim] = Σ(score·conf) / Σ(conf)          # diagnosis.py:144-146
```

**关键设计**：某个维度若一条结果都没有（如静音视频的 audio），该维度被**整体省略**、不拉低总分（`diagnosis.py:139-140` 的 `continue`）。

### 8.3 总分：用户画像加权

`_compute_overall_score()`（`diagnosis.py:164`）用 profile 的 `priority_weights` 给各维度加权。维度→权重字段的映射在 `_DIM_TO_WEIGHT_FIELD`（`diagnosis.py:156-162`）：

```
script→text_story_arc   video→visual_camera   audio→audio_emotion
crossmodal→cross_modal_sync   stability→visual_camera（与 video 共享）
```

只有**实际有分**的维度参与，被跳过维度的权重**从分母里剔除**再归一化（`diagnosis.py:175-189`）：

```python
total_weight = sum(active_weights.values())
return sum(dim_scores[dim] * w for dim, w in active_weights.items()) / total_weight
```

> **DB_001 + Profile 1 的总分如何成形**：Profile 1 的 `text_story_arc=0.55` 让**剧本维度在总分里占主导**。也就是说，同一段屋顶追逐成片，如果剧本（三幕结构「发现目标→追逐→制服」、因果逻辑）拍得完整，即便视觉一致性因快切略有瑕疵，对 Story-First 用户的总分影响也有限；反过来若换成 Profile 2「Visual-Heavy」（`visual_camera=0.50`），同一份子指标会算出**完全不同的总分**——视觉瑕疵会被放大。这就是"个性化评测"的实质：子指标客观不变，权重随用户改变。

### 8.4 grade 与瓶颈

`_assign_grade()`（`diagnosis.py:192`）按 `config.grade_boundaries` 降序匹配：A≥0.85 / B≥0.70 / C≥0.55 / D≥0.40 / 否则 F。

`_identify_bottlenecks()`（`diagnosis.py:207`）挑出所有 `score < bottleneck_threshold` 的子指标，按分**升序**（最差在前）排，连同其 `suggestions` 一起进报告（`diagnosis.py:211-222`）。最后 `_generate_narrative()` 让 LLM 把"总分 + grade + 瓶颈 + 建议"写成一段像剧组复盘的叙事诊断。

整份 `DiagnosisReport` 由 `main.py` 追加写入 JSONL（`ReportWriter.append`，`main.py:147-157`），同时把每次运行的工具调用轨迹（含耗时）写入 `tool_traces.jsonl`（`_append_tool_trace`，`main.py:31-60`）。

---

## 9. DB_001 全过程一句话回放

> 出题卡 `DB_001`（屋顶追逐、67s、要 tracking/whip_pan、强脉冲 BGM、脚步/撞击音效、中文对白）→ **Phase 0** 编排器把成片切镜头、抽音轨、跑 ASR、算逐转场 SSIM/光流 → **Phase 1** 四个专科 agent 并行：视频 agent 先用 ContentProfile 判定"有角色、无手持物、无场景切换"，跳过 `object_permanence/temporal_logic`，再用 OpenCV 算出相邻帧低 SSIM，并通过"算法–VLM 分歧复评"把"快切"与"换人穿帮"区分开；转场质量两层检测让合理的甩镜不被扣分 → **Phase 2** 跨模态核对"别跑！"口型与脚步音效对齐 → **Phase 3** 诊断器按置信度加权得各维度分，再按 **Profile 1 的 text_story_arc=0.55** 加权出总分、定 grade、列瓶颈、生成叙事报告 → 落 JSONL。

---

## 10. 创新点

1. **评测即诊断，而非打分**：输出瓶颈 + 可执行建议 + 叙事报告，对齐"导演复盘"而非"跑分榜"。
2. **多智能体 DAG（LangGraph）**：5 专科 agent 并行 + barrier + 综合，工程上把"分钟级多模态成片"的评测解耦成可并行、可追溯（`tool_traces.jsonl`）的流水线。
3. **checkpoint + ContentProfile 门控**：用 `applicable_when` 让评测"看人下菜"——没角色就不评人脸一致，不把 N/A 项算进分母。
4. **算法证据层校正 VLM**：视频 agent 用 OpenCV 的 SSIM/直方图/光流/人脸计数去**反驳** VLM 的乐观判断，并专门设计"低 SSIM 究竟是快切还是穿帮"的分歧复评，避免高动态运镜被冤枉。
5. **个性化加权**：同一份客观子指标，经 5 种用户画像的 `priority_weights` 得到不同总分与 grade——把"评测标准本身因人而异"显式建模。
6. **防御式打分**：缺陷优先提示、id 重试、factual_override、reasoning↔score 一致性检查，系统性压制 VLM 评测的常见失真。
