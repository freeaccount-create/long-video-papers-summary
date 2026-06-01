# 04 · VideoLucy

> NeurIPS 2025 · [arXiv 2510.12422](https://arxiv.org/abs/2510.12422) · [项目页](https://videolucy.github.io/)
> 官方代码：[github.com/Zplusdragon/VideoLucy](https://github.com/Zplusdragon/VideoLucy) · 数据集 `jlongzuo/EgoMem`

基于**官方源码（Demo 版）逐行分析**写成。

> 注：仓库发布的是**单视频开放式问答 Demo**，未发布 benchmark 批量评测代码，但方法主干（层级回溯）完整可读。

---

## 1. 源码结构

- `demo.py`：主推理流程（粗→细→超细 的迭代回溯主循环）。
- `VLMs/vlm_roles.py`：粗/细记忆抽取（Qwen2.5-VL）；`VLMs/utils.py`：抽帧、切短视频（decord + cv2）。
- `LLMs/llm_roles.py`：agent 角色（摘要/判定/时间窗检索/作答，火山引擎 Ark API）；`LLMs/utils.py`：全部 prompt 模板。
- `utils.py`：`parse_answer` / `filter_coarse_memory_by_time_periods` / `contains_ordinal_number`。
- `demo_cache/coarse_memory/*.json`：预生成粗记忆缓存。

---

## 2. 数据 / 输入格式

- **评测 benchmark**：自建 **EgoMem**（源自 EgoLife，42 个第一人称超长视频，均长 6.33h，504 题，分 Detail/Event）、以及 LVBench、Video-MME、MLVU、EgoLife。
- **EgoMem 输入格式**（单选题）：
```json
{"videoID":"A1_JAKE_DAY1","question_id":"0","type":"Detail",
 "question":"Who cleaned the whiteboard shortly after Jake and others ...",
 "options":["A. ...","B. Jake himself.","C. ...","D. ..."],"answer":"B"}
```
- **记忆条目结构**（`vlm_roles.py:102-106`）：`{"time_period": (start, end), "general_memory": caption}`。
- **代表性真实例**（缓存）：一条 790s 视频被切成 14 段（前 13 段 60s/段，末段 10s）。

---

## 3. 完整方法流程（粗→细，主控在 `demo.py:demo_infer` 62-251）

**Phase 1 — 顶层粗扫生成记忆**：模型 **Qwen2.5-VL-7B**（`vlm_roles.py:8-20`，flash-attn2）。按 `sampling_fps=1.0` 抽帧，每 60 帧切一段短视频（`VLMs/utils.py:32-109`），低分辨率 `coarse_memory_max_pixels=128*28*28`，批量生成 caption（`video_coarse_memory_extraction`，`vlm_roles.py:23-112`，带磁盘缓存）。LLM 生成全局摘要（`llm_roles.py:19-56`）。

**Phase 2 — 粗记忆直接作答 + 置信度判定**：`video_question_answer_with_coarse_memory`（`llm_roles.py:116`）要求 LLM 输出 `{"Confidence","Answer","Time Period","Reason"}`。`Confidence==True` 直接返回（`demo.py:95`）。否则若问题不含序数词（`contains_ordinal_number`，`utils.py:115`），先做问题类型判定收窄时间范围（`llm_roles.py:192`），返回最相关 2 个时间段，用 `filter_coarse_memory_by_time_periods`（`utils.py:60`）裁剪。`max_iterations=5`。

**Phase 3 — 收缩时间窗 + 更密采样 + 更细 caption 的迭代回溯**（`demo.py:143-234`，≤5 轮）：
1. **agent 选时间窗 + 下达指令**：`video_question_get_single_related_time_with_coarse_memory`（`llm_roles.py:173`）选出**单个**最相关时间段，并给出"让 caption 模型重点描述什么"的 `Instruction`，排除已探索段（`excluded_time_periods`）。
2. **细记忆抽取**（Qwen2.5-VL 再跑，高分辨率 `fine_memory_max_pixels=512*28*28`）：`split="entire"`（整窗一段）与 `split="divided"`（`fine_sampling_fps=2`、子段 ~10s）（`vlm_roles.py:115-198`）。
3. **超细层触发**：选中窗口长度恰为 `minimal_duration=10`s → `is_super_fine=True`，改用每 1s 一段做帧级精描，存入 `super_fine_memory_history`（`demo.py:206-216`）。
4. **带细记忆再作答**：`video_question_answer_with_coarse_and_fine_memory`（`llm_roles.py:135`）把 粗+细(entire/divided)+超细记忆**按时间排序嵌套拼接**（`LLMs/utils.py:126-158`，排序在 126 行）。`Confidence==True` 返回。

**Phase 4 — 强制作答**：5 轮仍不自信 → `must_answer`（`llm_roles.py:154`）强制给最优答案。

层级语义：**粗（60s/段, 1fps, 低分辨率）→ 细（窗口内 ~10s 子段, 2fps, 高分辨率）→ 超细（10s 窗口, 1s/段帧级）**，由置信度门控逐层加深。

---

## 4. 一条真实数据走完整层级回溯（"钥匙放哪了"类细节追溯）

问题：**"Where did the protagonist put the keys?"**（瞬时细节，无序数词），对应那条 790s 视频：

| 层级 | 时间窗 | 采样密度 | caption 粒度 | 触发条件 |
|---|---|---|---|---|
| 粗 | 60s/段，覆盖全片 | 1 fps | 整段概述、低分辨率(128·28²) | 初始 |
| 细 entire/divided | agent 选中单段(~60s)，内切 ~10s 子段 | 2 fps | 高分辨率(512·28²)、按 Instruction 聚焦 | 粗记忆不自信 |
| 超细 | 收缩到 10s（==minimal_duration） | 2 fps，每段 ≈1s | 帧级 timestamp 精描 | 选中窗口恰为 10s |

- **Phase 1（粗扫）**：14 段、60s/段、1fps，粗 caption 只说"在玄关活动"，不含钥匙位置。
- **Phase 2 判定**：`Confidence=False`；无序数词 → 问题类型判定为细节型，返回相关两段 `[120,180]`、`[180,240]`，裁剪粗记忆。
- **Phase 3 迭代①（细）**：agent 选 `[120,180]`，Instruction "重点描述手部动作与放置物品位置"。divided 切 6 个 ~10s 子段，定位到"约 [150,160] 在玄关放下物品"。仍 `Confidence=False`。
- **Phase 3 迭代②（超细）**：窗口收到 `[150,160]`（==10s）→ `is_super_fine=True`，2fps/每段 1s 帧级精描，描述"把钥匙挂到门口挂钩 / 放进玄关抽屉"。
- **作答**：嵌套拼接 粗+`[120,180]`细+`[150,160]`超细 喂 LLM；`Confidence=True` → 返回"挂在玄关门口挂钩上"，`Time Period=[(150,160)]`。

---

## 5. 模型 / 组件

- **MLLM（视觉 captioner）**：**Qwen2.5-VL-7B-Instruct**，本地，bf16 + flash_attention_2（`vlm_roles.py:11-16`）。
- **LLM（agent / 推理）**：README 标称 **DeepSeek-R1**；Demo 代码默认 `deepseek-v3-1-terminus`、`thinking="disabled"`（`demo.py:50-51`），可切 R1。经火山引擎 Ark SDK 调用（`llm_roles.py:3-13`）。
- **视频处理**：`decord`（VideoReader）抽帧、`cv2.VideoWriter` 写短视频、`you-get` 下载 Bilibili。

> 文档提示：README（DeepSeek-R1）与代码默认（V3 关闭思考）在 LLM 选择上不一致。

---

## 6. 创新点

1. **渐进粒度的层级记忆结构**：显式定义不同深度记忆的"细节级别 + 时间范围"（粗 60s/1fps → 细 ~10s/2fps → 超细 10s 帧级），用连续短视频段做 caption 而非单帧，解决逐帧建模"丢失时序上下文"的问题。
2. **agent 驱动的迭代回溯（置信度门控）**：LLM 反复判断信息是否足够，不足则自主选下一个最相关时间窗并下达定制 caption 指令，逐层加深直到自信作答；只对相关片段密集 caption，降低成本。
3. **EgoMem 基准**：面向超长第一人称视频（均长 6.33h），同时考察瞬时细节感知(Detail)与跨时序事件理解(Event, 6 子类)，504 题、人工标注、刻意规避捷径。
