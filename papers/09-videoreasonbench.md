# 09 · VideoReasonBench（benchmark）

> ICLR 2026 · [arXiv 2505.23359](https://arxiv.org/abs/2505.23359) · 官方代码：[github.com/llyx97/video_reason_bench](https://github.com/llyx97/video_reason_bench) · 视频 `lyx97/reasoning_videos`

基于**官方源码逐行分析**写成（克隆成功，HEAD `2524387`）。视频二进制需另从 HF 下 `videos.zip`，repo 含 QA json + 评测代码。

---

## 1. 源码结构

- `eval_api.py`：评测主入口（抽帧/推理/判分驱动）。
- `utils/eval_utils.py`：推理 API、LLM judge、各 demo 的状态机仿真判分。
- `utils/show_results.py`：汇总打分。
- `questions/{hrd,grid,cup,file_sys,card,chip}.json`：6 类视频 QA（各 240 条）。
- `vlmevalkit/`：VLMEvalKit 集成。

> 数字华容道(sliding number puzzle)对应文件名是 `hrd.json`（Hua-Rong-Dao），README 展示名为 "Number"。代码 6 个 demo 键 `['hrd','file_sys','cup','grid','card','chip']`（`eval_api.py:26`），其中 `grid`=Circle、`hrd`=Number。

---

## 2. 数据 / 真实格式

**6 类视频 → demo 键映射**（`eval_api.py:26`，判分分发 `:159-179`）：Number=`hrd`、Circle=`grid`、Cup=`cup`、File=`file_sys`、Card=`card`、Chip=`chip`。

**3 级推理 → 报告口径 6 个 `dim`**（分组见 `show_results.py:18-46`）：
- **L1 Recall（回忆可见操作）**：`order_operation`、`counting_operation`
- **L2 Infer（推断不可见 latent state）**：两个**独立**报告维度——
  - `infer_state`：由样本原始 `dim ∈ {order_state, counting_state}` **合并**而来。`show_results.py:44` 一行 `dim = "infer_state" if r['dim'] in ["order_state","counting_state"] else r['dim']` 把这两类原始维度统一改写成 `infer_state` 计分（`order_state` 用于 hrd/cup/grid/card，`counting_state` 用于 file_sys/chip）。
  - `comparison_state`：**第 6 个独立维度，并非 order/counting_state 的派生或乘积**。`show_results.py:44` 的三元式 `else r['dim']` 分支让 `dim=='comparison_state'` 的样本**原样透传**进 `comparison_state` 桶——它来自专门标注为"比较 latent 状态量"的题，与 infer_state 互不重叠。
  - （`summary` 初始化 `:18-20` 显式列出全部 6 个报告桶：`order_operation / counting_operation / infer_state / comparison_state / prediction_state / prediction_operation`，证实 comparison_state 是平级的独立列而非合并产物。）
- **L3 Predict（超出视频的预测）**：`prediction_state`、`prediction_operation`

**latent state 编码**：每条样本带 `states` 数组（初始状态及每步操作后的完整棋盘快照，length=`num_operation+1`），`moves` 记操作序列，`visible_time ∈ {start,end}` 指明 latent state（初/末态）哪端在视频中可见、另一端被蓝色遮罩。判分取"可见端"为已知起点：`states[0 if visible_time=='end' else -1]`（`eval_api.py:161`）。`hrd` 棋盘为 3×3/4×4 矩阵，`0`=空格；答案行 `a/b/c` 映射到矩阵 `a→最底行`（`eval_utils.py:489-500`）。

**真实样本**（`questions/hrd.json`，key=`state3_op5_order_operation_start`，逐字摘录）：
```
"video": "videos/hrd/state3_op5_start.mp4",
"question": "...What are the 2nd to 4th blue squares being moved?...Provide a summary of the final answer after 'Final Answer:'",
"answer": "2nd: (a,2) right\n3rd: (b,2) up\n4th: (c,2) up",
"dim": "order_operation", "visible_time": "start",
"num_state": 3, "num_operation": 5,
"moves": ["down","right","up","up","left"],
"states": [ [[6,5,8],[4,3,0],[2,1,7]], ... , [[6,8,0],[4,5,7],[2,3,1]] ]
```

---

## 3. 完整评测流程（file:line）

1. **加载/打散**：`load_data` 合并 6 json 并加 `demo` 前缀（`eval_api.py:24-32`）；`random.seed(42)` 后 shuffle、分 chunk。
2. **抽帧**：`load_video`(:71-95) `target_fps=1.0` 采样，超 `max_num_frames`(128) 则均匀重采；`resize_image` 缩到长边 448。
3. **Prompt 构造**：OpenAI 路径把每帧 base64 拼为 `image_url`(detail=high)+末尾文本(`:97-126`)；**Gemini 路径上传整段 mp4**，`contents=[video_file, question]`（`eval_utils.py:84`）。**thinking budget**：除 `gemini-2.0-flash` 外用 `types.ThinkingConfig(thinking_budget=...)`（默认 8192，脚本设 8192 / max_new_token 65536）。
4. **模型作答**：`inference`(:129-137)，`temperature=0`。
5. **答案抽取**：`extract_final_answer` 去 think 段后正则 `Final Answer\s*(.*)`（`eval_utils.py:177-188`），抽不到判 False。
6. **判分（两条路径）**：
   - **有 GT 的 5 个 dim** → `evaluate`(:139-148) → **LLM judge**：prompt 要求"response 须含 GT 全部信息"，输出 `Correct/Incorrect`（精确比较 `=="Correct"`，`eval_utils.py:25-46`）；judge 默认 `gpt-4o-2024-11-20`。
   - **`prediction_operation`（answer 为 None）** → `evaluate_operation`(:150-179) → **状态机精确仿真**：judge 先把自然语言答案抽成动作列表（`extract_move_hrd`），再用 `get_next_board_hrd` 从可见端逐步推演，最后 `np.array_equal(末态, 目标态)`（`eval_op_hrd`，`eval_utils.py:584-594`）。即 L3 操作题是"可执行性校验"而非匹配文本。
7. **汇总**：`show_results.py:16-53` 按 overall / num_op / state 数 / 各 dim 统计准确率与 token 消耗。

---

## 4. 一条真实数据全过程（数字华容道，`state3_op5_*_start`）

`visible_time=start` ⇒ 视频展示**初态**（可见），末态为 latent。`moves=[down,right,up,up,left]`，状态演化（矩阵，`0`=空）：
```
S0 [[6,5,8],[4,3,0],[2,1,7]]  (可见初态)
 down →[[6,5,8],[4,3,7],[2,1,0]]
 right→[[6,5,8],[4,3,7],[2,0,1]]
 up   →[[6,5,8],[4,0,7],[2,3,1]]
 up   →[[6,0,8],[4,5,7],[2,3,1]]
 left →[[6,8,0],[4,5,7],[2,3,1]] = S5 (latent 末态)
```

- **L1 Recall（`order_operation`）**：问"第 2~4 个被移动方块的次序/移动前坐标/方向？"。GT `2nd:(a,2) right; 3rd:(b,2) up; 4th:(c,2) up`。判分：LLM judge 比对 Final Answer 是否含全部 GT。只靠回忆可见操作。
- **L2 Infer（`order_state`）**：问"视频结束时棋盘排布？"。GT = latent 末态 S5（`(a,1):2 (a,2):3 ...`，行 a=矩阵最底行）。判分 LLM judge。需从可见初态 + 5 步操作**推演出不可见末态**。
- **L3 Predict**：*prediction_state*"从末态再执行 `right,right,down,up,left` 结果排布？"GT = S5 继续推 5 步，判分 judge。*prediction_operation*（`answer=None`）"从末态给出移动序列达到目标态"，判分**走仿真**：judge 抽方向序列 → 从 S5 用 `get_next_board_hrd` 推演 → `np.array_equal` 比目标态，允许任意可行解。

---

## 5. 模型 / 组件

- **被测模型**：Gemini 系列（默认 `gemini-2.5-flash-preview-04-17`，含 thinking budget；`gemini-2.0-flash` 关闭 thinking）、OpenAI（`gpt-4o-2024-11-20`）、Qwen2.5-VL-7B/72B。论文共评 18 个 MLLM，Gemini-2.5-Pro 最高 56.0%，GPT-4o 仅 6.9%。
- **判分辅助 LLM judge**：默认 `gpt-4o-2024-11-20`，可选 `Qwen2.5-72B-Instruct`；judge 兼任 L3 操作题的"答案→动作列表"抽取器。
- **状态机判分器（无 LLM）**：`get_next_board_hrd/grid/cup`、`get_next_files`、`eval_op_*`（`eval_utils.py:190-712`）。

---

## 6. 创新点

1. **Latent-state + 部分可见设计**：每段视频是"对一个潜在状态的细粒度操作序列"，状态只在首/末端短暂可见、其余被遮罩，迫使模型靠多步 CoT 推演不可见状态，而非检索知识——"vision-centric"。
2. **三级递进难度统一框架**：同一视频复用为 Recall/Infer/Predict 三层 6 维度，从"回忆可见操作"到"推断 latent"再到"预测视频之外"，解耦视觉感知与推理深度。
3. **可执行性仿真判分（非纯匹配）**：对 L3 操作题用确定性状态机仿真校验解的正确性（`eval_op_hrd`），允许多解；其余维度用 LLM judge 做语义级"信息完整性"判定。
