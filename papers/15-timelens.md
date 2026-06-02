# 15 · TimeLens（视频时序定位：数据质量 + 交错时间戳 + thinking-free RLVR）

> CVPR 2026 · [arXiv 2512.14698](https://arxiv.org/abs/2512.14698) · [项目页](https://timelens-arc-lab.github.io/) · 官方代码：[github.com/TencentARC/TimeLens](https://github.com/TencentARC/TimeLens)

基于**官方源码逐行分析**写成（克隆成功，约 724K；仅代码，权重/视频/标注 JSON 从 HF 下载）。main 分支为 TimeLens-8B / Qwen3-VL；TimeLens-7B / Qwen2.5-VL 在 `train` 分支（`README.md:403`）。

---

## 1. 源码可得性 / 结构

- `timelens/dataset/timelens_data.py`：数据集加载（100K 训练集、Bench 三测试集），字段解析。
- `timelens/utils.py`、`training/utils/parser.py`：`extract_time` / `iou` / `extract_answer`。
- `training/data/{grounding,hybrid,collator,inference_collator,preprocess}.py`：SFT/GRPO 样本构造与打包。
- `training/train/{train_sft_timelens,train_grpo_timelens,reward_funcs}.py`：训练入口与奖励函数。
- `training/trainer/grpo_trainer_qwenvl.py`（1692 行，改自 TRL v0.19.1）：GRPO 核心。
- `training/filter/infer_qwen3_vl_tvg_dataloader_filter_data.py`：难度过滤/重打分推理。
- `evaluation/{eval_dataloader,compute_metrics,utils}.py`：TimeLens-Bench 评测。
- `train_scripts/*.sh`、`scripts/{eval_timelens_bench.sh,filter_data/*}`：启动脚本。

---

## 2. 数据来源与真实格式

**TimeLens-100K**（HF `TencentARC/TimeLens-100K`，`timelens-100k.jsonl`，~20K 视频 / ~100K 标注，源自 cosmo_cap/didemo/hirest/internvid_vtime/queryd，Gemini-2.5-Pro 自动重标）。真实样本（jsonl 首行）：
```json
{"source":"cosmo_cap","video_path":"cosmo_cap/BVs52yd-RUQ.mp4","duration":117.42,
 "events":[{"query":"When does the speaker introduce himself and the company?","span":[[0.0,5.0]]},
           {"query":"Show me a close-up shot of the macadamia nut oil bottle.","span":[[8.0,12.0]]}, ...]}
```
即"每视频含 duration + 多条 event，每 event 为 (自然语言 query, span=[[start,end]])"，由 `TimeLens100KDataset.load_annos` 展平为单 query 标注（`timelens_data.py:75-86`）。

**TimeLens-Bench**（HF `TencentARC/TimeLens-Bench`，charades/activitynet/qvhighlights 三文件）。真实样本（charades vid=3MSZA）：
```json
"3MSZA": {"duration":31.0, "spans":[[25,30],[1,24],[0,1],[23,24]],
          "queries":["A woman is repeatedly flipping the switch on the wall.", ...]}
```
即 `dict[vid] → {duration, spans, queries}`，`spans` 与 `queries` 一一对应（`timelens_data.py:30-38`）。

**标注方式**：100K 经 Gemini-2.5-Pro 自动重标 pipeline 生成 query 与秒级 span；Bench 经人工 "Diagnose-then-Refine" 对 Charades-STA/ActivityNet/QVHighlights 审计纠错。query 经 `parse_query` 规范化（`timelens_data.py:8-10`）。

---

## 3. 完整方法 / 训练流程

**(a) Interleaved textual timestamp encoding**：把每帧采样时间戳作为文本 token 前缀插在该帧视觉 token 之前。
- prompt 显式说明：`"You are given a video with multiple frames. The numbers before each video frame indicate its sampling timestamp (in seconds)."`（`evaluation/utils.py:14-17`，`GROUNDER_PROMPT_TEXT_TIMESTAMP`）。
- 实际交错由 `qwen_vl_utils.process_vision_info(..., return_video_metadata=True)` + 处理器（`second_per_grid_ts`/video_metadata）完成（`evaluation/utils.py:74-103`；collator 透传 `second_per_grid_ts` `collator.py:52-53`）。7B(Qwen2.5-VL) downsample_rate=28、8B(Qwen3-VL)=32（`utils.py:41-50`）。7B 用显式交错文本前缀 prompt；8B/Qwen3-VL 由模型原生时间戳机制处理（不加该前缀句，`utils.py:26-30`）。

**(b) Thinking-free RLVR + IoU reward**：
- **thinking-free**：GRPO 仅用 `--reward_funcs tiou`（`run_grpo_qwen3_8b.sh:113`），**不启用** format（`<think>...</think>`）奖励；prompt 直接要求输出 `"The event happens in <start> - <end> seconds"`（`grounding.py:13-16`），无思维链。`beta=0.0`（`params.py:113`）→ 无 KL 项、不加载 reference model（`grpo_trainer_qwenvl.py:617,648,1536-1537`）。
- **IoU reward**：`tiou_reward`（`reward_funcs.py:19-48`）从 completion 提取答案 → `extract_time` 解析时间段 → 与 GT span 算时序 IoU 作 reward。IoU 定义 `max(min1-max0,0)/(max1-min0)`（`parser.py:21-30`）；非法/解析失败/start≥end → 0（`reward_funcs.py:36-41`）。
- **GRPO 更新**：每 prompt 采样 `num_generations=8`（`params.py:124`），组内归一化优势 `A = r - mean_group`（脚本显式传 `--scale_rewards False`，故不除 std；TRL 默认值本为 True，`grpo_trainer:1380-1389`）；PPO 式 clip 目标、`loss_type=bnpo`。
- **per-token 目标与 bnpo 归一（精确公式，`grpo_trainer_qwenvl.py:1523-1546`）**：
  - 重要性比 `coef_1 = exp(logπ_θ − logπ_old)`（:1526），裁剪 `coef_2 = clip(coef_1, 1−ε_low, 1+ε_high)`（:1527，`epsilon` 见 `:588-589`），`per_token_loss = −min(coef_1·A, coef_2·A)`（:1533-1535）即标准 PPO 双侧裁剪；因 `beta=0` 跳过 `+β·KL`（:1536-1537）。
  - **三种 loss_type 的差别只在分母**（:1539-1546），TimeLens 取 `bnpo`：
    - `grpo`：`((per_token_loss·mask).sum(-1) / mask.sum(-1)).mean()` —— **先每条序列内按自身 token 数平均，再对序列求平均**（每条等权，长序列每 token 权重小）。
    - `bnpo`（本文用）：`(per_token_loss·mask).sum() / mask.sum()` —— **全 batch 所有 completion token 拉平后整体除以总有效 token 数**，即每个 token 等权、长序列自然占更大比重，无"先序列内平均"这一步。
    - `dr_grpo`：`.sum() / (batch_size · max_completion_length)` —— 固定分母（去长度偏置）。
  - 选 bnpo 的效果：避免 grpo 把每条序列强行等权导致短答案 token 被过度放大，对"输出就一句 `start - end seconds`"的定长格式更稳。
- lr=1e-6、constant、`max_steps=100`、freeze vision tower（`run_grpo:97-123`）。

**(c) 数据重标/难度采样**（`README.md:327-380` 三阶段）：① 在 30K 采样上 SFT；② 用 SFT 模型对全量 100K 离线推理、对每条算 IoU 作难度（`infer_..._filter_data.py:150-164`，写回 `pred/answer/iou`）；③ GRPO 从 SFT ckpt 起训，按 IoU 做 Gaussian 难度采样（`fixed_gaussian_sampling=True, mean=0.05, std=0.2`，`run_grpo:84-87`；实现 `grounding.py:224-247`，按时长分桶 + 按 IoU 高斯加权采样做逆密度均衡）。

---

## 4. 一条真实数据的全过程

以 didemo 真实样本 query `"At what point does the child get off the toy?"`、GT span `[39.0,43.0]`、duration 52.6s 为例：

1. **抽帧+时间戳交错**：`GroundingDataset._getitem_grpo` 构造 message：video(fps=2, total_tokens=14336, max_frames=448) + 文本 prompt（`grounding.py:310-323, 54-67`）。`process_vision_info` 按 2fps 抽帧并把各帧采样秒数作为文本前缀与帧 token 交错送处理器（`grounding.py:332-352`）。
2. **模型输出时间段**：策略模型对该 prompt 采样 8 条 completion（`grpo_trainer:1213-1217`），如 `"The event happens in 38.0 - 44.0 seconds."`。
3. **IoU reward**：`extract_time` 解析得 `(38.0,44.0)`，与 GT `[39.0,43.0]` 算 IoU = `(43-39)/(44-38)=4/6≈0.667`（`reward_funcs.py:45`, `parser.py:21-30`）；其余 7 条各算 IoU。
4. **GRPO 更新**：`rewards.view(-1,8)` 求均值，`A_i = IoU_i - mean`（`grpo_trainer:1380-1387`）；优势>0 的强化、<0 的抑制，经 clip 目标回传更新 LLM+merger（vision tower 冻结），无 KL/无 reference（beta=0）。
5. **评测**：训练后在 TimeLens-Bench 上 `eval_dataloader.py` 贪心解码（max_new_tokens=512），`compute_metrics.py` 算 R1@{0.3,0.5,0.7} 与 mIoU（:82-93）。

---

## 5. 模型 / 组件

- **Base model**：TimeLens-8B ← **Qwen3-VL-8B-Instruct**（`model_loader.py:6-16`）；TimeLens-7B ← **Qwen2.5-VL-7B-Instruct**（train 分支）。SFT 中间产物 `JungleGym/TimeLens-Qwen3-VL-8B-SFT`。
- **RL 算法**：**GRPO**（改自 TRL v0.19.1，`grpo_trainer_qwenvl.py:1`）；G=8、bnpo loss、beta=0（thinking-free, reference-free）。
- **其他**：DeepSpeed ZeRO-1(GRPO)/ZeRO-3(SFT)、flash-attention-2、Liger(仅SFT)、nncore、`qwen_vl_utils`。重标注器 **Gemini-2.5-Pro**。
- **指标**：时序 IoU、Recall@1 IoU∈{0.3,0.5,0.7}、mIoU。

---

## 6. 创新点

1. **数据质量重构**：揭示主流 VTG 基准（Charades-STA/ActivityNet/QVHighlights）的高标注错误率，经人工 Diagnose-then-Refine 得高质量 **TimeLens-Bench**；用 Gemini-2.5-Pro 自动重标构建 **TimeLens-100K**（~100K 高质量 VTG 标注）。
2. **Interleaved textual timestamp encoding**：系统对比位置编码/视觉叠加/文本编码后，发现"原始时间戳作文本前缀交错插在各帧视觉 token 前"最有效且最简洁（`evaluation/utils.py:14-17`）。
3. **Thinking-free RLVR + 训练配方**：证明对感知主导的 VTG，无思维链、以时序 IoU 为唯一可验证奖励的 GRPO（beta=0、无 reference、不启用 format reward）性能与效率最优，前置 SFT 无显著增益；并给出奖励平台期早停、基于离线 IoU 难度的高斯采样两条配方。

> **核对说明**：file:line 来自实际克隆源码，真实样本取自 HF 原始 jsonl/json。leaderboard（`static/js/index.js:158-277`）可核实：开源 **TimeLens-8B**（Qwen3-VL-8B 基座）平均超过 **GPT-5**（mIoU Charades 55.2 vs 40.5 / ActivityNet 53.2 vs 42.9 / QVHighlights 65.5 vs 56.8）与 Gemini-2.5-Flash；相对其标注器 **Gemini-2.5-Pro** 总体仍偏低，但在 **Charades 上 mIoU 55.2 已反超 Pro 的 52.8**（ActivityNet/QVHighlights 仍低于 Pro）。论文 **Table 6（消融）** 中存在 **TimeLens-3B** 条目，结论是 3B 即可显著超过更大的 **Qwen2.5-VL-7B 基线**（注意是超过基线而非 TimeLens-7B）；公开仓库代码/leaderboard 未含 3B 权重与条目，故 3B 数据以论文表格为准。
