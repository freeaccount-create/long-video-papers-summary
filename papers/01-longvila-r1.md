# 01 · Scaling RL to Long Videos (LongVILA-R1 / Long-RL)

> NeurIPS 2025 · NVIDIA · [arXiv 2507.07966](https://arxiv.org/abs/2507.07966) · [项目页](https://research.nvidia.com/labs/eai/publication/longrl/)
> 官方代码：[github.com/NVlabs/Long-RL](https://github.com/NVlabs/Long-RL) · 模型 `Efficient-Large-Model/LongVILA-R1-7B` · 数据 `LongVideo-Reason/longvideo-reason`

本文是基于**官方源码逐行分析**写成：克隆自 `NVlabs/Long-RL`（fork 自 EasyR1/verl），并取 HuggingFace 上 `LongVideo-Reason` 的真实 `test.jsonl` 样本，追踪一条数据从输入视频到梯度更新的完整流程。

---

## 1. 源码结构

- `verl/`：RL 引擎（fork 自 verl/EasyR1）。`trainer/`（主循环）、`workers/{actor,rollout,reward}`、`utils/sequence_parallel/`（Ulysses/Ring 序列并行）、`utils/vila_remote_code/`（VILA 模型代码）。
- `longvideo-reason/`：数据生成流水线 `step1`～`step6` + `eval.py`。
- `examples/`：训练脚本、`reward_function/`、`format_prompt/*.jinja`。

> 注：CoT-SFT 阶段的训练 loop 不在本 repo（在 VILA 仓库），本 repo 负责**数据生成 + RL 阶段**。

---

## 2. 数据来源与真实格式

数据集 **LongVideo-Reason**（104K 长视频多步推理 QA）由一条全自动流水线生成（`longvideo-reason/`）：

1. `step1` 把长视频切成 10s 短片；
2. `step2` 用 **NVILA-8B-Video** 给每片打 caption；
3. `step3` 合并 caption；
4. `step4` 用 **DeepSeek-R1** 按 caption 生成"多步推理多选题"（prompt 见 `step4_gen_reasoning_data.py:42-64`）；
5. `step5` 正则解析（`step5_parse_reasoning_data.py:7-43`）；
6. `step6` 用 **GPT-4o** 把推理改写成自然 CoT，并抹掉 caption/时间戳痕迹（`step6_reformat_reasoning_data.py:53-92`）。

**一条真实样本**（HF `test.jsonl` 第 0 行，逐字摘录）：
```json
{"problem_id": 0,
 "problem": "What is the primary intention behind the video's sequence of showcasing an African elephant (0:00:00-0:00:30) ... a cheetah (0:08:00-0:08:50)?\nA. ...\nB. To contrast human athleticism with animal locomotion, culminating in the cheetah's evolutionary adaptation for speed.\nC. ...\nD. ...",
 "data_type": "video", "problem_type": "goal",
 "reasoning": "The sequence begins by showcasing the African elephant ... reinforcing the video's primary intention: to contrast human athleticism with animal locomotion ...",
 "videos": "longvila_videos/I_Q3ajyOrcI.mp4",
 "answer": "<answer>B</answer>"}
```
字段固定为：`problem_id / problem / data_type / problem_type / reasoning（CoT，仅 SFT 用）/ videos（相对路径）/ answer（<answer>X</answer>）`。

---

## 3. 完整训练流程

**两阶段**（README:22）：
- **① CoT-SFT（冷启动）**：用 `reasoning` 字段做监督，让模型学会写 `<think>…</think><answer>…</answer>` 的推理链。
- **② RL（本 repo 主体）**：默认算法 **GRPO**。

**RL 主循环**在 `verl/trainer/ray_trainer.py`：rollout → 算 reward → `compute_advantage(adv_estimator=GRPO)`（`ray_trainer.py:134`）→ `compute_grpo_outcome_advantage`（`core_algos.py:151`）对同一 prompt 的 n 条采样做**组内均值/方差归一化**（`core_algos.py:175,189` `scores[i]=(scores[i]-id2mean)/(id2std+eps)`）→ 带 KL 的 PPO-clip 更新（`config.yaml:22-26`，`use_kl_loss=true, kl_penalty=low_var_kl`）。

**Reward 函数**（视频用 `examples/reward_function/r1v.py`）：
```python
# 格式分：必须 <think>…</think><answer>…</answer>
format_match = re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", response, re.DOTALL)   # r1v.py:21
# 准确分：抽 <answer> 内首字母，mathruler.grade_answer 与 GT 比对                                # r1v.py:27-37
overall = 0.9*accuracy + 0.1*format                                                              # r1v.py:47
```
开放式 QA 走 LLM-judge（`vllm_rollout_spmd.py:286-307`，GPT 输出 `"yes"→1.0`）。

### MR-SP（核心创新）= 序列并行 + 视频 embedding 缓存
- **序列并行（Ulysses）**：`verl/utils/sequence_parallel/`。actor 前向把 (video+text) 长序列按 SP rank 切片：`dp_actor.py:413` 调 `prepare_inputs_for_sp_mm`（定义于 `dp_actor.py:116`），longvila 配置 `ulysses_size=4`。
- **embedding 缓存**：离线 `verl/utils/cache_video_embeds_vila.py:60` 用 vision encoder 算好视频 embed 并 `torch.save` 成 `.pt`；训练时 `dataset.py:328` 命中缓存则只放 1 帧占位、把缓存 embed 塞进 `multi_modal_data`，rollout 端 `use_cached_embeds` 跳过重复视觉编码（`vllm_rollout_spmd.py:227`）。号称 **2.1× 加速**。

---

## 4. 一条真实数据的全过程（追踪 `I_Q3ajyOrcI.mp4` 样本）

1. **输入**：dataset 读到该 jsonl 行，`_get_messages_vila`（`dataset.py:96-120`）拼出 `<video>` + 要求 `<think>/<answer>` 的 prompt 模板；video 路径 = `video_dir/longvila_videos/I_Q3ajyOrcI.mp4`。
2. **抽帧/embedding**：`__getitem__`（`dataset.py:319-339`）—— 有缓存则 `torch.load(I_Q3ajyOrcI.pt)` 得到 `[num_video_frames, hidden]` 视频 embed（脚本设 `num_video_frames=256, tokens_per_frame=257`），并把 `num_video_frames` 临时设 1；结果存入 `example["multi_modal_data"]["video"]`。
3. **SFT 监督的 CoT**（上一阶段）：`reasoning` 字段被包成 `<think>…</think><answer>B</answer>` 做交叉熵；RL 阶段不再用 `reasoning`，只用 `answer="B"` 当 GT。
4. **RL rollout**：`vllm_rollout_spmd.py:217-231`，VILA 路径先把视频 embed 与文本拼成 `prompt_embeds`（≈256×257≈65792 视觉 token + 文本），vLLM 生成 **n=5** 条回答，每条形如 `<think>…cheetah…</think><answer>B</answer>`。
5. **reward 打分**：`r1v.py:compute_score` 对每条算 `format(0/1)` 与 `accuracy`（抽 `<answer>` 字母与 "B" 比），`overall=0.9*acc+0.1*fmt` → 5 个标量 reward。
6. **更新**：`compute_grpo_outcome_advantage` 对 5 个 reward 组内归一化成 advantage；actor 前向时长序列经 `prepare_inputs_for_sp_mm` 按 4 个 SP rank 切片（`dp_actor.py:413`），算 `log_probs` 后跨 SP 拼回，做 KL-正则 PPO-clip 梯度更新。

**张量流转**：`视频 mp4 → [256, hidden] cached embed → cat(文本embed, video_embed, response) → SP 切片 [4×local_len] → logits → log_probs → reward 标量×5 → GRPO advantage → 梯度`。

---

## 5. 模型 / 组件

- **Policy / base**：LongVILA-7B（亦支持 NVILA-2B、Qwen2.5-VL-3B/7B/32B、Qwen2.5-Omni、Qwen3）。
- **Vision encoder**：SigLIP（`vila_remote_code/siglip_encoder.py`）+ `mm_projector`。
- **数据生成 LLM**：DeepSeek-R1（出题）、GPT-4o（CoT 改写）；caption 用 NVILA-8B-Video。
- **RL 算法**：GRPO（默认）；另支持 DAPO、REINFORCE++、RLOO、REMAX。
- **基础设施**：verl/EasyR1 + Ray + FSDP + vLLM（`enable_prompt_embeds=True`）+ Ulysses/Ring 序列并行。

---

## 6. 创新点

1. **LongVideo-Reason 数据集与自动标注流水线**：104K 长视频多步推理 QA，靠"切片→caption→R1 出题→GPT 改写 CoT"全自动生成。
2. **MR-SP（Multi-modal RL Sequence Parallelism）**：多模态序列并行（Ulysses）+ 离线视频 embedding 缓存，使单 8×A100 节点可训 3600 帧 / 256K token 的小时级视频 RL，约 2.1× 加速。
3. **两阶段 CoT-SFT→RL + 双模式奖励**：冷启动 CoT-SFT 后用 GRPO 强化；奖励同时支持多选题规则判分与开放式 GPT-judge，并覆盖 video/text/audio 多模态。
