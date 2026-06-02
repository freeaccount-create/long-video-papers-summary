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

> 注：`step6_reformat_reasoning_data.py` 生成时把 `problem_type` **硬编码为 `general`**（line 80-88）；上面样本中的 `"goal"` 来自 HF 发布版 `test.jsonl` 的另行标注，并非 step6 代码产物。

---

## 3. 完整训练流程

**两阶段**（README:22）：
- **① CoT-SFT（冷启动）**：用 `reasoning` 字段做监督，让模型学会写 `<think>…</think><answer>…</answer>` 的推理链。
- **② RL（本 repo 主体）**：默认算法 **GRPO**。

**RL 主循环**在 `verl/trainer/ray_trainer.py`：rollout → 算 reward → `compute_advantage(adv_estimator=GRPO)`（`ray_trainer.py:134`）→ `compute_grpo_outcome_advantage`（`core_algos.py:151`）对同一 prompt 的 n 条采样做**组内均值/方差归一化**（`core_algos.py:186-190` `scores[i]=(scores[i]-id2mean)/(id2std+eps)`）→ 带 KL 的 PPO-clip 更新（`config.yaml:24-25`，`use_kl_loss=true, kl_penalty=low_var_kl`）。

**Reward 函数**（视频用 `examples/reward_function/r1v.py`）：
```python
# 格式分：必须 <think>…</think><answer>…</answer>
format_match = re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", response, re.DOTALL)   # r1v.py:21-24
# 准确分：抽 <answer> 内首字母，mathruler.grade_answer 与 GT 比对                                # r1v.py:27-37
overall = 0.9*accuracy + 0.1*format                                                              # r1v.py:47
```
开放式 QA 走 LLM-judge（`vllm_rollout_spmd.py:286-307`，GPT 输出 `"yes"→1.0`）。

### MR-SP（核心创新）= 序列并行 + 视频 embedding 缓存

长视频 RL 的瓶颈是单条样本的视觉 token 极长（256 帧 × 257 ≈ 6.6 万 token，3600 帧时达 256K），单卡放不下也算不动。MR-SP 用 **Ulysses 序列并行**把这条长序列沿 token 维切到 SP 组的多张卡上，每卡只算 `1/SP` 的序列长度。longvila 脚本设 `ulysses_size=4`（`examples/new_supports/longvila_7b_video_grpo.sh:13`；`config.yaml` 默认 1，即关闭）。

#### (1) 序列怎么拼（切片前）
actor 前向先在 `dp_actor.py:399-407` 把三段拼成完整多模态序列：
```
input_embeds = embed_tokens(input_ids)                                   # 文本 token → embed
multi_modal_embeds = cat([ 文本前缀 , video_embeds , response ], dim=1)   # dp_actor.py:404
```
同时构造 `multi_modal_labels` 标记每个位置类型（`dp_actor.py:409-411`）：`IGNORE_INDEX(-100)`=视频 embed、`1`=文本、`-1`=padding。这个 label 是后面切片定位视频段的依据。

#### (2) 按"视频 token 段"切片（`prepare_inputs_for_sp_mm`，`dp_actor.py:115-198`）
关键设计：**只把视频 token 那一段均分到各 rank，文本前缀和 response 不切**。

- 视频 token 总数 `video_token_num = (labels==IGNORE_INDEX).sum()`（`:118`），每 rank 分得 `sp_middle_rank_len = video_token_num // sp_size`（`:145`）。
- 三类 rank 各取不同切片（`:149-185`）：
  - **rank 0（头）**：`[0 : video_first + sp_middle_rank_len]` —— 系统提示/文本前缀 + 第 1/SP 段视频；
  - **中间 rank**：`[video_first + len·rank : video_first + len·(rank+1)]` —— **只含自己那 1/SP 段视频 token**；
  - **最后 rank（尾）**：vila 模型取 `[-(sp_middle_rank_len + response_length) : ]` —— 最后 1/SP 段视频 **+ 完整 response**（`:162-166`，故 response 始终落在末 rank，便于后面只在该 rank 算 log_prob）。
- **position_ids 先全局生成再切片**（`:126-133`）：用"距首个非 padding 位的相对偏移"算出全局 position_ids，再按上面区间切——保证每 rank 拿到的是它在**原始全序列**里的真实位置，注意力位置编码不串味。
- **左 padding 对齐**（`:187-196`）：各 rank 切片长度不等（头/尾带文本与 response、长于中间 rank），统一左 pad 到 `output_length = max(input_text_length, response_length) + sp_middle_rank_len`（`:146`），padding 段 label=-1、attention_mask=False、position_ids=-1，不参与计算。

> 另有 `input_utils.py` 提供更细的切分原语：`extract_local_from_list`（`:26-30`，`divmod` 余数均摊的负载均衡切分）与 zigzag 变体（`:33-44`，供 Ring attention 用，使每 rank 同时拿序列首尾块、均衡因果掩码下的计算量）。

#### (3) Ulysses 注意力：序列切片下如何算"全局注意力"
切片后每 rank 只有 `seq/SP` 个 token 但持有**全部注意力头**，无法直接算需要全序列 K/V 的注意力。Ulysses 用 **all-to-all 转置**解决（`all_to_all.py:all_to_all_4D`，`monkey_patch.py` 把 HF 的 `_flash_attention_forward` 替换成 `UlyssesAttention`，`:230`）：
1. 进 attention 前 all-to-all：`(bs, seq/SP, head, dim) → (bs, seq, head/SP, dim)`（`ulysses_attn.py:158`）——换成"持有全序列、但只算 1/SP 的头"；
2. 每 rank 用 `flash_attn_varlen_func` 在**完整序列**上算自己那批头的注意力；
3. 再 all-to-all 转置回 `(bs, seq/SP, head, dim)`。
GQA 下若 `ulysses_size > num_kv_heads`，先 `expandKV` 把 KV 头复制 `sp//kv` 份再切（`ulysses_attn.py:163-165`），反向时按 query group 求和回收梯度（`all_to_all.py:76-98`）。

#### (4) 算完再拼回（gather）
前向输出 logits 后，**只在持有 response 的末 rank 段**算局部 `log_probs`（`dp_actor.py:437-438`），再 `gather_outputs_and_unpad(log_probs, gather_dim=1, unpad_dim=1)`（`:441`）跨 SP 组 all-gather 并去掉左 padding，最后切出 `[bsz, response_length]`（`:443`）。这样梯度回传时每 rank 只持有 `1/SP` 的激活，显存和算力都摊薄。

#### (5) embedding 缓存（与 SP 正交的第二招）
离线 `verl/utils/cache_video_embeds_vila.py`（`_embed_media_tokens:61` 算 embed、`:64` `torch.save` 成 `.pt`）；训练时 `dataset.py:328` 命中缓存则只放 1 帧占位、把缓存 embed 塞进 `multi_modal_data`，rollout 端 `use_cached_embeds` 跳过重复视觉编码（`vllm_rollout_spmd.py:227`）。SP 摊薄长序列计算 + 缓存免去重复视觉编码，合起来号称 **2.1× 加速**。

---

## 4. 一条真实数据的全过程（追踪 `I_Q3ajyOrcI.mp4` 样本）

1. **输入**：dataset 读到该 jsonl 行，`_get_messages_vila`（`dataset.py:96-120`）拼出 `<video>` + 要求 `<think>/<answer>` 的 prompt 模板；video 路径 = `video_dir/longvila_videos/I_Q3ajyOrcI.mp4`。
2. **抽帧/embedding**：`__getitem__`（`dataset.py:319-339`）—— 有缓存则 `torch.load(I_Q3ajyOrcI.pt)` 得到 `[num_video_frames, hidden]` 视频 embed（脚本设 `num_video_frames=256, tokens_per_frame=257`），并把 `num_video_frames` 临时设 1；结果存入 `example["multi_modal_data"]["video"]`。
3. **SFT 监督的 CoT**（上一阶段）：`reasoning` 字段被包成 `<think>…</think><answer>B</answer>` 做交叉熵；RL 阶段不再用 `reasoning`，只用 `answer="B"` 当 GT。
4. **RL rollout**：`vllm_rollout_spmd.py:217-231`，VILA 路径先把视频 embed 与文本拼成 `prompt_embeds`（≈256×257≈65792 视觉 token + 文本），vLLM 生成 **n=5** 条回答，每条形如 `<think>…cheetah…</think><answer>B</answer>`。
5. **reward 打分**：`r1v.py:compute_score` 对每条算 `format(0/1)` 与 `accuracy`（抽 `<answer>` 字母与 "B" 比），`overall=0.9*acc+0.1*fmt` → 5 个标量 reward。
6. **更新**：`compute_grpo_outcome_advantage` 对 5 个 reward 组内归一化成 advantage；actor 前向把 `cat(文本前缀, 256×257 视频 embed, response)` 这条 ~6.6 万 token 长序列经 `prepare_inputs_for_sp_mm`（`dp_actor.py:417`）沿**视频段**按 `ulysses_size=4` 切成 4 片——rank0=文本前缀+¼视频、中间 rank=各¼视频、rank3=末¼视频+完整 response，各片左 pad 对齐；Ulysses all-to-all 让每片在全序列上算 ¼ 的注意力头；末 rank 算出 `log_probs` 后 `gather_outputs_and_unpad` 跨 4 rank 拼回 `[bsz, response_length]`，再做 KL-正则 PPO-clip 梯度更新。

**张量流转**：`视频 mp4 → [256, hidden] cached embed → cat(文本embed, video_embed, response) ≈6.6万token → SP 沿视频段切 4 片(各≈len/4，左pad到 max(text,resp)+视频段/4) → 各 rank Ulysses all-to-all 算全序列×¼头注意力 → logits → 末rank log_probs → gather+unpad 拼回 [bsz, resp_len] → reward 标量×5 → GRPO advantage → 梯度`。

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
