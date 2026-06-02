# 16 · VideoAuto-R1（Thinking Once, Answering Twice 的视频自适应推理 RL）

> CVPR 2026 · [arXiv 2601.05175](https://arxiv.org/abs/2601.05175) · [项目页](https://ivul-kaust.github.io/projects/videoauto-r1) · 官方代码：[github.com/IVUL-KAUST/VideoAuto-R1](https://github.com/IVUL-KAUST/VideoAuto-R1)

基于**官方源码逐行分析**写成（克隆成功，约 2.9M，Apache-2.0；标注 JSON 在 HF `IVUL-KAUST/VideoAuto-R1-Data`，repo 内无 json）。出自 KAUST IVUL 组（一作 Shuming Liu）。

---

## 1. 源码可得性 / 结构

- `videoauto_r1/`：训练主体——`train_grpo_qwen2_5_vl.py` / `train_grpo_qwen3_vl.py`（GRPO 入口）、`train_sft_qwen2_5_vl.py`；`reward/`（`reward.py`、`mc_grader.py`、`tg_grader.py`）；`trainer/`（`grpo_vllm_trainer_qwen2_5_vl.py` 1593 行，GRPO+vLLM）；`utils/data_rl.py`（数据加载）；`model/`（Qwen2.5-VL / Qwen3-VL monkey-patch）。
- `lmms_eval/`：评测框架（含 `models/simple/early_exit.py` 置信度早退）。
- `data/data_config.yaml`：11 个训练源路径声明。
- `scripts/train/grpo_autothink/*.sh`、`inference_demo.py`。

---

## 2. 数据来源与真实格式

训练混合 11 个数据集（脚本）：文本数学 DAPO-Math、图像 ViRL/ThinkLite-VL-Hard、视频 QA（VideoR1、TVBench、STI-Bench、MMR-VBench）、时序定位 TVG（Charades-STA、ActivityNet-TVG、TimeR1）、定位+QA（NeXT-GQA）。

repo 无 json 摘录，但各 loader 明确真实字段。**Video-R1** 视频多选样本（`data_rl.py:287-334`）原始字段：
```json
{"problem_type": "multiple choice", "problem": "...", "options": ["A. ...", ...],
 "path": "videos/xxx.mp4", "data_type": "video", "solution": "<answer>B</answer>"}
```
TVG（Charades）：`{"description","timestamps":[s,e],"video"}`（`data_rl.py:511-542`）。NeXT-GQA：`{"question","options","answer","timestamps","video"}`（`data_rl.py:544-578`）。loader 统一转成 `{"messages":[system,user], "response": <gt>, "problem_type": exact_match|math|iou|gqa}`（`data_rl.py:154-206`）。

---

## 3. 完整方法 / 训练流程

**范式 = "Thinking Once, Answering Twice"，GRPO 训练**，rl_mode=`answer_twice_rl`。base = Qwen2.5-VL-7B / Qwen3-VL-8B。

- **System prompt 强制模板**（`data_rl.py:20-27`）：`\boxed{初答}<think>推理</think>\boxed{复核答}`；若无法直答则首框输出 `\boxed{Let's analyze the problem step by step.}`。
- **采样**：每 prompt 出 `G=16` 条 completion，但**不是靠 `SamplingParams(n=16)`**。实现是两段配合：
  1. **数据侧把 prompt 复制 16 份**：`_get_train_sampler` 返回 `RepeatSampler(mini_repeat_count=self.num_generations)`（`:749-754`），即同一条 prompt 在一个 batch 内被连续重复 `num_generations=16` 次进入 dataloader。
  2. **vLLM colocate 每卡只生成 1 条**：真正的生成在 vLLM 路径 `:1125-1169`，`generation_kwargs["n"]=1`，源码注释直言 `# vLLM on each GPU generates only 1 in colocate mode`（`:1126`）。16 份重复 prompt 各自 n=1 生成，合起来得 16 条采样。
  （注：`:1205-1217` 的 `unwrapped_model.generate(**prompt_inputs, generation_config=...)` 是**无 vLLM 时的 transformers 后备路径**，并非默认 vLLM 生成路径，不要据此理解 G 的来源。）
- **奖励**（3 函数，权重 `0.9/1.1/1`，`reward.py:167-174`）：
  1. `accuracy_boxed1`（权 0.9）：取**第一个** `\boxed`（`<think>` 之前）判分（`reward.py:91-118`）；
  2. `accuracy_boxed2`（权 1.1）：取**第二个** `\boxed`（`</think>` 之后）判分（`reward.py:121-153`）；含 **fallback 奖励**：复核答正确（reward>0.7）且初答是 `Let's analyze...`（诚实选择需思考）时额外加 `0.3/1.1`，惩戒"虚假初猜"（`reward.py:144-147`）；
  3. `format_twice_boxed`（权 1）：正则校验严格 boxed-think-boxed 格式（`reward.py:156-164`）。
- **判分按 `problem_type` 分流**（`reward.py:52-68`）：`exact_match`→`equal_answer`（`mc_grader.py:61`）；`math`→`math_verify`；`iou`→区间 IoU（`tg_grader.py:86-108`）；`gqa`→答案 IoU 相加。
- **优势归一（GRPO 组内标准化）**（`grpo_vllm_trainer_qwen2_5_vl.py:1327-1346`）：`rewards=Σ wᵢ·rewardᵢ`，按 G 重排求组均值/标准差，`advantages=(rewards−mean_group)/(std_group+1e-4)`。
- **损失**：PPO 式裁剪 + KL 正则 `β=0.01`（`:1486-1494`）。tune mm_llm+mlp，冻结 vision。

**推理（自适应早退，`inference_demo.py`）**：Stage-1 生成到 `<think>` 停（:95）→ `compute_first_boxed_answer_probs` 算首框答案长度归一化置信度（token logprob 均值取 exp，`early_exit.py:67`）→ 若 ≥ τ(=0.98) 则**直接早退**不推理；否则续 `<think>` 完成 CoT 输出复核答（:119-157）。

---

## 4. 一条真实数据的全过程

以一条 Video-R1 多选样本为例：
1. **原始**：`{problem_type:"multiple choice", problem:"What does the person do after...", options:[...], path:"videos/x.mp4", data_type:"video", solution:"<answer>B</answer>"}`。
2. **加载**（`data_rl.py:287-334`）：拼问句+Options；`extract_answer` 取 `B`；`problem_type→"exact_match"`；构造 messages（system=answer-twice，user=[video+text]）。
3. **采样**：vLLM 生成 16 条，如 `\boxed{B}<think>The person first... so it's B</think>\boxed{B}`。
4. **奖励**：boxed1 取首框 `B`==`B`→1.0×0.9；boxed2 取末框 `B`→1.0×1.1（首框非 `Let's analyze` 不触发 fallback）；format 合法→1.0×1。该条 `rewards=3.0`。
5. **优势**：与同组 15 条标准化 `(3.0−mean)/(std+1e-4)`（:1346）→ 裁剪+KL 损失更新策略。
6. **推理时**：若首框 `B` 置信度 ≥0.98 直接早退输出 `B`（不生成 CoT），否则触发 `<think>` 推理后输出复核答。

---

## 5. 模型 / 组件

- **Base model**：Qwen2.5-VL-7B-Instruct、Qwen3-VL-8B（patched，`model/modeling_qwen*_patched.py` + `monkey_patch.py`）。
- **RL 算法**：GRPO（组内优势归一）+ PPO 式裁剪 + KL 正则（β=0.01），自实现 vLLM colocate trainer。
- **奖励组件**：`math_verify`、自写 MC 归一器、IoU/GIoU/DIoU 时序定位 grader。
- **推理组件**：长度归一化置信度早退（`early_exit.py`）；评测基于 `lmms_eval`。

---

## 6. 创新点

1. **Thinking Once, Answering Twice 范式**：`\boxed{初答}<think>推理</think>\boxed{复核答}` 单序列内同时监督初答与复核答（双 boxed 双奖励），让模型学会"先答再按需思考"。
2. **置信度自适应早退**：推理期用首框答案长度归一化置信度（`early_exit.py:67`）对比 τ 决定是否进入 CoT，平均响应长度降约 3.3×（144→44 tokens），感知类低思考率、推理类高思考率。
3. **Fallback 奖励防"虚假初猜"**：模型选择 `Let's analyze...`（声明需思考）且复核答正确时给额外奖励（`reward.py:144-147`），鼓励对真正需推理的题诚实弃直答，避免初答乱猜被偶然命中。

> 说明：训练标注 json 在 HF 单独下载，第 2/4 节字段依据 `data_rl.py` 各 loader 真实解析逻辑给出。
