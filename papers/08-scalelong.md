# 08 · ScaleLong（benchmark）

> ICLR 2026 · [arXiv 2505.23922](https://arxiv.org/abs/2505.23922) · 官方代码：[github.com/multimodal-art-projection/ScaleLong](https://github.com/multimodal-art-projection/ScaleLong) · 数据 `m-a-p/ScaleLong`

基于**官方源码逐行分析**写成（克隆成功）。

---

## 1. 源码结构（基准自有代码；`LLaMA-VID/LLaVA-Mini/...` 为内嵌被测模型）

- `inference.py`：评测主入口（构造 prompt、调模型、抽答案、判分、统计）。
- `video_bench/video_process.py`：抽帧/`smart_resize`。
- `video_bench/registry.py`：模型注册表；`video_bench/models/*.py`：20 个被测模型封装（另有 `basic_model.py` 基类）。
- `download_video.sh`：`yt-dlp` 按 `dataset/test_video_ids.txt` 下视频。

> 数据不在 repo 内：题目 `questions.jsonl` 与视频 tar 仅托管于 HF。

---

## 2. 数据 / 真实格式

HF `questions.jsonl` 实测 **269 行 = 269 个视频**，每行 `{video_id: [8 道题]}`，**每视频 8 题、四尺度各 2 题**（实测分布 `{'Video Clip':2,'Video Shot':2,'Video Event':2,'Video Story':2}`）。

四尺度落在同一视频上靠每题 `answer_location`（答案所在时间区间）区分粒度。实测题目跨度中位数清晰呈层级递增：**Clip 3s → Shot 10s → Event 90s → Story 2988s(~50min)**。

字段（每题 7 个）：`data_id, question, question_type, granularity, answer, answer_location, distractors`。原始数据是"正确答案 + 3 个干扰项"，**选项字母在评测时才随机生成**。

**一条真实样本**（视频 `uIj03RsGrJA`，Video Clip 题，逐字摘录）：
```json
{"data_id": 0,
 "question": "How many matches did Chinese athletes win before the gold medal match?",
 "question_type": "Counting Problem",
 "granularity": "Video Clip",
 "answer": "Chinese athletes won a total of five matches before the gold medal match",
 "answer_location": "0:01-0:04",
 "distractors": ["...six matches...","...four matches...","...three matches..."]}
```

---

## 3. 完整评测流程（全部在 `inference.py`）

1. **载题 + 构造选择题**：`load_questions_from_jsonl`(:194) 逐行读，`convert_to_multiple_choice`(:160) 把 `distractors+answer` **`random.shuffle`**(:163)，`chr(ord('a')+i)` 生成 a/b/c/d(:167)，记录 `correct_option` 字母(:187)。题面 `format_question`(:156) 拼 `问题？\na. ...\nb. ...`（选项一律小写字母） 并追加固定指令 `MULTI_CHOICE_PROMPT="Answer with the option's letter from the given choices directly."`。
2. **抽帧**：各模型自带，如 internvl2_5 `load_video`(:151) decord 读全片、`get_index`(:133-148) 在 `[start_idx,end_idx]` **均匀取 num_segments 帧**。其签名默认 `num_segments=32`（`internvl2_5.py:133`，非 128；脚本沿用 32）。采样点是**每段的中心**而非段首——`seg_size=(end_idx-start_idx)/num_segments`，第 i 帧索引 `int(start_idx + seg_size/2 + round(seg_size·i))`（`:141-146`），`seg_size/2` 偏置确保取段中点、避免系统性偏向片段开头。无 `bound` 时 `start/end=∓100000` 即退化为全片均匀采样。抽帧与题目 `answer_location` 无关——模型须自己在长视频里定位。
3. **构造 prompt**：`process_video_questions`(:297) 用 `ffprobe` 取真实时长，把全局 `PROMPT`（`"...divided into {frame_num} evenly spaced frames spanning {duration} seconds..."`）拼到题面前；internvl 封装再加 `Frame1: <image>\nFrame2: ...` 前缀（`internvl2_5.py:241`）。
4. **模型作答**：`model.generate_video_only(...)`(:337)，贪心 `do_sample=False, max_new_tokens=1024`。
5. **答案抽取 + 判分**：`check_answer`(:227)——**纯正则/规则，无 LLM**：优先匹配 `<answer>: X`(:236)；否则找 `(a)`/` a `/`a.` 候选字母取**最后一个**(:241-245)；无输出则**随机猜**(:231,:245)；返回 `(pred==correct_option, pred)`。
6. **判分指标 = 多选准确率**：逐题累加 `total/correct`，按 `question_type` 与 `"total"` 统计，`correct_rate=correct/total`(:274,:433)。
7. **LLM 兜底匹配**：**没有**。`is_text_similar`(SequenceMatcher) 已定义但在 `check_answer` 中未被调用；判分链路不调用任何 LLM。

---

## 4. 一条真实数据的四尺度全过程（视频 `uIj03RsGrJA`，nframes=32）

对同一视频均匀抽 32 帧、同一 `PROMPT` 前缀，四题各跑一遍 generate→`check_answer`→准确率累加：

| 尺度 | data_id | 题目(节选) | 正确答案 | answer_location | 跨度 |
|---|---|---|---|---|---|
| Clip | 0 | How many matches did Chinese athletes win before the gold medal match? | "...five matches..." | 0:01-0:04 | ~3 s |
| Shot | 2 | What color sports drink ... after a yellow one? | "...blue sports drink." | 18:33-18:48 | 15 s |
| Event | 4 | What did Chinese athletes do during their first break? | "...wiped sweat and drank a sports drink." | 31:54-32:10 | 16 s |
| Story | 6 | Final scores of the two athletes in the first game? | "21:15" | 05:51-30:26 | ~25 min |

每题：输入 = `PROMPT(frame_num=32,duration)` + `问题？\n a.<shuffle 选项>...\n` + `MULTI_CHOICE_PROMPT` + 32 帧 `Frame_i:<image>`；输出 = 贪心生成文本（期望一个字母）；判分 = `check_answer` 提字母与该题随机 `correct_option` 比较，命中 `correct+1`。论文层面再按 `granularity` 聚合得到四条曲线。

---

## 5. 模型 / 组件

- **被测模型（`video_bench/models/` 实测 20 个封装 + 注册名）**：InternVL2、InternVL2.5、Qwen2-VL、Qwen2.5-VL、LLaVA-OneVision、LLaVA-Video、LLaVA-Video-Image、LLaVA-Mini、LLaMA-VID、LongVA、LongVILA、NVILA、LongVU、MiniCPM-V、MiniCPM-o、Aria、Phi-3.5、Phi-4、VideoLLaMA3、Mammoth-VL。（闭源 GPT-4o/Gemini 及 mPLUG-Owl3/Ola/InternVideo2.5 仅在 `inference.py` 的 `try/except ImportError` 中尝试导入，repo **未附**其封装，默认不可跑。）
- **抽帧/预处理**：decord（首选）/ torchvision 后备，`smart_resize` 控分辨率，`ffprobe` 取时长。
- **判分辅助模型**：**无**——纯规则正则判分。

---

## 6. 创新点

1. **Intra-Video Multi-Timescale 设计**：在**同一段长视频**内嵌入 Clip/Shot/Event/Story 四层级各 2 题，实测跨度中位数 3s→10s→90s→2988s 呈数量级递增，在叙事上下文一致前提下分离"时间粒度"这一变量。
2. **"U 型曲线"发现**：准确率在两端（Clip、Story）最高、中间尺度（Shot、Event）显著下陷，呈 U 形——揭示 MLLM 对"中等时间跨度"理解最弱。
3. **规模与多样性 + 任务画像**：269 个长视频（均 ~86min）、5 大类 36 子类、5 种任务类型；发现 Object Recognition 最高、Counting 最低；消融给出"固定分辨率增帧普遍提升、固定 token 预算提分辨率收益递减甚至为负"。
