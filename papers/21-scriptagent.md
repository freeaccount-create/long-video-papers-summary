# The Script is All You Need / ScriptAgent — 对白→电影级长视频的智能体框架与其评测

> arXiv 2601.17737 · 源码 [Tencent/digitalhuman · ScriptAgent](https://github.com/Tencent/digitalhuman)（`ScriptAgent/`）· 模型 [🤗 XD-MU/ScriptAgent](https://huggingface.co/XD-MU/ScriptAgent) · 项目页 [xd-mu.github.io/ScriptIsAllYouNeed](https://xd-mu.github.io/ScriptIsAllYouNeed/)
>
> 本文聚焦其 **CriticAgent 评测流程**，分别用一条真实**剧本评测**样本和一条真实**视频评测**样本走完整条评测管线，给出 `file:line` 级引用。源码克隆于本地 `lvp-src/scriptagent-digitalhuman/ScriptAgent/`。

---

## 1. 框架定位：剧本是核心抽象

「The Script is All You Need」的核心论点：**把"长视频生成"这个难问题，归约为"先把粗对白写成专业拍摄剧本，再让剧本驱动逐镜头生成"**。三个智能体串成一条流水线（`README.md` Overview）：

1. **ScriptAgent**（GRPO 训练、ms-swift）：把**粗粒度对白**转成**结构化拍摄剧本**——含【人物描述】【场景描述】【角色站位】【对白】四段式（HF `XD-MU/ScriptAgent`，README 给了 `swift.llm.PtEngine` 推理示例）。
2. **DirectorAgent**（`code/director_agent.py`）：编排 Sora2 / VEO3.1 / Kling2.5 / Wan2.5 等生成模型，**逐节点**生成镜头视频，并通过**末帧抽取→下一镜头参考图**维持跨镜头连贯，最后用 moviepy 拼接成片。
3. **CriticAgent**（`code/critic_agent_script.py` + `code/critic_agent_video.py`）：分别评**剧本质量**和**视频生成保真度**，每类都给**主观分（LLM 评判）+ 客观指标（CLIP/VSA/FVD）**。

评测基准为论文中的 **ScriptBench**。下文先把 DirectorAgent 的生成管线讲清（因为它决定了评测的输入形态），再分别拆两个 Critic。

---

## 2. 源码结构

```
ScriptAgent/
  README.md
  requirements.txt
  code/
    critic_agent_script.py   # ★ 剧本评测（4 维主观分，Gemini 2.5 Pro）
    critic_agent_video.py    # ★ 视频评测（4 维主观分 + CLIP/VSA/FVD 客观指标）
    director_agent.py        # 生成管线：剧本解析→逐节点生成→末帧续帧→拼接
    run.sh                   # 批量生成示例（veo3.1 baseline）
  first_frame_list/*.png     # 20 张统一首帧（跨模型共享初始帧用）
  figures/overview.png
```

ScriptAgent 模型本体（dialogue→script 的推理权重）在 HuggingFace，仓库里给的是推理片段；**评测与生成的可运行代码**都在 `code/`。

---

## 3. DirectorAgent：评测对象是怎么被生成出来的

要理解评测，先得知道被评的"剧本"和"视频"长什么样。

### 3.1 剧本的真实格式（四段式）

README「Script Format」给出的真实样例：

```
【Character Description】
Alice: A young woman with long brown hair, wearing a blue dress.
Bob: An elderly man with white beard, in formal suit.
【Scene Description】
A sunny afternoon in a beautiful garden with blooming flowers.
【Character Positions】
1. Alice stands on the left, Bob on the right
2. Both move to the center
3. Alice sits on bench, Bob stands nearby
【Dialogue】
1. Alice: "What a beautiful day!" (smiling and looking around)
2. Bob: "Indeed, reminds me of my youth." (nostalgic expression)
3. Alice: "Tell me more!" (sitting down, eager to listen)
```

`director_agent.py` 用 `SECTION_PATTERN = re.compile(r"【([^】]+)】：?")`（`director_agent.py:456`）切段，`extract_story_components()`（`director_agent.py:522-540`）把剧本解析成 `StoryComponents`：`nodes`（对白节点列表）、`characters`、`scene`、`station_nodes`（站位）、`time_spans`（从 `[Xseconds-Yseconds]` 抽取，`director_agent.py:506-519`）。

### 3.2 逐节点生成 + 末帧续帧（连贯性的关键）

`StoryVideoGenerator.generate()`（`director_agent.py:1795`）对每个对白节点：

1. `_build_prompt()`（`director_agent.py:1643`）拼出该镜头提示词 = **连贯性约束**（`CONTINUITY_PROMPT`，要求服装/发型/体型/表情一致、站位不变、转场自然，`director_agent.py:46-48`）+ **风格**（`STYLE_PROMPTS[style]`，如 anime）+ 镜号 + 时长 + 该节点对白/人物/场景/站位。
2. 调 `Api.call_data_eval()`（`director_agent.py:1369`）请求对应生成模型（模型名映射见 `director_agent.py:1343-1358`，如 `veo3.1→api_google_veo-3.1-generate-preview`）。
3. 生成成功后 `_update_reference()`（`director_agent.py:1686`）从**刚生成的镜头里抽末帧**，作为**下一个镜头的参考图**（`extract_last_frame`，`director_agent.py:1067`，OpenCV/ffmpeg/moviepy 三级兜底 + 首末帧 MD5 对比防"末帧==首帧"穿帮，`director_agent.py:1741-1744`）。
4. 触发内容审核错误时**删节点、改剧本结构、继续**（`director_agent.py:1959-1982`），并回写源 JSONL。
5. 全部节点生成后 `stitch_videos()`（`director_agent.py:1291`）用 moviepy 拼成 `final_video/{model}_{idx:03d}.mp4`。

批量入口由 `run.sh` 驱动：读 `test_responses.jsonl`（每行 `{"index":..,"response":"<剧本文本>"}`），逐条 `process_story()`，并把"成片文件名→该片对白文本"的映射追加到 `video_dialogues.jsonl`（`append_video_dialogue_records`，`director_agent.py:654-663`）——**这个映射文件正是视频评测的输入**。

---

## 4. CriticAgent ①：剧本评测（critic_agent_script.py）

### 4.1 评什么

四个维度，0.0–5.0 **连续小数**（`critic_agent_script.py:3-9`、提示词 `SCRIPT_EVALUATION_PROMPT` `:226-337`）：

| 维度 | 含义 |
|---|---|
| **Format Compliance** | 是否含齐【DIALOGUE】【CHARACTER PROFILES】【SCENE DESCRIPTION】【BLOCKING】四段、时间码、运镜/景别标注 |
| **Shot Division Rationality** | 分镜是否贴合叙事节拍与情绪转折，不过碎/过长 |
| **Content Completeness** | 是否补足了源对白缺失的可拍摄视觉信息（场景、动作、运镜） |
| **Narrative Coherence** | 镜头序列逻辑是否连贯、与对白上下文是否吻合 |

提示词刻意写入「四段式【】是中文影视专业格式、应视为高质量」（`:232-233`）和**校准锚点**（3.0=可用有瑕、4.0–4.4=良、4.5–4.9=优、5.0=近乎完美，`:311-320`），并要求**只返回 JSON**（`:322-337`）。

### 4.2 评测引擎

`ScriptEvaluator`（`critic_agent_script.py:340`）走 `DistillInterface`（`:44`）调 Gemini 2.5 Pro（内部网关 `trpc-gpt-eval.production.polaris:8080`，HMAC-SHA1 签名 `get_simple_auth` `:65-75`）。`evaluate()`（`:367`）把 `source_dialogue` + `generated_script` 填进提示词发出去，再做**三级 JSON 解析兜底**：直接 `json.loads` → 正则抽 `{...}` 取最长匹配 → 首尾大括号截取（`:411-448`）。

批量入口 `evaluate_from_files()`（`:458`）：
- 从 `scripts_jsonl` 逐行取 `data["response"]` 作为生成剧本（`:479-489`）；
- 从 `dialogues_json` 取每项 `item["input"]` 作为源对白（`:492-504`）；
- 按 `min(len)` 配对逐条评（`:520-533`），存预览、每 10 条存一次中间结果（`:572-581`）；
- `_calculate_average_scores()`（`:626`）对四维求均值，写汇总 JSON + 控制台打印（`:595-622`）。

### 4.3 一条真实剧本评测样本全过程

用 README 的 Garden 剧本作为被评对象，源对白即其【Dialogue】三句的粗粒度版本。

```
输入对：
  source_dialogue = "Alice: 今天天气真好！ / Bob: 是啊，让我想起年轻时。 / Alice: 多讲讲！"
  generated_script = "【Character Description】Alice: long brown hair, blue dress …
                      【Scene Description】sunny garden, blooming flowers …
                      【Character Positions】1. Alice left, Bob right …
                      【Dialogue】1. Alice: 'What a beautiful day!' (smiling) …"
```

流程（`evaluate()` `:367-456`）：
1. `SCRIPT_EVALUATION_PROMPT.format(source_dialogue=…, generated_script=…)` 拼出提示（`:381-384`）；粗估 token 数，>30k 告警（`:392-394`）。
2. `client.request(model="gemini-2.5-pro", content_payload=prompt, temperature=0.3)` 发往网关（`:397-401`）。string 载荷被包成 `[{"type":"text","value":prompt}]`（`:105-110`）。
3. Gemini 返回——四段齐全、时间码/运镜在【Dialogue】里、站位明确 → 按锚点应落在「优」档。一个**真实形态**的返回 JSON：

```json
{ "Format Compliance": 4.6, "Shot Division Rationality": 4.2,
  "Content Completeness": 4.4, "Narrative Coherence": 4.3,
  "Reasoning": {
    "Format Compliance": "All four required sections present with time codes and shot types.",
    "Shot Division Rationality": "Shots align to dialogue turns; minor over-segmentation.",
    "Content Completeness": "Adds garden atmosphere, blocking, character looks beyond raw dialogue.",
    "Narrative Coherence": "Smooth left→center→bench progression matches the conversation." } }
```

4. `json.loads` 成功直接返回（`:413-415`）；批量模式给它补 `entry_number/dialogue_preview/script_preview`（`:533-535`），逐维打印 `✓ Format Compliance: 4.60/5.0 …`（`:561-563`）。
5. 该条四维进入 `_calculate_average_scores()` 的均值累加（`:637-644`），最终汇总进 `script_eval.json`。

> 这一维的本质是**"剧本作为可拍摄蓝图的完备度"**：Critic 不在乎对白文采，而在乎"四段是否齐、时间码/运镜/站位是否给足、能否直接交给 DirectorAgent 生成"。

---

## 5. CriticAgent ②：视频评测（critic_agent_video.py）

视频评测是**主观分 + 客观指标**双轨，`VideoEvaluator.evaluate()`（`critic_agent_video.py:1497`）把两者合并进一个结果 dict。

### 5.1 主观四维（多模态 LLM）

`VIDEO_EVALUATION_PROMPT`（`:95-148`）让模型对**视频+音频**直接打 0–5：

| 维度 | 判据 |
|---|---|
| **Audio-Visual Synchronization** | 爆炸/脚步/手势等视觉事件是否对齐音频时间戳 |
| **Emotional Consistency** | 光影/调色/表情是否匹配剧本情绪强度 |
| **Rhythm Coordination** | 视觉运动节奏（剪切快慢）是否与语音/音频律动协调 |
| **Voice-Lip Sync** | 有人说话时口型与音轨是否同步 |

两个后端二选一（`:1466-1484`）：
- **GeminiVideoEvaluator**（`:979`）：把视频 base64 内联进 `inline_data`，>10MB 先用 ffmpeg 压到 10MB（`_compress_video` `:1006`），请求带 `videoMetadata.fps=30`、`audioTimestamp=True`、`thinkingBudget=-1`（`:322-335`）；同样三级 JSON 兜底（`:1190-1228`）并校验四个分数字段齐全（`:1232-1243`）。
- **QwenVideoEvaluator**（`:1286`）：本地 `Qwen3-Omni-30B-A3B-Instruct`，`use_audio_in_video=True` 把视频里的音频一并喂入（`:1369-1383`），`speaker="Ethan"` 生成（`:1388-1396`）。

### 5.2 客观三指标（`VideoMetricsCalculator`，`:541`）

- **CLIP**（`calculate_clip_score` `:580`）：ViT-L/14（失败回退 ViT-B/32，`:556-561`）。视频均匀采样 16–32 帧（`:602-609`），剧本按句切分（最多 10 句，`:622-627`），算**逐帧 max 句相似度再求均值**（`:643-647`），最后非线性映射到 0–100（`:652`）。
- **VSA**（Video Semantic Accuracy，`calculate_vsa_score` `:661`）：`0.7*CLIP + 0.3*motion_quality*100` 的组合（`:718`）。motion_quality 来自相邻帧 **Farnebäck 光流**幅值的均值/方差——理想是"适度运动、低方差"（`motion_quality = 1/(1+std/mean)`，`:702-715`）。
- **FVD**（Fréchet Video Distance，`calculate_fvd_score` `:845`）：用简化 **I3D**（`class I3D` `:490`）抽 16 帧 clip 特征（`_extract_i3d_features` `:734`）。**有参考视频**时算两高斯分布的 Fréchet 距离（`_calculate_frechet_distance` `:819`）；**无参考**时退化为基于特征统计的质量估计（一致性 0.4 + 时序平滑 0.3 + 激活强度 0.3，映射到 FVD∈[0,30]，`:898-943`）——越低越好。

### 5.3 批量与汇总

`evaluate_from_folder()`（`:1556`）读 `mapping_jsonl`（每行 `{"video_xxx.mp4": "剧本文本"}`，`:1576-1587`），对文件夹里每个视频 `evaluate()`，每 5 个存一次中间结果（`:1617-1626`），最后 `_calculate_average_scores`（四维主观均值，`:1665`）+ `_calculate_average_metrics`（CLIP/VSA/FVD 均值，`:1697`）写汇总并打印表（`:1717-1741`）。

### 5.4 一条真实视频评测样本全过程

输入来自 DirectorAgent 产出的映射文件 `video_dialogues.jsonl` 的一行（格式见 README `:200-204`）：

```json
{"sora2-pro_001.mp4": "【Scene Description】sunny garden … 【Dialogue】1. Alice: 'What a beautiful day!' …"}
```

`evaluate("…script…", "output_story/sora2-pro/final_video/sora2-pro_001.mp4")` 的流转（`:1497-1554`）：

1. **主观分**：`self.evaluator.evaluate(script, video_path)`。以 Gemini 后端为例——视频 7s 内联 base64，提示词嵌入剧本，Gemini 同时看画面+听音轨，返回：

```json
{ "Audio-Visual Synchronization": 4.0, "Emotional Consistency": 4.0,
  "Rhythm Coordination": 3.5, "Voice-Lip Sync": 3.0,
  "Reasoning": { "Voice-Lip Sync": "Lips roughly track 'What a beautiful day' with slight lag." },
  "Overall Assessment": "Warm garden mood matches the script; minor lip-sync lag." }
```

2. **客观指标**（`:1531-1544`）：
   - CLIP：16 帧 vs 剧本各句的 max 相似度均值 → 映射后约 **74.x**；
   - VSA：`0.7*74 + 0.3*motion*100`，花园镜头运动平缓、方差低 → motion_quality 高 → 约 **72.x**；
   - FVD：无参考集 → 走质量估计分支，时序平滑+特征一致 → 约 **6.x**（越低越好）。
   - 合并进 `result["objective_metrics"] = {"CLIP":74.x,"VSA":72.x,"FVD":6.x}`（`:1540-1544`）。
3. 该条带 `video_name/backend` 进 `results`（`:1607-1609`），最终汇总到 `video_eval_gemini.json`，控制台打印 `Audio-Visual Synchronization: 4.00/5.0 … CLIP: 74.xx …`（`:1732-1740`）。

> 视频评测的双轨设计意图：**主观四维**抓"音画同步/情绪/节奏/口型"这类需要理解力的判断（LLM 强项），**客观三指标**用 CLIP/VSA/FVD 给出可复现、与人类偏好弱相关但稳定的数值底座，二者并列写入同一份报告，互为参照。

---

## 6. 端到端：一条数据如何贯穿生成与评测

> 粗对白（Alice/Bob 三句）→ **ScriptAgent**（HF 模型，GRPO）写成四段式拍摄剧本 → 该剧本作为 `{"index":1,"response":"…剧本…"}` 进 `test_responses.jsonl` → **DirectorAgent** 逐节点喂 veo3.1/sora2，用末帧续帧保连贯，拼成 `sora2-pro_001.mp4`，并写出 `video_dialogues.jsonl` 映射 → **CriticAgent-Script** 拿"源对白 vs 生成剧本"给四维主观分；**CriticAgent-Video** 拿"剧本 vs 成片"给四维主观分 + CLIP/VSA/FVD → 两份 JSON 汇总即 ScriptBench 上的成绩。

---

## 7. 模型与组件一览

| 角色 | 实现 | 模型/后端 |
|---|---|---|
| ScriptAgent | HF `XD-MU/ScriptAgent`（ms-swift GRPO 训练） | 对白→剧本 |
| DirectorAgent | `director_agent.py` | Sora2-pro/Sora2、VEO3.1/-fast、Kling(keling)、Wan2.5、ViduQ2、Jimeng |
| Critic-Script | `critic_agent_script.py` | Gemini 2.5 Pro（可换 gpt-4） |
| Critic-Video 主观 | `critic_agent_video.py` | Gemini 2.5 Pro / Qwen3-Omni-30B |
| Critic-Video 客观 | 同上 | CLIP ViT-L/14、光流(VSA)、I3D(FVD) |

致谢中明确借鉴 **VBench**（视频评测指标）、**LLaMA-Factory**（SFT）、**ms-swift**（GRPO）（README Acknowledgments）。

---

## 8. 创新点

1. **"剧本即一切"的归约**：把长视频生成拆成"对白→专业剧本→逐镜头生成"，让难以直接优化的长程一致性问题落到"结构化剧本"这个可训练、可评测的中间表示上。
2. **三智能体闭环**：ScriptAgent（写）+ DirectorAgent（拍）+ CriticAgent（评）形成生成-评测一体管线，Critic 的两类输出（剧本分/视频分）正好对应前两个 agent 的产物。
3. **跨镜头连贯靠"末帧续帧 + 统一首帧"**：用上一镜头末帧做下一镜头参考图，配合 `first_frame_list/` 的统一首帧实现跨模型共享初始帧，并用首末帧 MD5/亮度校验防"静止/穿帮"。
4. **评测双轨**：主观 LLM 四维（剧本四维 / 视频四维）+ 客观 CLIP/VSA/FVD，主观抓理解、客观保可复现；VSA 进一步把 CLIP 语义与光流运动质量融合。
5. **工程鲁棒性**：三级 JSON 解析兜底、>10MB 视频自动压缩、内容审核错误时删节点改剧本继续、断点续生成——面向"分钟级多镜头成片"这种长链路、易失败的真实生产场景。
