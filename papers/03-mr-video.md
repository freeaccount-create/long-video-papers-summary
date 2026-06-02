# 03 · MR. Video（MapReduce for Long Video）

> NeurIPS 2025 · [arXiv 2504.16082](https://arxiv.org/abs/2504.16082) · 官方代码：[github.com/ziqipang/MR-Video](https://github.com/ziqipang/MR-Video)

> **重要前提**：官方仓库目前**未发布 agent 源码**（README 写 "preparing the code and dataset step by step"），仅发布了 **caption 数据**（HF `ziqipang/MR-Video`，含 `lvbench_captions.zip` 等）。因此 agent 流程无 `脚本:行` 引用——这类引用在当前 repo 中不存在。本文用两条可核验证据还原流程：(a) 论文方法描述与 prompt 片段；(b) 官方发布的**真实 LVBench caption 数据**（已下载解压，100 个视频逐段 caption），它正是 Map/Reduce 第一阶段的真实中间产物。

---

## 1. 源码可得性

- Repo `github.com/ziqipang/MR-Video`：克隆成功，但只含 `README.md`、`LICENSE`、`assets/overview.png`、`assets/teaser.png`——**无 `*.py`、无配置、无 prompt 文件**。
- 已发布数据（HF）：`lvbench_captions.zip`（6.8 MB → 100 个 `cut_caption/{youtube_id}.json`）等四个 benchmark 的 caption。
- 结论：**无法做 agent 代码的 file:line 级追踪**；基于论文 + 真实 caption 数据还原。

---

## 2. 数据 / 输入格式

- **Benchmark**：LVBench、EgoSchema(Val)、Video-MME(Long)、LongVideoBench(Val)。主战场 LVBench（+10% over SOTA）。
- **caption JSON**：段列表，每段 `t_start / t_end（秒）/ caption（三段式文本）`。真实例 `2sriHX3PbXw.json` 共 437 段，窗口非等长（`(0,10),(10,13),(13,22.5)…`，证明做了场景切分）。
- **问题输入**：LVBench 多选题，如 *"How many sticks does the protagonist put in the incense burner? A.3 B.2 C.5 D.1"*。问题在第二阶段才注入。
- **一条真实 caption（Map 真实产物）**：
```
[1. Brief Description]: The rats scurry across the floor, then the scene changes to <woman_a-1> standing in a room.
[2. Appeared Characters]: [rat_a-6, woman_a-1]
[3. Detailed Description]: A large group of black rats, including <rat_a-6>, scurries ... <woman_a-1> standing in a room, looking distressed...
```

---

## 3. 完整方法流程（两轮 MapReduce）

**模型分工**（论文 §4.2）："consistently utilize **Gemini-2.0-Flash** as our VLM, **GPT-4o** as the default LLM"。成本：1 小时视频 caption ≈ $0.8，每问 ≈ $0.4。

### 第一轮 = 阶段 A：Captioning（§3.2）
- **Map — 短段切分 + 并行 caption**（VLM = Gemini-2.0-Flash）：
  1. 对每个 10s 短片（20 帧 @2fps）先 prompt VLM 判断是否单一场景，含多场景则返回过渡帧 index → 得到原子单位"scene"（解释了 caption 的非等长窗口与 `BEGIN/END_OF_NEW_DESCRIPTION` 标记）。
  2. 对 2min 段（30 帧 @0.25fps）列出显著角色/物体及最佳展示帧，prepend 到 caption 上下文。
  3. 逐 scene 三段式 caption：`[1. Brief Description] / [2. Appeared Characters] / [3. Detailed Description]`。
- **Reduce — 人名/物体名归一化**（仍由 VLM 看 salient frames 做关联）：合并"同人不同名"、拆分"同名不同人"，分配**全局唯一新名**并回写所有 caption，统一格式 `<NAME>`。真实数据形态：`<person_d-10>`、`<woman_a-1>`、`<rat_a-6>`，命名规则 `<类型_字母-数字>`。

  **真实 caption 数据反推证据**（实测 `2sriHX3PbXw.json`）：全文件 **437 段**；场景切分标记 `BEGIN_OF_NEW_DESCRIPTION` / `END_OF_NEW_DESCRIPTION` **各 268 次**，印证 Map 阶段确实做了"多场景片段再切分"（437 段 > 268 个切分点说明部分段被进一步分割）；`<类型_字母-数字>` 实体引用共 **1155 个 token、26 个唯一实体**，证明 Reduce 把全片角色归并到了 26 个全局唯一名。其中 `<person_d-10>` 出现最多——**321 次是"全文 token 级"出现数**（占 1155 个实体 token 的 27.8%），**段级**则是出现在 **113/437 段**（25.9%）；二者口径不同，"主角 = 出现最频繁者"由此成立。

### 第二轮 = 阶段 B：Analysis（逐问题，LLM = GPT-4o）
- **§3.3 问题意图分析 + 定位**：Map 把视频按~32 scene 切大段，LLM 读"聚合 caption + 中间帧"判断段内相关 scene，恢复 when/where/who 隐含线索；Reduce 聚合成视频级统一意图，产出候选 scene 集合。
- **§3.4 目标导向感知 + 推理**：Map（受 ViperGPT 启发）让 LLM 自行向 VLM 提出定制查询（真实例 `vlm.query("How many incense are put into the burner?")`）；Reduce 用 Local（段内密集采帧，细节）/ Global（跨段稀疏采帧，全局推理）整合出最终 `[Answer]`。

---

## 4. 一条真实数据走完整流程

问题：*"How many sticks does the protagonist put in the incense burner?（A.3 B.2 C.5 D.1）"*（概览图真实题），用真实视频 `2sriHX3PbXw` 数据形态演示：

1. **Map(caption, Gemini-2.0-Flash)** → 每 scene 三段式产物，如窗口 13.0–22.5s：`[1] The camera pans to the left, revealing <person_h-10> standing...`（角色名此时还可能不一致）。
2. **Reduce(角色归一化, VLM)** → 全视频统一为 `<person_d-10>` 等并回写（即下载到的最终 JSON）。"主角" = 出现最频繁者 `person_d-10`（全文 token 级 321 次 / 段级 113 段，均居首；见 §3.2 数据反推证据）。
3. **Map(意图分析, GPT-4o)** → 读 32-scene/段聚合 caption，定位含"sticks / incense burner / 主角放东西"的候选 scene（概览图标注命中 `10:45, 10:50`，`40:40` 为干扰段）。
4. **Reduce → Goal-Aware**：GPT-4o 提出 `vlm.query("How many incense are put into the burner?")`，对候选 scene Local 密集采帧让 Gemini 数 stick 数 → 推出 **[Answer]: D(1)**。

中间产物形态：caption 阶段 = 三段式文本 + `<NAME>` 列表；意图阶段 = 候选 scene 时间戳 + 相关性判断；目标阶段 = LLM 生成的 VLM query 字符串 + VLM 逐帧计数 → 最终单个选项字母。

---

## 5. 模型 / 组件（代码/论文明确写出）

- **看帧的所有步骤（caption / 角色关联 / VLM query）**：**Gemini-2.0-Flash**。
- **Reduce / Analysis / 答题 LLM**：**GPT-4o**。
- 框架定位：**training-free**，仅需"1 个 LLM + 1 个 VLM"，无微调。

---

## 6. 创新点

1. **把 MapReduce 作为长视频理解的统一原则**：Map 并行密集感知短片、Reduce 全局聚合推理，兼顾全局语境 + 局部细节，规避 context 长度限制，比"迭代检索关键帧"的 video agent 更可并行扩展。
2. **带角色/物体一致性的两步 caption**：先做场景均匀性检查切原子 scene，Reduce 阶段跨片合并/重命名为全局唯一 `<NAME>`，解决并行 caption 的"同名异人/同人异名"问题。
3. **意图分析 + 目标感知双阶段推理**：先显式恢复 when/where/who 并结合宽上下文定位候选片段，再借鉴 Visual Programming 让 LLM 自行向 VLM 提 query，按 Local/Global 灵活取帧——LVBench 60.8%，较 SOTA 提升 >10%。
