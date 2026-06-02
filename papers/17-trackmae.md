# 17 · TrackMAE（运动感知的视频掩码自编码器）

> CVPR 2026 · [arXiv 2603.27268](https://arxiv.org/abs/2603.27268) · 官方代码：[github.com/rvandeghen/TrackMAE](https://github.com/rvandeghen/TrackMAE)（仅占位 README，实现未提交）

> **源码尚未发布**：GitHub 仓库克隆成功，但仅含 `LICENSE` + 2 行占位 `README.md`（"[CVPR 2026] Original PyTorch implementation ..."），无任何实现。第 3–5 节基于论文 arXiv:2603.27268（ULiège & KAUST, Vandeghen 等），公式引用论文符号，**无 file:line 可引**。

---

## 1. 源码可得性

- **Repo**：`github.com/rvandeghen/TrackMAE`，克隆成功（commit `5102431` "Initial commit"），目录仅 `.gitignore / LICENSE / README.md`，**无源码**。
- 论文 HTML v1 已抓取。

---

## 2. 数据来源与真实格式

- **预训练**：Kinetics-400（K400）与 Something-Something V2（in-domain），ViT-B；消融用子集 `K400_s` + ViT-S；可扩展至 K700。
- **下游评测**（6 个）：K400、Something-Something V2、UCF-101、HMDB-51、FineGym、SEVERE benchmark。
- **真实样本字段**：一段短视频 clip
  - 张量 `V ∈ R^{T×H×W×3}`（按 VideoMAE 协议 T=16 帧、224×224）。
  - tubelet 切分（`2×16×16`，t=2,p=16），token 数 `N=(T/t)·(H/p)·(W/p)`。
  - **运动目标**（在线生成，非数据集自带）：CoTracker3 以**第一帧**的均匀网格中心点为 query 点，向后跨帧跟踪，得位移轨迹 `M ∈ R^{(T/t)×(H/p)×(W/p)×2}`（时间维与 tubelet 对齐为 T/t=8，2=x,y 位移）作为重建监督。

---

## 3. 完整方法 / 训练流程

双分支编码器-双解码器 MAE（论文图 2）。

**(a) Tubelet 嵌入**：3D conv 将 `2×16×16` tubelet 映射为 token，加固定位置编码，得 `T={τ_i}, τ∈R^D`。

**(b) Mask 策略**：
- 基线：随机 tube masking，token 级 Bernoulli，高掩码比例（像素重建 **90%**，CLIP 特征重建 **80%**）。
- **创新 — motion-aware masking**：用 CoTracker3 轨迹算每个 query 点时间平均位移 `M̄` 作采样分布；分 high-motion / low-motion 两个 uniform bins，用 motion ratio `ρ_motion=50%` 控制每 bin 抽取可见 token 数，保证可见 token 同时覆盖动/静区域（随机 tube masking 为其特例）。

**(c) 编码器**：标准 ViT（ViT-B 主 / ViT-S 消融），只编码可见 token `Z=Φ(T^visible)`。

**(d) 双解码器**：`Z` 拼可学习 `[MASK]` token + 位置编码后送入：
- 空间解码器 `Ĉ=Ψ_spatial([Z,[MASK]])`（重建像素或语义特征）；
- 运动解码器（轻量）`M̂=Ψ_motion([Z,[MASK]])`（预测轨迹位移）。

**(e) 损失**（均只在 masked token 上算，防泄漏）：
- 像素 `L_pixel=(1/|masked|)Σ‖c_i−ĉ_i‖₂²`；
- 特征 `L_feature=(1/|masked|)Σ‖f_i−f̂_i‖₂²`（f 由 CLIP ViT-B / DINO 提取）；
- 运动 `L_motion=(1/|masked|)Σ‖m_i−m̂_i‖₂²`（预测位移而非绝对坐标）；
- 总目标 `L=L_spatial+λ·L_motion`：像素重建 `λ=1`，特征(CLIP)重建 `λ=0.25`。

**(f) 运动目标技巧**：
- *运动目标的精确构造（归一化时间差分，§3.2 + §11.2）*：CoTracker3 直接输出的是各 query 点在后续帧的**绝对 2D 坐标** `(x,y)`，构成轨迹张量 `M ∈ R^{(T/t)×(H/p)×(W/p)×2}`。但 TrackMAE **不**回归绝对坐标，而分两步转成回归目标 `m_i`：① **时间差分（→位移）**沿时间维取相邻 motion token 的位置变化，使目标绑定"运动量"而非"画面位置"（§3.2 "predict the displacement ... instead of absolute trajectory values"）；② **归一化**——论文明确实际用的是 **normalized temporal differences**（§11.2 原文 "we use the normalized temporal differences ... as the target, rather than absolute values"），把不同视频/分辨率下尺度迥异的位移拉到统一尺度。这是 `L_motion` 数值稳定、并能与空间损失按固定 `λ` 线性加权而不被某支梯度量纲淹没的前提（也解释 Tab.10c 加轨迹噪声仅掉 0.5% 的鲁棒性）。*注*：论文仅文字声明"归一化时间差分"，**未给闭式公式**（按帧尺寸还是按轨迹统计量归一未明示），此处不臆造除数。
- *效率*：CoTracker3 每隔一帧喂入，输出时间维 size=2 与 tubelet 对齐。
- *Upsampling trick*：每 patch 只跟 1 点（稀疏），假设 patch 内邻近像素运动相近，空间插值上采样 `υ` 倍（等效每 patch 跟 `υ²` 点），主实验 `υ=2`（14×14→28×28），零额外跟踪开销即提分。

**关键超参**：K400 上 ViT-B，CoTracker3 offline grid 14×14，`υ=2`，`ρ_motion=50%`，CLIP ViT-B 特征目标，800 epochs，其余沿用 VideoMAE。

---

## 4. 一条真实数据的全过程（K400 一段 16 帧 224×224 clip `V`）

1. **运动目标提取（上分支）**：每隔一帧（stride-2）喂 CoTracker3，以第一帧 14×14 网格中心点为 query 跨帧跟踪 → 逐 token (x,y) 位移 `M`（时间维 T/t=8）；上采样 υ=2 → 28×28 密集位移目标。同时算时间平均位移 `M̄` 作 mask 采样分布。
2. **Patchify + Mask（下分支）**：`V` 经 3D conv 切成 `2×16×16` tubelet token（N=8×14×14=1568）。按 `M̄` 的 high/low-motion 两 bin、`ρ_motion=50%` 抽样可见 token，掩掉 80%（CLIP 特征目标；像素目标则 90%），得 `T^visible`（约 314）。
3. **编码**：仅 `T^visible` 进 ViT-B → `Z=Φ(T^visible)`（仅 10% token）。
4. **解码 / 预测**：`Z`+`[MASK]`+位置编码 → `Ψ_spatial` 输出 `Ĉ`（masked token 的 CLIP 特征/像素），`Ψ_motion` 输出 `M̂`（位移）。
5. **Loss**：对 masked token 算 `L_feature`(或 `L_pixel`) 与 `L_motion`，加权 `L=L_spatial+λ·L_motion`（CLIP λ=0.25），反传更新 Φ、Ψ_spatial、Ψ_motion（CoTracker3 与 CLIP 冻结）。
6. **下游**：800 epoch 后两种用法——linear probing（**冻结 Φ**、只训分类头）或 full finetuning（**不冻结 Φ**、端到端微调）；两者均丢弃空间/运动解码器。

---

## 5. 模型 / 组件

- **Backbone / 编码器**：ViT-B（主）/ ViT-S（消融）/ ViT-L（扩展），VideoMAE 风格 tubelet ViT。
- **解码器**：两个轻量 Transformer 解码器（空间 + 运动）。
- **点跟踪器**：CoTracker3（off-the-shelf，offline，grid 14×14）——生成运动目标 + 驱动 motion-aware masking。
- **特征目标提供者**：CLIP ViT-B（主）/ DINO（补充）。
- **目标函数**：`L_pixel`、`L_feature`、`L_motion`（均 masked-only L2），`L=L_spatial+λ·L_motion`。

---

## 6. 创新点

1. **显式运动重建目标**：用 CoTracker3 实拍 RGB 轨迹的**位移**作额外重建监督，把 MVM 中隐式的时序运动变成一手监督信号（区别于 MME 依赖光流+HOG、SMILE 注入合成运动）。
2. **Motion-aware tube masking**：用轨迹平均位移构造采样分布，分 high/low-motion 两 bin 按 `ρ_motion=50%` 均衡采样可见 token，使可见上下文同时覆盖动/静区域。
3. **稀疏跟踪 + 上采样技巧**：每 patch 仅跟 1 点（省算力），再空间插值上采样 `υ²` 倍模拟密集轨迹，零额外跟踪成本提升下游精度；运动目标对像素与 CLIP/DINO 语义目标均互补。

> 说明：代码未开源，第 3–5 节为论文重建，无 file:line。
