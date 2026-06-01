# ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding

[**🤗 ScaleLong Dataset**](https://huggingface.co/datasets/m-a-p/ScaleLong)

**ScaleLong** specifically engineered for the **fine-grained** assessment of the **Multi-Timescale** Capabilities of Multimodal LLMs (MLLMs) in **Long Videos**.

<div align="center">
<img src=imgs/LongVideoBench.png width=90% />
</div>



## 👀 Instruction to ScaleLong

ScaleLong is designed to assess the **Multi-Timescale** Capabilities of Multimodal LLMs (MLLMs) in **Long Videos**. By embedding questions at **four hierarchical** temporal scales (Clip, Shot, Event, and Story) within the same video content, it enables robust evaluation of MLLM performance at each distinct scale. ScaleLong includes 269 diverse videos (averaging **86 minutes**), with 8 questions per video (two per scale), across 5 major categories and 36 subcategories.

### Features

- **Multi Timescale Queries Queries.** Unlike existing benchmarks, ScaleLong structures queries at four meticulously defined temporal scales—Clip, Shot, Event, and Story—all within each individual video. Such a design enables precise evaluation of how MLLMs handle different temporal granularities while keeping the narrative context consistent.
- **Diverse Video Content and Task Design.** For comprehensive MLLM evaluation, ScaleLong offers extensive content diversity, featuring 5 main video categories (e.g., Sports, Documentaries) spanning 36 subcategories. It also incorporates 5 distinct task types (e.g., Causal Reasoning, Action Understanding) designed to probe deeper comprehension. This structured variety ensures representative assessment across diverse, real-world long-video scenarios.


## 🎞️ Representative examples from ScaleLong
Representative samples from ScaleLong. Each sample in ScaleLong comprises a video paired with carefully designed questions, structured across four hierarchical temporal scales. The correct answers are indicated in yellow.

<div align="center">
<img src=imgs/FG_LongVideoBench.png width=90% />
</div>



## 🆚 Comparion with other long video benchmarks
Comparison with other benchmarks, where the abbreviations are defined as follows: **Anno.** (Annotation Method), **A** (Automatic Annotation), **M** (Manual Annotation), **#Genres** (Number of Video Genres).  **MTS** is the abbreviation for Multi-Timescale, and **IV-MTS** is the abbreviation for Intra-Video Multi-Timescale.

<div align="center">
<img src=imgs/comparison.png width=90% />
</div>


## 🛠️ How to use ScaleLong

### 1. Installation

We have provided the complete environment configuration required for evaluating the models in the paper. For detailed installation instructions and dependency settings, please refer to the [installation.md](installation.md) file.

### 2. Download dataset from huggingface

huggingface-cli download --repo-type dataset --resume-download ScaleLong/ScaleLong --local-dir your_local_path

### 3. Model Evaluation

```python
conda activate image_video
python inference.py \
    --model_name="$MODEL_NAME" \
    --question_file="$QUESTION_FILE" \
    --model_path="$MODEL_PATH" \
    --video_dir="$VIDEO_DIR" \
    --image_dir="$IMAGE_DIR" \
    --has_image="$HAS_IMAGE" \
    --nframes="$NFRAMES" \
    --output_file="$OUTPUT_FILE"
```

One example for evaluating InternVL-2.5 can be seen in [internvl2_5.sh](scripts/internvl2_5.sh)

## 📊 Results

### Main Results

**Main.** We observe a pronounced U-shaped trend: accuracy peaks at the two extremes (Clip and Story) but dips markedly at the intermediate timescales (Shot and Event).

<div align="center">
<img src=imgs/experiment.png width=90% />
</div>

**Performance Disparities.** For the vast majority of models, Object Recognition tasks achieve the highest accuracy, whereas Counting Problems tasks incur the lowest. 

<div align="center">
<img src=imgs/experment_task_types.png width=90% />
</div>

### Ablation Study

**Q:** How does performance change as we increase the total number of visual tokens—either by sampling more frames or by raising resolution?

**A:** Under a fixed resolution, increasing the number of input frames consistently improves multi-timescale long-video understanding, with the greatest gains on Clip-level tasks.

**Q:** When the total visual‐token budget is held constant, does distributing tokens across more frames or into higher resolution yield greater gains?

**A:** Under a fixed frame count, raising resolution generally improves performance across Clip, Shot, Event, and Story tasks, but sometimes yields diminishing or even negative returns.

### Error Analysis

Although overall error rates were comparable across models, two categories—missing information and spatial replacement—stood out with the highest failure rates.

<div align="center">
<img src=imgs/wrong_design.png width=90% />
</div>
