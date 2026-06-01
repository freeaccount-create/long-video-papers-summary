# Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding
![Category-Splitting Concept Figure](https://github.com/KaitingLiu/kaitingliu.github.io/blob/main/Category-Splitting/static/images/concept_v2-3.jpg)

This is the repository for our paper *Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding* ([arXiv](https://arxiv.org/abs/2602.16545)), accepted at ICLR 2026.  
It contains code, benchmark annotations, and instructions to reproduce our experiments.

## Requirements

For running the code, download this reprository and create the environment:

```bash
conda create -n category-splitting python=3.10.18
conda activate category-splitting
```

Then install packages:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The code is tested with CUDA 12.1.1.

---

## Benchmark

To use the benchmark, first download the video data:

* <a href="https://www.qualcomm.com/developer/software/something-something-v-2-dataset" target="_blank">SSv2 video samples</a> [1]
* <a href="https://sdolivia.github.io/FineGym/" target="_blank">FineGym video samples</a> [2]

Then use our annotation files:
* <a href="https://github.com/KaitingLiu/Category-Splitting/tree/main/benchmark" target="_blank">Annotation files</a>

---

## Reproducing Results
To reproduce the results in the paper, first download the <a href="https://drive.google.com/drive/folders/16djNLmTKiTQ-Rn5_f1h1uAjTECNWRKsl?usp=sharing" target="_blank">mixed-granularity base models</a>. These models are used as the starting point for all category splitting experiments.

Before running the scripts, place the downloaded files in the following directories:

Video data:
* ./video/finegym/ for FineGym videos
* ./video/ssv2/ for SSv2 videos

Mixed-granularity base models:
* ./checkpoints/ for all checkpoint files

Due to computational resource constraints, we resize the original FineGym videos from 1920×1080 to 480×270. To reproduce our results, please apply the same resizing to the videos.

### Comparative Zero-Shot Results (Table 2)

To reproduce the category splitting evaluation results of our method (modifier alignment) for the four benchmarks (SSv2-Split-A, SSv2-Split-B, FineGym-Split-A, and FineGym-Split-B), we first run these four scripts. Each script evaluates the splitting of all corresponding coarse-grained categories in the benchmark with three random seeds.

Run the scripts:
```bash
./scripts/Table2-SSv2-Split-A.sh
./scripts/Table2-SSv2-Split-B.sh
./scripts/Table2-FineGym-Split-A.sh
./scripts/Table2-FineGym-Split-B.sh
```

After all runs are completed, compute the average results across all coarse category split targets and three seeds for each benchmark using the following command:

```bash
python summery.py ./output/Table2/SSv2-Split-A/ma
python summery.py ./output/Table2/SSv2-Split-B/ma
python summery.py ./output/Table2/FineGym-Split-A/ma
python summery.py ./output/Table2/FineGym-Split-B/ma
```

### Zero-Shot Ablation (Table 3)

To reproduce the category splitting evaluation results for the three methods (modifier alignment, modifier retrieval, and VLM), run the following script. The script evaluates the splitting of all coarse-grained categories in the SSv2-Split-A benchmark. For modifier alignment, run three rounds with different random seeds. For modifier retrieval and VLM, only one run is needed since there is no stochasticity.

Run the script:

```bash
./scripts/Table3-SSv2-Split-A.sh
```

After all runs are completed, compute the average results across all coarse category split targets (and three seeds for modifier alignment) using:

```bash
python summery.py ./output/Table3/SSv2-Split-A/vlm
python summery.py ./output/Table3/SSv2-Split-A/mr
python summery.py ./output/Table3/SSv2-Split-A/ma
```

### One-Shot Finetuning Ablation (Table 4)

To reproduce the category splitting evaluation results for one-shot finetuning with newly added head initialized using different methods (corresponding to the last three rows in Table 4), run the following script. The script evaluates the splitting of all coarse-grained categories in the SSv2-Split-A benchmark, with six rounds in total.  

Run the script:

```bash
./scripts/Table4-SSv2-Split-A.sh
```

After all runs are completed, compute the average results across all coarse category split targets and all seeds using:

```bash
python summery.py ./output/Table4/SSv2-Split-A/ft_random
python summery.py ./output/Table4/SSv2-Split-A/ft_coarse_grained_class_weight
python summery.py ./output/Table4/SSv2-Split-A/ft_ma
```

**Note:** All scripts for reproducing (`Table2-*.sh`, `Table3-*.sh`, `Table4-*.sh`) internally call `job.sh`, which is a SLURM submission script configured for our cluster. You can edit `job.sh` to match your own system if needed before running the scripts.

---

## Citation

If you use this repository, please cite our paper.

```bibtex
@article{Liu2026Let,
  title={Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding},
  author={Liu, Kaiting and Doughty, Hazel},
  journal={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://kaitingliu.github.io/Category-Splitting/}
}
```

---
## Reference
[1] Goyal, Raghav, et al. "The" something something" video database for learning and evaluating visual common sense." Proceedings of the IEEE international conference on computer vision. 2017.

[2] Shao, Dian, et al. "Finegym: A hierarchical video dataset for fine-grained action understanding." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
