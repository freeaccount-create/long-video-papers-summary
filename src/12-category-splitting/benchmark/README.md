# Category Splitting Benchmark

This repository provides the **label files and grouping schemes** for the category splitting benchmarks introduced in our paper *Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding*.

**Note:** This repository only provide the annotation files, we do not redistribute raw videos. Please download the video samples from the official sources:

* <a href="https://www.qualcomm.com/developer/software/something-something-v-2-dataset" target="_blank">SSv2 video samples</a>
* <a href="https://sdolivia.github.io/FineGym/" target="_blank">FineGym video samples</a>

---

## Datasets Overview
We constructed our category splitting benchmark **SSv2-Split** and **FineGym-Split** based on two existing datasets (Something-Something V2 dataset [1] and FineGym dataset [2]).

  For original FineGym, some videos are missing and cannot be downloaded from the web. These missing samples are therefore not included in our dataset. What's more, in the original FineGym288 labels ([original labels](https://sdolivia.github.io/FineGym/resources/dataset/gym288_categories.txt)), there are three groups of text labels that are exactly duplicated, differing only by their index:

  - `(BB) salto backward tucked with 1 twist`: Clabels 180 and 216  
  - `(BB) salto backward tucked`: Clabels 181 and 217  
  - `(BB) salto backward stretched with 1 twist`: Clabels 185 and 222  

  We kept only the samples with Clabels 216, 217, and 222, and removed the others. Our dataset was then constructed based on the remaining samples.
  
We then take SSv2-Split as an example below to describe how we construct it; FineGym-Split follows the same construction.

We reorganize the labels of the SSv2 dataset into a **coarse-to-fine structure**:

- **Coarse categories**: each coarse category is formed by grouping a set of semantically related fine-grained action labels  
- **Fine-grained categories**: original action labels  

All fine-grained categories are assigned to exactly one coarse category.

We then partition the **coarse categories** into two complementary subsets, A and B. Based on this partition, we construct two **mixed-granularity datasets**:

- **Dataset A**: fine-grained categories within coarse categories in subset A are **collapsed into coarse labels** (to be split), while those in subset B remain **fine-grained**
- **Dataset B**: fine-grained categories within coarse categories in subset B are **collapsed into coarse labels** (to be split), while those in subset A remain **fine-grained**

---

## File Structure

```
Dataset/
├── SSv2-Split/
│   ├── group_scheme.json
│   ├── A/
│   │   ├── label.json
│   │   ├── train.csv / val.csv
│   │   ├── <coarse_label>/
│   │   │   ├── ft_set.csv
│   │   │   ├── equivalent_set.csv
│   │   │   ├── unrelated_set.csv
│   ├── B/
│   │   ├── (same structure as A)
│
├── FineGym-Split/
│   ├── (same structure as SSv2-Split)
```

---

## Grouping Scheme

File: `group_scheme.json`

```json
{
  "group_scheme": {
    "coarse_label_1": ["fine_grained_label_1", "fine_grained_label_2", ...],
    "coarse_label_2": ["fine_grained_label_3", "fine_grained_label_4", ...]
  },
  "A": ["coarse_label_1", "coarse_label_2", ...],
  "B": [...]
}
```

* `group_scheme`: mapping from **coarse → fine grained labels**
* `A`, `B`: partition of **coarse categories** into two complementary subsets, determining which categories are collapsed into coarse labels (splitting targets) in Dataset A and Dataset B

---

## Mixed-Granularity Datasets

Each dataset (`A/`, `B/`) contains:

File: `label.json`

Mapping from **text label → index**

File: `train.csv / val.csv` has format:

```
video_path,label_index
```

They are used to train and evaluate a **mixed-granularity base model**, where coarse and fine-grained labels coexist. This base model serves as the starting point for category splitting task.

We also provide **mixed-granularity base model** checkpoints, fine-tuned from the MVD ViT-Small model [3] (<a href="https://drive.google.com/file/d/1HqvGxx7_JYO5JKvRT0giesl-p-Iaaesa/view" target="_blank">mvd_s_from_l_ckpt_399.pth</a>), at: <a href="https://drive.google.com/drive/folders/16djNLmTKiTQ-Rn5_f1h1uAjTECNWRKsl?usp=sharing" target="_blank">mixed-granularity base models</a>

---

## Category Splitting Evaluation

Each dataset (`A/`, `B/`) contains a folder for each coarse category, which is the splitting target, used for category splitting evaluation:

```
<coarse_label>/
├── ft_set.csv
├── equivalent_set.csv
├── unrelated_set.csv
```

### Files

All files (`ft_set.csv`, `equivalent_set.csv`, `unrelated_set.csv`) follow the format:

```
video_path,label_index
```

- **`ft_set.csv`**  
  - Training samples from the coarse category corresponding to the folder name, derived from `train.csv`, labeled with fine-grained category indices
  - Used for category splitting methods that require fine-grained supervision

- **`equivalent_set.csv`**  
  - Validation samples from the same coarse category, derived from `val.csv`, labeled with fine-grained category indices
  - Used to evaluate **generality** of the category splitting method  

- **`unrelated_set.csv`**  
  - Validation samples from all the untouched categories, derived from `val.csv`, with their original indices in `val.csv`
  - Used to evaluate **locality** of the category splitting method

Fine-grained category indices in these files extend the label space defined in `label.json`, starting from the next available index (i.e., if the last index in `label.json` is N, these indices start from N+1)

---

## Citation

If you use this dataset, please cite our paper.

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

[3] Wang, Rui, et al. "Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
