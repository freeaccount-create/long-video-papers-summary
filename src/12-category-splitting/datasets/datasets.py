import importlib
import torch
from torch.utils.data import Dataset, DataLoader, default_collate 
import os
import pandas as pd
import random
import json
import numpy as np
from collections import defaultdict
import copy

def custom_collate_fn(batch):
    # batch is a list of tuples: [([tensor1, tensor2], [label1, label2]), ...]
    all_tensors = []
    all_labels = []
    for tensors, labels in batch:
        all_tensors.extend(tensors)
        all_labels.extend(labels)
    # Convert to batched tensor if all tensors have same shape
    all_tensors = torch.stack(all_tensors)
    all_labels = torch.tensor(all_labels)
    return all_tensors, all_labels

def custom_collate_fn_eval(batch):
    # batch is a list of tuples: [([tensor1, tensor2], [label1, label2]), ...]
    all_tensors = []
    all_labels = []
    all_fns = []
    for tensors, labels, fns in batch:
        all_tensors.extend(tensors)
        all_labels.extend(labels)
        all_fns.extend(fns)
    # Convert to batched tensor if all tensors have same shape
    all_tensors = torch.stack(all_tensors)
    all_labels = torch.tensor(all_labels)
    return all_tensors, all_labels, all_fns

def apply_k_shot_sampling(labels, filenames, k):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    selected_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) < k:
            print(f"Skipping k-shot sampling: class {label} has only {len(indices)} samples (needs {k})")
            return labels, filenames  # return full dataset if any class is too small
        selected_indices.extend(random.sample(indices, k))
    # Return subset
    new_labels = [labels[i] for i in selected_indices]
    new_filenames = [filenames[i] for i in selected_indices]
    return new_labels, new_filenames

def stratified_split_filenames_labels(filenames, labels, val_ratio=0.1):
    cls2idx = defaultdict(list)
    for i, y in enumerate(labels):
        cls2idx[y].append(i)

    train_idx_all = []
    val_idx_all = []

    for y, idxs in cls2idx.items():
        random.shuffle(idxs)
        n_val = int(np.ceil(len(idxs) * val_ratio))

        val_idx = idxs[:n_val]
        trn_idx = idxs[n_val:]

        val_idx_all.extend(val_idx)
        train_idx_all.extend(trn_idx)

    # 根据索引组装输出
    train_filenames = [filenames[i] for i in train_idx_all]
    train_labels    = [labels[i]    for i in train_idx_all]
    val_filenames   = [filenames[i] for i in val_idx_all]
    val_labels      = [labels[i]    for i in val_idx_all]

    return train_filenames, train_labels, val_filenames, val_labels

def get_dataloader(dataset_module, mode, annotation_path, dataset_config):
    # read the label and video data path according to different mode
    if not os.path.isfile(annotation_path) or not os.path.isdir(dataset_config.video_dir):
        raise ValueError("Missing or invalid required arg: label_path or video_dir. Please ensure both exist.")
    df = pd.read_csv(annotation_path, header=None, delimiter=",")
    filenames = df[0].tolist()
    labels = df[1].tolist()
    filenames = [os.path.join(dataset_config.video_dir, f) for f in filenames]
    if mode == 'train' and dataset_config.shot == 'all':
        train_filenames, train_labels, val_filenames, val_labels = stratified_split_filenames_labels(filenames, labels, val_ratio=dataset_config.val_ratio)
        dataset = dataset_module.build_dataset(mode, train_filenames, train_labels, dataset_config)
        dataset_loader = DataLoader(
            dataset,
            batch_size=dataset_config.batch_size,
            num_workers=dataset_config.num_workers,
            pin_memory=dataset_config.pin_mem,
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        val_dataset = dataset_module.build_dataset('val', val_filenames, val_labels, dataset_config)
        val_dataset_loader = DataLoader(
            dataset,
            batch_size=dataset_config.batch_size,
            num_workers=dataset_config.num_workers,
            pin_memory=dataset_config.pin_mem,
            collate_fn=custom_collate_fn,
            shuffle=False
        )
        return dataset_loader, val_dataset_loader
    elif mode == 'train':
        labels, filenames = apply_k_shot_sampling(labels, filenames, k=dataset_config.shot)
        dataset = dataset_module.build_dataset(mode, filenames, labels, dataset_config)
        dataset_loader = DataLoader(
            dataset,
            batch_size=dataset_config.batch_size,
            num_workers=dataset_config.num_workers,
            pin_memory=dataset_config.pin_mem,
            collate_fn=custom_collate_fn,
            shuffle=True
        )
        return dataset_loader, None
    elif mode == 'eval': 
        dataset = dataset_module.build_dataset(mode, filenames, labels, dataset_config)
        dataset_loader = DataLoader(
            dataset,
            batch_size=dataset_config.eval_batch_size,
            num_workers=dataset_config.num_workers,
            pin_memory=dataset_config.pin_mem,
            collate_fn=custom_collate_fn_eval,
            shuffle=False
        )
        return dataset_loader
    else:
        raise ValueError("Unkonw dataloader mode.")