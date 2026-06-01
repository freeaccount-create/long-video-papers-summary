import json
import argparse
import importlib
import os
import torch
from timm.models import create_model
from datasets.datasets import get_dataloader
from box import Box
import random
import numpy as np
import json
import numpy as np
from torch.utils.data import DataLoader

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_model(model_config, checkpoint):
    # load the trained classifier.
    importlib.import_module(f"models.{model_config.class_name}")
    model = create_model(
        model_config.init_args.model,
        num_classes=model_config.init_args.num_classes,
        img_size=model_config.init_args.img_size,
        all_frames=model_config.init_args.num_frames,
        tubelet_size=model_config.init_args.tubelet_size,
        use_cls_token=model_config.init_args.use_cls_token,
        use_mean_pooling=model_config.init_args.use_mean_pooling
    )
    # load checkpoint
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint}' not found.")
    print("Load trained calssifier from checkpoint {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location='cpu')
    load_success = False
    for model_key in model_config.model_key.split('|'):
        if model_key in checkpoint:
            model.load_state_dict(checkpoint[model_key])
            load_success = True
            print("Load state_dict by model_key = %s" % model_key)
            break
    if not load_success:
        raise RuntimeError("Failed to load model: No valid model_key found in checkpoint.")
    else:
        print("Model = %s" % str(model))
    return model

def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    """
    argument for runing the program and set seed if needy.
    """
    parser = argparse.ArgumentParser(description="This program aim for editing the trained classification head with limited data or even no fine-grained data to split coarse grained class into finer classes.")
    parser.add_argument("--config_dir", type=str, default="./configs", help="dir for saving config file about model, dataset and alg.")
    # about base model
    parser.add_argument("--model_config", type=str, help="the config file for model.")
    parser.add_argument("--checkpoint", type=str, help="model's checkpoint path.")
    # about benchmark dataset
    parser.add_argument("--dataset", type=str, default="ssv2", help="dataset name. option: ssv2, finegym.")
    parser.add_argument("--label_dir", type=str, help="dir of category splitting benchmark annotation file")
    parser.add_argument("--video_dir", type=str, help="dir of category splitting benchmark video samples")
    # about algorithm
    parser.add_argument("--alg", type=str, default="ma", help="algorithm name. option: ft, mr, ma, vlm.") # mr for modifier retrival, ma for modifier alignment.
    parser.add_argument("--weight_init", type=str, default="coarse_grained_class_weight", help="method to initialize the newly added weights. options: coarse_grained_class_weight, random, ma.")
    parser.add_argument("--coarse_grained_text_label", type=str, default="", help="text label of the coarse class which will be splited later")
    parser.add_argument("--modifiers_in_base_model", type=str, help="file about the modifiers in the original base model.")
    parser.add_argument("--modifiers_for_new_classes", type=str, help="file about the modifiers for new fine-grained classes.")
    parser.add_argument("--dropout_p", type=float, default=0.25, help="dropout rate before classification head and after backbone")
    # other
    parser.add_argument("--device", type=str, default="cuda", help="device.")
    parser.add_argument("--output_dir",  type=str, default="./outputs", help="dir for saving all the output file.")
    parser.add_argument('--enable_seed', action='store_true', help="enable fixed random seed")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    # args and set seed if needy
    args = parser.parse_args()
    if args.enable_seed:
        set_all_seeds(args.seed)

    """
    make output folder if not exist,
    get all the config and corresponding module,
    check if everything is ready for the program.
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # model config 
    if not os.path.exists(args.model_config):
        raise ValueError("Invalid model config. Config file not found.")
    model_config = Box(load_json(args.model_config))

    # alg config
    alg_config_path = os.path.join(args.config_dir, "algs", f"{args.alg}.json")
    if not os.path.exists(alg_config_path):
        raise ValueError("Invalid algorithm name. Config file not found.")
    alg_config = Box(load_json(alg_config_path))
    
    # dataset config
    dataset_config_path = os.path.join(args.config_dir, "datasets", f"{args.dataset}.json")
    if not os.path.exists(dataset_config_path):
        raise ValueError("Invalid dataset name. Config file not found.")
    dataset_config = Box(load_json(dataset_config_path))
    dataset_config.video_dir = args.video_dir
    
    # we first load the model
    model = get_model(model_config, args.checkpoint)
    # load the dataset module for making data loader later
    dataset_module = importlib.import_module(f"datasets.{dataset_config.class_name}")
    #load the algorithm for editing the classification head to split the coarse grained class into finer classes
    alg_module = importlib.import_module(f"algs.{alg_config.class_name}")
    alg_class = getattr(alg_module, alg_config.class_name.upper())
    
    """
    Initialization of the algorithm and add the extra head to the base model.
    """
    # load the text label and indexes from the original base model
    base_model_label = os.path.join(args.label_dir, "labels.json")
    with open(base_model_label, 'r') as f:
        base_model_label = json.load(f)
    # get the index of the coarse grained class that will be split later
    args.coarse_grained_class_index = base_model_label[args.coarse_grained_text_label]
    # load the classes hierachy of the benchmark
    with open(args.modifiers_for_new_classes, 'r') as f:
        group_scheme_with_modifiers = json.load(f)["group_scheme"]
    # get the number of newly finegrianed classes for this coarse grained class
    args.num_fine_grained_class = len(group_scheme_with_modifiers[args.coarse_grained_text_label])
    # and get text label of each newly finegrianed classes in this coarse grained class
    args.fine_grained_class_items = group_scheme_with_modifiers[args.coarse_grained_text_label]
    # get the number of crops for each video when evaluating a video
    args.group_size = dataset_config.eval_num_segments * dataset_config.eval_num_crops
    # set params
    args.dataset_config = dataset_config
    args.dataset_module = dataset_module
    # initilize the editor
    editor = alg_class(model, alg_config.class_name, args.device, args.output_dir, args, alg_config)
    print(f"Loaded Algorithm: {alg_class.__name__}")
    # print out the param
    print(model_config, dataset_config, alg_config)


    """
    start edit the model.
    """
    if alg_config.zero_shot: # branch for mr, ma and vlm
        editor.edit()
    else: # branch for ft
        dataset_config.shot = alg_config.train_args.shot # number of sample use for finetuning
        dataset_config.val_ratio = alg_config.train_args.val_ratio # ratio between val samples and train samples, only have val_dataset when train with full dataset.
        dataset_config.batch_size = alg_config.train_args.batch_size # training batch size
        ft_set_path = os.path.join(args.label_dir, args.coarse_grained_text_label, "ft_set.csv")
        ft_set_loader, val_dataset_loader = get_dataloader(dataset_module, 'train', ft_set_path, dataset_config)
        # if use modifier alignment as the initilization method
        if args.weight_init == 'ma':
            class_name = args.weight_init
            config_path = os.path.join("./configs", "algs", f"{class_name}.json")
            config = Box(load_json(config_path))
            module = importlib.import_module(f"algs.{class_name}")
            alg_class = getattr(module, class_name.upper())
            initializer = alg_class(model, class_name, args.device, args.output_dir, args, config)
            initializer.edit()
            editor.edited_model = initializer.edited_model
        # now edit (ft) the weight using annotated fine-grained samples
        editor.edit(ft_set_loader, val_dataset_loader)

    """
    eval the model after editing.
    """
    equivalent_set_path = os.path.join(args.label_dir, args.coarse_grained_text_label, "equivalent_set.csv") # for generality
    equivalent_set_loader = get_dataloader(dataset_module, 'eval', equivalent_set_path, dataset_config)
    unrelated_set_path = os.path.join(args.label_dir, args.coarse_grained_text_label, "unrelated_set.csv") # for locality
    unrelated_set_loader = get_dataloader(dataset_module, 'eval', unrelated_set_path, dataset_config)
    print(f"Split target: {args.coarse_grained_text_label}_seed{args.seed}")
    editor.eval(equivalent_set_loader, unrelated_set_loader)

if __name__ == "__main__":
    main()
