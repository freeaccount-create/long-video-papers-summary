import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import os
import copy
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import csv

class NewHead(nn.Module):
    def __init__(self, head, init_args):
        super().__init__()
        self.dropout = nn.Dropout(p=init_args.dropout_p)
        self.head = head

        self.coarse_grained_class_index = init_args.coarse_grained_class_index
        self.num_fine_grained_class = init_args.num_fine_grained_class
        self.weight_init = init_args.weight_init

        self.fine_grained_head = self.get_fine_grained_head()

    def get_fine_grained_head(self):
        # create a new classification head for fine grained classes and init the weight
        num_features = self.head.weight.size(1)
        fine_grained_head = nn.Linear(num_features, self.num_fine_grained_class)
        if self.weight_init == 'coarse_grained_class_weight':
            weight_matrix, bias_vector = self.head.weight.data, self.head.bias.data
            w_list, b_list = [], []
            w = weight_matrix[self.coarse_grained_class_index : self.coarse_grained_class_index + 1].repeat(self.num_fine_grained_class, 1)
            b = bias_vector[self.coarse_grained_class_index : self.coarse_grained_class_index + 1].repeat(self.num_fine_grained_class)
            w_list.append(w)
            b_list.append(b)
            fine_grained_head.weight.data = torch.cat(w_list, dim=0)
            fine_grained_head.bias.data = torch.cat(b_list, dim=0)
            print("initialize weights and bias by copying coarse grained class weights and bias")
        else:
            trunc_normal_(fine_grained_head.weight, std=.02)
            nn.init.constant_(fine_grained_head.bias, 0)
            print("initialize weights and bias randomly")
        return fine_grained_head

    def forward(self, x):
        x = self.dropout(x)
        x1 = self.head(x) # original coarse grained classification head
        x2 = self.fine_grained_head(x) # new head for fine grained classes
        output = torch.cat([x1, x2], dim=1)
        return output

class Editor():
    def __init__(self, model, alg, device, output_dir, init_args):
        self.model = model
        self.alg = alg
        self.device = device
        self.coarse_grained_class_index = init_args.coarse_grained_class_index
        self.num_fine_grained_class = init_args.num_fine_grained_class
        self.output_dir = output_dir
        self.group_size = init_args.group_size
        self.dataset_config = init_args.dataset_config
        self.dataset_module = init_args.dataset_module
        # get the model after edited with the splited head
        self.edited_model = copy.deepcopy(model)
        self.edited_model.head = NewHead(list(model.children())[-1], init_args)
        self.edited_model.num_classes = model.num_classes + init_args.num_fine_grained_class
        # freeze original model
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def save(self):
        state_dict = self.edited_model.state_dict()
        state_dict["head.weight"] = torch.cat([state_dict["head.head.weight"], state_dict["head.fine_grained_head.weight"]], dim=0)
        state_dict["head.bias"] = torch.cat([state_dict["head.head.bias"], state_dict["head.fine_grained_head.bias"]])
        del state_dict["head.head.weight"]
        del state_dict["head.head.bias"]
        del state_dict["head.fine_grained_head.weight"]
        del state_dict["head.fine_grained_head.bias"]
        # Create save dictionary
        to_save = {
            "model": state_dict
        }
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir,"{}.pth".format(self.alg))
        torch.save(to_save, checkpoint_path)
        print(f"Edited model saved to {checkpoint_path}")

    def aggregate_groups(self, probs, labels=None, filenames=None):
        N, C = probs.shape
        num_groups = (N // self.group_size)

        preds = []
        gts = []
        fns = []

        for g in range(num_groups):
            s = g * self.group_size
            e = min(s + self.group_size, N)
            # aggregate prediction
            mean_probs = probs[s:e].mean(dim=0)
            pred = mean_probs.argmax().item()
            preds.append(pred)
            # label
            if not labels == None:
                group_labels = labels[s:e]
                if torch.all(group_labels == group_labels[0]):
                    gts.append(group_labels[0].item())
                else:
                    print("eval aggragation wrong!")
                    exit()
            # video name
            if not filenames == None:
                group_fns = filenames[s:e]
                if len(set(group_fns)) == 1:
                    fns.append(group_fns[0])
                else:
                    print("eval aggragation wrong!")
                    exit()

        return torch.tensor(preds), torch.tensor(gts), fns


    def evaluation(self, model, data_loader, for_generality=False):
        model.eval()
        correct = 0
        total = 0
        correct_per_class = {}
        total_per_class = {}
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                # get the logits
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                # Mask out all coarse grained class indices
                probs[:, self.coarse_grained_class_index] = 0
                probs = probs / probs.sum(dim=1, keepdim=True)
                preds, labels, _ = self.aggregate_groups(probs, labels)
                # get the prediction
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                if for_generality:
                    for pred, label in zip(preds, labels):
                        label = label.item()
                        correct_per_class[label] = correct_per_class.get(label, 0) + int(pred == label)
                        total_per_class[label] = total_per_class.get(label, 0) + 1
                    # Accumulate for confusion matrix
                    all_labels.extend(labels.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
        if for_generality:
            per_class_acc = {
                label: (correct_per_class.get(label, 0) / total_per_class[label], correct_per_class.get(label, 0), total_per_class[label])
                for label in sorted(total_per_class)
            }
            cm, label_order = self.get_cm(all_labels, all_preds)
            return correct / total, per_class_acc, cm, label_order
        else:
            return correct / total

    def get_cm(self, all_labels, all_preds, unknown_class=-1):
        valid_labels = sorted(set(all_labels))

        def map_unknown(x):
            return x if x in valid_labels else unknown_class

        mapped_labels = [map_unknown(l) for l in all_labels]
        mapped_preds = [map_unknown(p) for p in all_preds]

        label_order = valid_labels + [unknown_class]
        
        cm = confusion_matrix(mapped_labels, mapped_preds, labels=label_order)
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm = cm / row_sums  # convert to percentage
        cm = np.round(cm, 2)

        return cm, label_order

    def eval(self, equivalent_set_loader, unrelated_set_loader):
        self.edited_model.to(self.device)
        self.model.to(self.device)
        generality, generality_per_class, cm, label_order = self.evaluation(self.edited_model, equivalent_set_loader, for_generality=True)
        print(f"Generality: {generality}, Generality_per_class: {generality_per_class}")
        print(f"Confusion_matrix: {label_order}\n {cm}")
        locality = self.evaluation(self.edited_model, unrelated_set_loader) / self.evaluation(self.model, unrelated_set_loader)
        print(f"Locality: {locality}")
    
    def edit(self):
        raise NotImplementedError("Subclasses must implement this method")
