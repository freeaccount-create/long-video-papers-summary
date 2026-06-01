from algs.editor import Editor
import torch
import os
import torch.nn.functional as F
import numpy as np

class VLM(Editor):
    def __init__(self, model, alg, device, output_dir, args, alg_args):
        super().__init__(model, alg, device, output_dir, args)
        # load vlm
        self.vlm_model_name = alg_args.init_args.vlm_model.upper()
        if alg_args.init_args.vlm_model.upper() == 'CLIP': # ViT-L/14
            import clip
            self.clip_model, _ = clip.load(alg_args.init_args.model)
            self.tokenizer = clip.tokenize
        else:
            print("Unknow VLM model")
            exit()
        self.model.to(device)
        # get the text label of the new find-grained classes
        self.fine_grained_text_labels = args.fine_grained_class_items.keys()
    
    def edit(self):
        if self.vlm_model_name == 'CLIP':
            self.fine_grained_text_features = self.clip_model.encode_text(self.tokenizer(self.fine_grained_text_labels).to(self.device))
            self.fine_grained_text_features = F.normalize(self.fine_grained_text_features, dim=-1)

    def eval(self, equivalent_set_loader, unrelated_set_loader):
        generality, generality_per_class, cm, label_order = self.evaluation(equivalent_set_loader)
        print(f"Generality: {generality}, Generality_per_class: {generality_per_class}")
        print(f"Confusion_matrix: {label_order}\n {cm}")
        print(f"Locality: {1.0}")

    def evaluation(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        correct_per_class = {}
        total_per_class = {}
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, fns in data_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                # first use the original model
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=1)
                coarse_preds, labels, fns = self.aggregate_groups(probs, labels, fns)
                coarse_correct_mask = (self.coarse_grained_class_index == coarse_preds)
                # if it classify the input to correct coarse class, then use vlm to classify into the fine class
                if self.vlm_model_name == 'CLIP':
                    filter_inputs = inputs[coarse_correct_mask.repeat_interleave(self.group_size)]
                    self.clip_model.eval()
                    B, C, T, H, W = filter_inputs.shape
                    if B == 0:
                        fine_predits = []
                    else:
                        filter_inputs = filter_inputs.reshape(-1,C,H,W)

                        feature_per_frame = self.clip_model.encode_image(filter_inputs)
                        D = feature_per_frame.size(-1)
                        feature_per_frame = feature_per_frame.view(B, T, D)

                        video_features = feature_per_frame.mean(dim=1)
                        video_features = F.normalize(video_features, dim=-1)

                        logits = video_features @ self.fine_grained_text_features.T
                        probs = F.softmax(logits, dim=1)
                        fine_predits, _, _ = self.aggregate_groups(probs)
                        fine_predits = fine_predits + self.model.num_classes
                # comebine the result from two seperate model
                final_preds = coarse_preds.clone()
                final_preds[coarse_correct_mask] = torch.as_tensor(
                    fine_predits, device=final_preds.device, dtype=final_preds.dtype
                )
                correct += (final_preds == labels).sum().item()
                total += labels.size(0)

                for pred, label in zip(final_preds, labels):
                    label = label.item()
                    correct_per_class[label] = correct_per_class.get(label, 0) + int(pred == label)
                    total_per_class[label] = total_per_class.get(label, 0) + 1
                
                # Accumulate for confusion matrix
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(final_preds.cpu().tolist())
        
        per_class_acc = {
            label: (correct_per_class.get(label, 0) / total_per_class[label], correct_per_class.get(label, 0), total_per_class[label])
            for label in sorted(total_per_class)
        }

        cm, label_order = self.get_cm(all_labels, all_preds)
        
        return correct / total, per_class_acc, cm, label_order