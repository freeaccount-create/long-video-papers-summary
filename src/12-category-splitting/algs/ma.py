from algs.editor import Editor
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import json
import csv
import copy
import numpy as np

class MA(Editor):
    def __init__(self, model, alg, device, output_dir, args, alg_args):
        super().__init__(model, alg, device, output_dir, args)
        # load the existing modifier knowledge in the original base model
        with open(args.modifiers_in_base_model, 'r') as f:
            self.modifiers_in_base_model = json.load(f)
        # load the text model
        self.text_model_name = alg_args.init_args.text_model.upper()
        if alg_args.init_args.text_model.upper() == 'CLIP':
            import clip
            self.text_model, _ = clip.load(alg_args.init_args.model, device=device)
            self.len_text_embed = self.text_model.text_projection.shape[1]
            self.tokenizer = clip.tokenize
        else:
            print("Unknow text encoder")
            exit()
        # load the args for training the alignment module
        self.train_args = alg_args.train_args
        # get the modifier for the new fine-grained classes
        self.fine_grained_class_modifier_texts = [value["modifier"] for value in args.fine_grained_class_items.values()]
        # this is the mapping function
        self.mapping_function = self.build_mapping_function(alg_args.init_args.mapping_function_hidden_layer_dims, device)
        self.train_mapping()

    
    def build_mapping_function(self, layer_dims, device):
        layer_dims.insert(0, self.len_text_embed)
        layer_dims.append(self.edited_model.head.head.weight.shape[1])
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.GELU())
        return nn.Sequential(*layers).to(device)

    def get_modifier_dict(self, modifiers_in_base_model):
        texts = []
        vectors = []
        for coarse_grained_label, fine_grained_labels in modifiers_in_base_model.items():
            index_list = [v["index"] for v in fine_grained_labels.values()]
            if len(index_list) == 1:
                texts.append(coarse_grained_label)
                vectors.append(self.edited_model.head.head.weight[index_list[0]].detach())
            else:
                coarse_grained_prototype = torch.mean(self.edited_model.head.head.weight[index_list], dim=0, keepdim=False).detach()
                texts.append(coarse_grained_label)
                vectors.append(coarse_grained_prototype)
                
                for key, value in fine_grained_labels.items():
                    texts.append(key)
                    vectors.append(self.edited_model.head.head.weight[value["index"]].detach())

                    modifier_knowledges = self.edited_model.head.head.weight[value["index"]].detach() - coarse_grained_prototype
                    texts.append(value["modifier"])
                    vectors.append(modifier_knowledges)
        if self.text_model_name == 'CLIP':
            text_embeds = self.text_model.encode_text(self.tokenizer(texts).to(self.device)).detach()
        return text_embeds.float(), torch.stack(vectors)

    def train_mapping(self):
        text_embeds, vectors = self.get_modifier_dict(self.modifiers_in_base_model)

        dataset = TensorDataset(text_embeds, vectors)
        dataloader = DataLoader(dataset, batch_size=self.train_args.batch_size, shuffle=True)
        
        # optimizer and learning rate scheduler and criterion
        optimizer = AdamW(self.mapping_function.parameters(), lr=self.train_args.lr, weight_decay=self.train_args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.train_args.epochs, eta_min=self.train_args.min_lr)
        criterion = nn.MSELoss()
        
        # start training the mapping function
        best_ema_cos = -float('inf')
        ema_cos = None
        patience_counter = 0
        best_state_dict = None
        for epoch in range(self.train_args.epochs):
            self.mapping_function.train()
            epoch_loss = 0.0
            epoch_cos = 0.0
            total_batches = 0
            for text_embeds, vectors in dataloader:
                text_embeds, vectors = text_embeds.to(self.device), vectors.to(self.device)
                
                optimizer.zero_grad()
                output = self.mapping_function(text_embeds)
                loss = criterion(output, vectors)
                loss.backward()
                optimizer.step()

                # Print progress
                epoch_loss += loss.item()
                total_batches += 1

                with torch.no_grad():
                    cos_sim = F.cosine_similarity(output, vectors, dim=1).mean().item()
                    epoch_cos += cos_sim

            
            avg_loss = epoch_loss / total_batches
            avg_cos = epoch_cos / total_batches
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{self.train_args.epochs}, Loss: {avg_loss:.10f}, Cos: {avg_cos:.10f}, LR: {lr:.10f}")

            if self.train_args.early_stop:
                ema_cos = avg_cos if ema_cos is None else self.train_args.ema_beta * ema_cos + (1 - self.train_args.ema_beta) * avg_cos
                print(f"Epoch {epoch+1}/{self.train_args.epochs}, COS_EMA: {ema_cos:.10f}")
                improved = (ema_cos - best_ema_cos) > self.train_args.ema_min_delta
                if improved:
                    best_ema_cos = ema_cos
                    patience_counter = 0
                    best_state_dict = copy.deepcopy(self.mapping_function.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.train_args.ema_patience:
                        self.mapping_function.load_state_dict(best_state_dict)
                        print(f"[EarlyStop] epoch {epoch+1}: loss EMA plateau.")
                        break
            scheduler.step()

    def edit(self):
        if self.text_model_name == 'CLIP':
            modifier_text_embeds = self.text_model.encode_text(self.tokenizer(self.fine_grained_class_modifier_texts).to(self.device)).to(dtype=self.edited_model.head.fine_grained_head.weight.dtype).detach()
        modifier_vectors = self.mapping_function(modifier_text_embeds).detach().cpu().numpy()       
        for i, vector in enumerate(modifier_vectors):
            self.edited_model.head.fine_grained_head.weight.data[i] += torch.tensor(vector)