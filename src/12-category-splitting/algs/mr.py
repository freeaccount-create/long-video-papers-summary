from algs.editor import Editor
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import csv
import json
import numpy as np

# this method utilize the semantically similarity referenced modifier knowledge to edit and split the coarse class
class MR(Editor):
    def __init__(self, model, alg, device, output_dir, args, alg_args):
        super().__init__(model, alg, device, output_dir, args)
        # load the existing modifier knowledge
        with open(args.modifiers_in_base_model, 'r') as f:
            self.modifiers_in_base_model = json.load(f)
        # load the text model
        self.text_model_name = alg_args.init_args.text_model.upper()
        if alg_args.init_args.text_model.upper() == 'CLIP':
            import clip
            self.text_model, _ = clip.load(alg_args.init_args.model, device=device)
            self.tokenizer = clip.tokenize
        else:
            print("Unknow text encoder")
            exit()
        # set the hyperparams
        self.alpha = alg_args.init_args.alpha
        self.beta = alg_args.init_args.beta
        # get the text label and modifier text for new fine-grained classes
        self.fine_grained_class_items = args.fine_grained_class_items

    def get_knowledge_dict(self, modifiers_in_base_model):
        fine_grained_texts = []
        modifier_texts = []
        modifier_vectors = []
        for coarse_grained_label, fine_grained_labels in modifiers_in_base_model.items():
            index_list = [v["index"] for v in fine_grained_labels.values()]
            if len(index_list) == 1:
                continue
            coarse_grained_prototype = torch.mean(self.edited_model.head.head.weight[index_list], dim=0, keepdim=True).detach()
            for key, value in fine_grained_labels.items():
                vector = self.edited_model.head.head.weight[value["index"]].detach() - coarse_grained_prototype
                fine_grained_texts.append(key)
                modifier_texts.append(value["modifier"])
                modifier_vectors.append(vector) 
        if self.text_model_name == 'CLIP':
            fine_grained_embeds = F.normalize(self.text_model.encode_text(self.tokenizer(fine_grained_texts).to(self.device)).detach(), dim=-1)
            modifier_embeds = F.normalize(self.text_model.encode_text(self.tokenizer(modifier_texts).to(self.device)).detach(), dim=-1)
        return fine_grained_texts, modifier_texts, fine_grained_embeds, modifier_embeds, modifier_vectors

    def edit(self):
        fine_grained_texts, modifier_texts, fine_grained_embeds, modifier_embeds, modifier_vectors = self.get_knowledge_dict(self.modifiers_in_base_model)
        target_modifier_texts = [value["modifier"] for value in self.fine_grained_class_items.values()]
        target_fine_grained_texts = [text for text in self.fine_grained_class_items.keys()]
        if self.text_model_name == 'CLIP':
            target_modifier_embeds = F.normalize(self.text_model.encode_text(self.tokenizer(target_modifier_texts).to(self.device)).detach(), dim=-1)
            target_fine_grained_embeds = F.normalize(self.text_model.encode_text(self.tokenizer(target_fine_grained_texts).to(self.device)).detach(), dim=-1)

        sim_modifier = target_modifier_embeds @ modifier_embeds.T 
        sim_fine_grained = target_fine_grained_embeds @ fine_grained_embeds.T
        sim_total = self.alpha * sim_modifier + self.beta * sim_fine_grained

        for t in range(self.num_fine_grained_class):
            max_val = sim_total.max()
            fine_grained_class_index, knowledge_index = (sim_total == max_val).nonzero(as_tuple=False)[0]
            
            print(max_val, "|", fine_grained_texts[knowledge_index], "|", target_fine_grained_texts[fine_grained_class_index], "|", modifier_texts[knowledge_index], "|", target_modifier_texts[fine_grained_class_index])
            self.edited_model.head.fine_grained_head.weight.data[fine_grained_class_index] += modifier_vectors[knowledge_index].view(-1)
            
            sim_total[fine_grained_class_index, :] = -float('inf')
            sim_total[:, knowledge_index] = -float('inf')
            