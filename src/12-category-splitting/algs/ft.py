from algs.editor import Editor
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torch.utils.data import Subset
from collections import defaultdict
import random
import copy

class FT(Editor):
    def __init__(self, model, alg, device, output_dir, args, alg_args):
        super().__init__(model, alg, device, output_dir, args)
        # freeze model's backbone and old head for following training
        self.freeze_backbone = alg_args.init_args.freeze_backbone
        self.freeze_old_head = alg_args.init_args.freeze_old_head
        if self.freeze_backbone and self.freeze_old_head:
            for name, param in self.edited_model.named_parameters():
                if "fine_grained_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.freeze_backbone:
            for name, param in self.edited_model.named_parameters():
                if "head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in self.edited_model.named_parameters():
                param.requires_grad = True
        for name, param in self.edited_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        self.train_args = alg_args.train_args

    def edit(self, train_set_loader, val_dataset_loader):
        # optimizer and learning rate scheduler and criterion
        if self.freeze_backbone and self.freeze_old_head:
            optimizer = AdamW(self.edited_model.head.parameters(), lr=self.train_args.lr, weight_decay=self.train_args.weight_decay)
        elif self.freeze_backbone:
            optimizer = AdamW(self.edited_model.head.parameters(), lr=self.train_args.lr, weight_decay=self.train_args.weight_decay)
        else:
            optimizer = AdamW(self.edited_model.parameters(), lr=self.train_args.lr, weight_decay=self.train_args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.train_args.epochs, eta_min=self.train_args.min_lr)
        criterion = nn.CrossEntropyLoss()
        # prepare model
        self.edited_model.to(self.device)
        self.edited_model.train()
        # start edit (finetune)
        best_ema_loss = float('inf')
        ema_loss = None
        patience_counter = 0
        best_state_dict = None
        for epoch in range(self.train_args.epochs):
            self.edited_model.train()
            epoch_loss = 0.0
            total_batches = 0
            for inputs, labels in train_set_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                logits = self.edited_model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1
            
            avg_loss = epoch_loss / total_batches
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{self.train_args.epochs}, Train Loss: {avg_loss:.4f}, LR: {lr:.6f}")
            if self.train_args.early_stop:
                if val_dataset_loader:
                    self.edited_model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for v_in, v_lab in val_dataset_loader:
                            v_in, v_lab = v_in.to(self.device), v_lab.to(self.device)
                            v_logits = self.edited_model(v_in)
                            v_l = criterion(v_logits, v_lab).item()
                            val_loss += v_l
                            val_batches += 1
                    val_loss /= max(1, val_batches)
                    avg_loss = val_loss
                
                ema_loss = avg_loss if ema_loss is None else self.train_args.ema_beta * ema_loss + (1 - self.train_args.ema_beta) * avg_loss
                print(f"Epoch {epoch+1}/{self.train_args.epochs}, Train/Val Loss: {avg_loss:.4f}, EMA Loss={ema_loss:.4f}")
                
                improved = (best_ema_loss - ema_loss) > self.train_args.ema_min_delta
                if improved:
                    best_ema_loss = ema_loss
                    patience_counter = 0
                    if val_dataset_loader:
                        best_state_dict = copy.deepcopy(self.edited_model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >=  self.train_args.ema_patience:
                        print(f"[EarlyStop] epoch {epoch+1}: loss EMA plateau.")
                        if best_state_dict:
                            self.edited_model.load_state_dict(best_state_dict)
                            print("Restored best val model before stopping.")
                        break
                        
            scheduler.step()      
