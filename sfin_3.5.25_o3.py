#!/usr/bin/env python
"""
AdvancedSFIN.py

A streamlined implementation of a complex-valued hierarchical attention model.
Key updates include:
  • Native complex tensor support for embeddings & positional encodings.
  • Renamed and simplified complex attention layers.
  • Adaptive hierarchical processing with skip connections.
  • A refined memory module.
  • Improved checkpointing and explainability utilities.
  
Usage modes: train, evaluate, generate, explain.
"""

import os
import math
import time
import random
import json
import logging
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import entropy

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("advanced_sfin_training.log")]
)
logger = logging.getLogger("AdvancedSFIN")
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global configuration
config = {
    "use_tensorboard": True,
    "save_attention_heatmaps": True,
    "plot_gradient_flow": True,
    "explainability_mode": "all"
}
if config["use_tensorboard"]:
    writer = SummaryWriter("runs/advanced_sfin_experiment")

###############################################################################
#                      COMPLEX-VALUED MODULES                                 #
###############################################################################

class ComplexEmbedding(nn.Module):
    """
    Complex-valued embedding using PyTorch’s complex dtype.
    """
    def __init__(self, vocab_size, dim, scale=0.02):
        super().__init__()
        self.dim = dim
        self.embedding_real = nn.Embedding(vocab_size, dim)
        self.embedding_imag = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding_real.weight, mean=0.0, std=scale)
        nn.init.normal_(self.embedding_imag.weight, mean=0.0, std=scale)
        
    def forward(self, x):
        real = self.embedding_real(x)
        imag = self.embedding_imag(x)
        # Return a complex tensor
        return torch.complex(real, imag)

class ComplexPositionalEncoding(nn.Module):
    """
    Learnable complex-valued positional encoding.
    Computes encoding as exp(i * position * freq), added to input.
    """
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        # Create a frequency vector (learnable)
        self.freq = nn.Parameter(torch.linspace(0, 1, steps=dim), requires_grad=True)
        self.max_len = max_len

    def forward(self, x):
        # x: [batch, seq_len, dim] complex
        batch, seq_len, dim = x.shape
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)  # [seq_len, 1]
        angle = positions * self.freq.unsqueeze(0)  # [seq_len, dim]
        # Compute complex encoding: exp(i * angle) = cos(angle) + i*sin(angle)
        pe = torch.complex(torch.cos(angle), torch.sin(angle))
        pe = pe.unsqueeze(0).expand(batch, -1, -1)
        return x + pe

class ComplexLayerNorm(nn.Module):
    """
    Complex Layer Normalization: applies LayerNorm separately on real and imaginary parts.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
        
    def forward(self, x):
        return torch.complex(self.ln(x.real), self.ln(x.imag))

###############################################################################
#                     ATTENTION & HIERARCHICAL MODULES                         #
###############################################################################

class ComplexAttentionLayer(nn.Module):
    """
    Simplified complex-valued self-attention layer.
    Uses complex linear projections and computes scaled dot-product attention
    based on the real part of the inner product.
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "Dimension must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Real-valued linear layers applied to both real and imaginary parts
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim] complex
        batch, seq_len, dim = x.shape
        # Separate real and imaginary parts and compute linear projections
        # We compute projections for real and imaginary parts, then recombine as complex.
        q = self.q_linear(x.real) + 1j * self.q_linear(x.imag)
        k = self.k_linear(x.real) + 1j * self.k_linear(x.imag)
        v = self.v_linear(x.real) + 1j * self.v_linear(x.imag)
        
        # Reshape and transpose for multi-head attention
        def shape_proj(t):
            t = t.view(batch, seq_len, self.heads, self.head_dim)
            return t.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        
        q, k, v = shape_proj(q), shape_proj(k), shape_proj(v)
        # Compute attention scores using the real part of the inner product
        scores = (q * k.conj()).sum(dim=-1).real * self.scale  # [batch, heads, seq_len, seq_len]
        if mask is not None:
            # mask shape expected to be [batch, seq_len] or broadcastable to scores
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # Weighted sum of values
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        # Final projection – apply separately to real and imaginary parts then combine
        out_real = self.out_linear(out.real)
        out_imag = self.out_linear(out.imag)
        out = torch.complex(out_real, out_imag)
        return out, attn

class HierarchicalModule(nn.Module):
    """
    Hierarchical processing module with adaptive pooling and skip connections.
    Processes input at word, phrase, and sentence levels.
    """
    def __init__(self, dim, heads=8, depth=3, dropout=0.1):
        super().__init__()
        self.word_layers = nn.ModuleList([
            ComplexAttentionLayer(dim, heads, dropout) for _ in range(depth)
        ])
        self.phrase_layers = nn.ModuleList([
            ComplexAttentionLayer(dim, heads, dropout) for _ in range(depth)
        ])
        self.sentence_layers = nn.ModuleList([
            ComplexAttentionLayer(dim, heads, dropout) for _ in range(max(1, depth//2))
        ])
        self.norm = ComplexLayerNorm(dim)
        # Linear projections for skip connections across scales
        self.proj_word_to_phrase = nn.Linear(dim, dim)
        self.proj_phrase_to_sentence = nn.Linear(dim, dim)
        
    def adaptive_pool(self, x, kernel_size):
        # x is a complex tensor: average pool separately on real and imaginary parts
        real = F.avg_pool1d(x.real.transpose(1,2), kernel_size, stride=kernel_size).transpose(1,2)
        imag = F.avg_pool1d(x.imag.transpose(1,2), kernel_size, stride=kernel_size).transpose(1,2)
        return torch.complex(real, imag)
    
    def forward(self, x):
        # x: [batch, seq_len, dim] complex
        # Word-level processing with residual connection
        word_out = x
        for layer in self.word_layers:
            delta, _ = layer(self.norm(word_out))
            word_out = word_out + delta  # residual
        
        # Adaptive pooling for phrase level: pool over a window (here fixed, could be learned)
        phrase_in = self.adaptive_pool(word_out, kernel_size=2)
        phrase_out = phrase_in
        for layer in self.phrase_layers:
            delta, _ = layer(self.norm(phrase_out))
            phrase_out = phrase_out + delta
        
        # Upsample phrase-level back to word-level size (simple nearest-neighbor repeat)
        up_phrase = phrase_out.repeat_interleave(2, dim=1)
        up_phrase = self.proj_word_to_phrase(up_phrase)
        
        # Combine with word-level with skip connection
        combined = word_out + up_phrase
        
        # Sentence-level: further pool and process
        sentence_in = self.adaptive_pool(combined, kernel_size=2)
        sentence_out = sentence_in
        for layer in self.sentence_layers:
            delta, _ = layer(self.norm(sentence_out))
            sentence_out = sentence_out + delta
        up_sentence = sentence_out.repeat_interleave(2, dim=1)
        up_sentence = self.proj_phrase_to_sentence(up_sentence)
        
        # Final fusion via skip connection and normalization
        fused = self.norm(combined + up_sentence)
        return fused

###############################################################################
#                           MEMORY MODULE                                    #
###############################################################################

class SimpleComplexMemory(nn.Module):
    """
    A simplified differentiable memory module for complex representations.
    Uses attention-based read and a gated write.
    """
    def __init__(self, mem_size, dim):
        super().__init__()
        self.mem_size = mem_size
        self.dim = dim
        # Initialize memory as a learnable complex tensor
        self.memory = nn.Parameter(torch.complex(
            torch.randn(mem_size, dim) * 0.01,
            torch.randn(mem_size, dim) * 0.01
        ))
        self.read_linear = nn.Linear(dim, dim)
        self.write_linear = nn.Linear(dim, dim)
        self.gate_linear = nn.Linear(dim * 2, 1)
        
    def address(self, x):
        # x: [batch, seq_len, dim] complex
        # Compute similarity (using real part) between x and memory entries
        # x_norm: [batch, seq_len, 1]
        x_norm = torch.abs(x)
        mem_norm = torch.abs(self.memory).unsqueeze(0)  # [1, mem_size, dim]
        scores = torch.einsum("bsd,md->bsm", x, self.memory.conj().real)
        scores = scores / (x_norm + 1e-8)
        weights = F.softmax(scores, dim=-1)
        return weights  # [batch, seq_len, mem_size]
    
    def read(self, x):
        weights = self.address(x)  # [batch, seq_len, mem_size]
        mem_content = torch.matmul(weights, self.memory)  # [batch, seq_len, dim] complex
        read_out = self.read_linear(mem_content.real) + 1j * self.read_linear(mem_content.imag)
        return read_out
    
    def write(self, x):
        # For simplicity, write using the last token of x (detached to avoid loops)
        last_token = x[:, -1, :].detach()  # [batch, dim] complex
        write_gate = torch.sigmoid(self.gate_linear(torch.cat([last_token.real, last_token.imag], dim=-1)))
        update = self.write_linear(last_token.real) + 1j * self.write_linear(last_token.imag)
        # Update memory with gated update (here applied to the first memory slot)
        self.memory.data[0] = (1 - write_gate) * self.memory.data[0] + write_gate * update
        return

###############################################################################
#                    FINAL COLLAPSE & MODEL DEFINITION                       #
###############################################################################

class CollapseModule(nn.Module):
    """
    Collapse module that maps complex representations to vocabulary logits.
    Uses squared-magnitude as the density measure.
    """
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        # x is complex, take squared magnitude then apply linear mapping
        density = x.real ** 2 + x.imag ** 2  # [batch, seq_len, dim]
        logits = self.linear(density)
        return F.relu(logits) + 1e-10

class AdvancedSFIN(nn.Module):
    """
    Advanced SFIN with complex-valued embeddings, hierarchical attention,
    a simple differentiable memory, and a collapse module.
    """
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, dropout=0.1,
                 max_seq_len=256, mem_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        self.embedding = ComplexEmbedding(vocab_size, dim)
        self.pos_encoding = ComplexPositionalEncoding(dim, max_len=max_seq_len)
        self.hierarchical = HierarchicalModule(dim, heads, depth//3, dropout)
        self.memory = SimpleComplexMemory(mem_size, dim)
        self.norm = ComplexLayerNorm(dim)
        self.collapse = CollapseModule(dim, vocab_size)
        
    def forward(self, x, mask=None, return_attention=False):
        # x: [batch, seq_len] token ids
        emb = self.embedding(x)                      # complex [B, L, dim]
        emb = self.pos_encoding(emb)                 # add positional encoding
        emb = self.norm(emb)
        # Hierarchical processing
        h = self.hierarchical(emb)
        # Read memory (during evaluation, memory is used; during training we might detach)
        mem_out = self.memory.read(h)
        h = h + mem_out
        h = self.norm(h)
        # Collapse to logits
        logits = self.collapse(h)
        return logits

###############################################################################
#                            DATASET & DATALOADER                            #
###############################################################################

class EnhancedSFINDataset(Dataset):
    """
    Dataset for SFIN. Uses a given tokenizer to process texts.
    """
    def __init__(self, texts, tokenizer, max_length=256, min_length=8, augment=False, augment_prob=0.1):
        self.tokenizer = tokenizer
        self.inputs = []
        self.max_length = max_length
        self.min_length = min_length
        self.augment = augment
        self.augment_prob = augment_prob
        logger.info(f"Processing {len(texts)} texts for dataset creation")
        for text in tqdm(texts, desc="Processing dataset"):
            if len(text.split()) < min_length:
                continue
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            self.inputs.append({
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": encodings["input_ids"].squeeze(),
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def _augment_text(self, input_ids, attention_mask):
        augmented_ids = input_ids.clone()
        if not self.augment or torch.rand(1).item() > self.augment_prob:
            return augmented_ids, attention_mask
        valid_positions = torch.where(attention_mask == 1)[0]
        if len(valid_positions) <= 2:
            return augmented_ids, attention_mask
        mask_positions = torch.rand(len(valid_positions)) < 0.15
        positions_to_mask = valid_positions[mask_positions]
        for pos in positions_to_mask:
            prob = torch.rand(1).item()
            if prob < 0.8:
                augmented_ids[pos] = self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else 0
            elif prob < 0.9:
                augmented_ids[pos] = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()
        return augmented_ids, attention_mask
    
    def __getitem__(self, idx):
        item = {k: v.clone() for k, v in self.inputs[idx].items()}
        if self.augment:
            item["input_ids"], item["attention_mask"] = self._augment_text(item["input_ids"], item["attention_mask"])
        return item

###############################################################################
#                    TRAINING, EVALUATION, & HYPERPARAMETER TUNING            #
###############################################################################

def compute_metrics(preds, targets):
    preds = preds.flatten()
    targets = targets.flatten()
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def evaluate(model, dataloader, loss_fn, use_fp16=True):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if use_fp16:
                with autocast():
                    logits = model(input_ids, mask=attn_mask)
                    active_loss = attn_mask.view(-1) == 1
                    active_logits = logits.view(-1, model.vocab_size)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fn(active_logits, active_labels)
            else:
                logits = model(input_ids, mask=attn_mask)
                active_loss = attn_mask.view(-1) == 1
                active_logits = logits.view(-1, model.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
            total_loss += loss.item()
            preds = torch.argmax(active_logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(active_labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds.numpy(), all_labels.numpy())
    metrics["loss"] = total_loss / len(dataloader)
    return metrics

def plot_gradient_flow(model, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    avg_grads = []
    layers = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(avg_grads)), avg_grads, alpha=0.6)
    plt.xticks(range(len(avg_grads)), layers, rotation="vertical")
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_flow.png"))
    plt.close()

def train_model(model, train_dataloader, eval_dataloader=None, epochs=3, lr=5e-5,
                warmup_steps=100, use_fp16=True, log_interval=10, checkpoint_dir="checkpoints",
                scheduler_type="linear", weight_decay=0.01, grad_accum_steps=1,
                max_grad_norm=1.0, early_stopping_patience=3):
    os.makedirs(checkpoint_dir, exist_ok=True)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_dataloader) * epochs // grad_accum_steps
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_fp16 else None

    global_step = 0
    best_eval_loss = float("inf")
    early_stop_counter = 0
    model.train()
    if config["use_tensorboard"]:
        tb_writer = SummaryWriter(os.path.join(checkpoint_dir, "runs"))
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            try:
                if use_fp16:
                    with autocast():
                        logits = model(input_ids, mask=attn_mask)
                        active_loss = attn_mask.view(-1) == 1
                        active_logits = logits.view(-1, model.vocab_size)[active_loss]
                        active_labels = labels.view(-1)[active_loss]
                        loss = loss_fn(active_logits, active_labels) / grad_accum_steps
                else:
                    logits = model(input_ids, mask=attn_mask)
                    active_loss = attn_mask.view(-1) == 1
                    active_logits = logits.view(-1, model.vocab_size)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fn(active_logits, active_labels) / grad_accum_steps
                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % log_interval == 0 and config["use_tensorboard"]:
                        tb_writer.add_scalar("train/loss", loss.item() * grad_accum_steps, global_step)
                epoch_loss += loss.item() * grad_accum_steps
            except Exception as e:
                logger.error(f"Error at step {step}: {str(e)}")
                optimizer.zero_grad()
                continue
        
        logger.info(f"Epoch {epoch+1} loss: {epoch_loss/len(train_dataloader):.4f} in {time.time()-start_time:.2f}s")
        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"advanced_sfin_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_loss/len(train_dataloader)
        }, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        # Evaluation
        if eval_dataloader is not None:
            metrics = evaluate(model, eval_dataloader, loss_fn, use_fp16)
            logger.info(f"Eval metrics at epoch {epoch+1}: {metrics}")
            if metrics["loss"] < best_eval_loss:
                best_eval_loss = metrics["loss"]
                early_stop_counter = 0
                best_path = os.path.join(checkpoint_dir, "advanced_sfin_best_model.pt")
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "loss": best_eval_loss,
                    "metrics": metrics,
                }, best_path)
                logger.info(f"New best model saved: {best_path}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break
    if config["use_tensorboard"]:
        tb_writer.close()
    return

###############################################################################
#                           TEXT GENERATION                                  #
###############################################################################

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8,
                  top_p=0.9, top_k=50, repetition_penalty=1.1, num_return_sequences=1):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_sequences = []
    for _ in range(num_return_sequences):
        cur_ids = input_ids.clone()
        for i in range(max_length):
            with torch.no_grad():
                logits = model(cur_ids)
                next_logits = logits[:, -1, :] / temperature
                # Apply repetition penalty
                for token_id in cur_ids.view(-1).unique():
                    next_logits[:, token_id] /= repetition_penalty
                # Top-k filtering
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(next_logits, top_k)
                    filtered_logits = torch.full_like(next_logits, float('-inf'))
                    filtered_logits.scatter_(1, topk_idx, topk_vals)
                    next_logits = filtered_logits
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_idx_to_remove = cumulative_probs > top_p
                    sorted_idx_to_remove[..., 0] = 0
                    for batch_idx in range(next_logits.size(0)):
                        next_logits[batch_idx, sorted_idx[batch_idx][sorted_idx_to_remove[batch_idx]]] = -float('inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        generated_sequences.append(cur_ids)
    decoded = [tokenizer.decode(seq[0], skip_special_tokens=True) for seq in generated_sequences]
    return decoded

###############################################################################
#                         EXPLAINABILITY TOOLS                               #
###############################################################################

class ExplainabilityTools:
    """
    Tools to visualize attention and analyze token-level contributions.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def visualize_attention(self, text_input, layer_idx=0, head_idx=0, output_dir="visualizations"):
        os.makedirs(output_dir, exist_ok=True)
        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(device)
        with torch.no_grad():
            # For simplicity, we assume our hierarchical module’s first attention layer is of interest.
            _, attn = self.model.hierarchical.word_layers[layer_idx](self.model.norm(self.model.embedding(input_ids)))
        attn_map = attn[0, head_idx].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f"Attention Map (Layer {layer_idx}, Head {head_idx})")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"attn_layer{layer_idx}_head{head_idx}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Attention visualization saved to {save_path}")
        return attn_map

    def explain_generation(self, prompt, max_length=30, temperature=0.8):
        # Token-by-token analysis using attention snapshots (placeholder implementation)
        self.model.eval()
        token_explanations = []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()
        for i in range(max_length):
            with torch.no_grad():
                logits = self.model(generated)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                token = self.tokenizer.decode(next_token[0])
                token_explanations.append({"token": token, "probability": probs[0, next_token].item()})
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        final_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return final_text, token_explanations

    def model_parameter_summary(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info = {"total_parameters": total, "trainable_parameters": trainable}
        return info

###############################################################################
#                             MAIN FUNCTION                                  #
###############################################################################

def main(mode="train", use_hyperopt=False, load_checkpoint=None):
    try:
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        logger.info(f"Tokenizer initialized with vocab size: {vocab_size}")
        
        # Model configuration
        model_args = {
            "vocab_size": vocab_size,
            "dim": 768,
            "depth": 7,
            "heads": 8,
            "dropout": 0.15,
            "max_seq_len": 256,
            "mem_size": 32
        }
        if load_checkpoint:
            logger.info(f"Loading model checkpoint from {load_checkpoint}")
            checkpoint = torch.load(load_checkpoint, map_location=device)
            model = AdvancedSFIN(**model_args).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model loaded successfully.")
        else:
            logger.info("Creating a new AdvancedSFIN model...")
            model = AdvancedSFIN(**model_args).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset_raw = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
        texts = []
        for ex in dataset_raw:
            if "text" in ex:
                texts.append(ex["text"])
            elif "instruction" in ex:
                texts.append(ex["instruction"])
            elif "conversation" in ex:
                texts.append(ex["conversation"])
            else:
                combined = " ".join(str(v) for v in ex.values() if isinstance(v, str))
                if combined:
                    texts.append(combined)
        texts = [t for t in texts if len(t.split()) >= 8]
        split_idx = int(0.9 * len(texts))
        train_texts, eval_texts = texts[:split_idx], texts[split_idx:]
        logger.info(f"Dataset loaded: {len(train_texts)} train and {len(eval_texts)} eval samples")
        
        # Configure batch size
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                batch_size = max(1, min(6, int(free_mem / (1024**3 * 0.8))))
            except:
                batch_size = 4
        else:
            batch_size = 1
        logger.info(f"Using batch size: {batch_size}")
        
        if mode in ["train", "evaluate"]:
            train_dataset = EnhancedSFINDataset(train_texts, tokenizer, max_length=256)
            eval_dataset = EnhancedSFINDataset(eval_texts, tokenizer, max_length=256)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        if mode == "train":
            logger.info("Starting training mode...")
            train_model(
                model,
                train_loader,
                eval_dataloader=eval_loader,
                epochs=15,
                lr=6e-5,
                warmup_steps=500,
                use_fp16=torch.cuda.is_available(),
                log_interval=10,
                checkpoint_dir="checkpoints",
                scheduler_type="cosine",
                weight_decay=0.01,
                grad_accum_steps=max(1, 8 // batch_size),
                max_grad_norm=1.0,
                early_stopping_patience=5
            )
        elif mode == "evaluate":
            logger.info("Starting evaluation mode...")
            loss_fn = nn.CrossEntropyLoss()
            metrics = evaluate(model, eval_loader, loss_fn, use_fp16=torch.cuda.is_available())
            logger.info("Evaluation results:")
            for key, value in metrics.items():
                logger.info(f"{key}: {value:.4f}")
        elif mode == "generate":
            logger.info("Starting text generation mode...")
            generated = generate_text(model, tokenizer, prompt="The nature of language is", max_length=100,
                                      temperature=0.85, top_p=0.92, top_k=40)
            for idx, text in enumerate(generated):
                logger.info(f"Generated sequence {idx+1}: {text}")
        elif mode == "explain":
            logger.info("Starting explainability mode...")
            explainer = ExplainabilityTools(model, tokenizer)
            param_summary = explainer.model_parameter_summary()
            logger.info(f"Model parameter summary: {param_summary}")
            sample_text = "Complex interactions in language emerge through layered representations."
            attn_map = explainer.visualize_attention(sample_text, layer_idx=0, head_idx=0)
            gen_text, token_explanations = explainer.explain_generation("Language as a complex system", max_length=30)
            logger.info(f"Generated text: {gen_text}")
            for idx, token_info in enumerate(token_explanations):
                logger.info(f"Token {idx+1}: {token_info}")
        else:
            logger.info(f"Unknown mode: {mode}")
        
        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AdvancedSFIN: Complex-Valued Hierarchical Attention Model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "generate", "explain"], help="Operation mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=None, help="Override automatic batch size selection")
    parser.add_argument("--learning_rate", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--model_dim", type=int, default=768, help="Model hidden dimension size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(mode=args.mode, load_checkpoint=args.checkpoint)
