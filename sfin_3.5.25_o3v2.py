#!/usr/bin/env python
"""
ImprovedAdvancedSFIN.py

An improved single-file implementation of a quantum-inspired, hierarchical,
complex-valued attention model ("SFIN") that preserves core ideas:
  - Complex embeddings and positional encodings
  - Quantum noise injection
  - Entangled (multi-head) interference layers
  - Hierarchical attention (word/phrase/sentence scales)
  - Wavefunction collapse
  - Simple memory module
  - Training, evaluation, and generation in one file

Usage:
  python ImprovedAdvancedSFIN.py --mode train
  python ImprovedAdvancedSFIN.py --mode evaluate
  python ImprovedAdvancedSFIN.py --mode generate
  python ImprovedAdvancedSFIN.py --mode explain
"""

import os
import math
import time
import random
import logging
import warnings
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

###############################################################################
#                           LOGGING & CONFIG                                  #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("improved_advanced_sfin.log")
    ]
)
logger = logging.getLogger("ImprovedAdvancedSFIN")
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

config = {
    "use_tensorboard": True,
    "explainability_mode": "all",
    "plot_gradient_flow": True,
    "save_attention_heatmaps": True
}
if config["use_tensorboard"]:
    writer = SummaryWriter("runs/improved_sfin_experiment")

###############################################################################
#                      QUANTUM / COMPLEX UTILS                                #
###############################################################################

def add_quantum_noise(tensor, noise_prob=0.05, noise_scale=0.1):
    """
    Inject quantum-inspired noise:
      - Phase flips (sign flip)
      - Small amplitude perturbations
      - Basic measurement noise (randomly zero out)
    """
    if not tensor.requires_grad or noise_scale <= 0:
        return tensor
    # Phase flip
    phase_mask = (torch.rand_like(tensor.real) < noise_prob).float()
    flip = torch.where(phase_mask > 0, -1.0, 1.0)
    # amplitude noise
    amp_mask = (torch.rand_like(tensor.real) < noise_prob).float()
    real_noise = torch.randn_like(tensor.real) * noise_scale * amp_mask
    imag_noise = torch.randn_like(tensor.imag) * noise_scale * amp_mask
    
    new_real = tensor.real * flip + real_noise
    new_imag = tensor.imag * flip + imag_noise
    # measurement noise: random zero out
    measure_mask = (torch.rand_like(tensor.real) < noise_prob * 0.5).float()
    new_real = new_real * (1 - measure_mask)
    new_imag = new_imag * (1 - measure_mask)
    return torch.complex(new_real, new_imag)

###############################################################################
#                      COMPLEX EMBEDDINGS & ENCODING                          #
###############################################################################

class ComplexEmbedding(nn.Module):
    """
    Complex embedding with separate real/imag parts, combined as torch.complex.
    """
    def __init__(self, vocab_size, embedding_dim, init_scale=0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.emb_real = nn.Embedding(vocab_size, embedding_dim)
        self.emb_imag = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.emb_real.weight, mean=0.0, std=init_scale)
        nn.init.normal_(self.emb_imag.weight, mean=0.0, std=init_scale)

    def forward(self, input_ids):
        real = self.emb_real(input_ids)
        imag = self.emb_imag(input_ids)
        return torch.complex(real, imag)

class ComplexPositionalEncoding(nn.Module):
    """
    Complex positional encoding, using sin/cos expansions but stored as complex.
    """
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim)
        )
        pe_real = torch.zeros(max_len, embedding_dim)
        pe_imag = torch.zeros(max_len, embedding_dim)
        pe_real[:, 0::2] = torch.sin(pos * div_term)
        pe_real[:, 1::2] = torch.cos(pos * div_term)
        pe_imag[:, 0::2] = torch.cos(pos * div_term)
        pe_imag[:, 1::2] = -torch.sin(pos * div_term)
        self.register_buffer("pe_real", pe_real)
        self.register_buffer("pe_imag", pe_imag)

    def forward(self, x):
        # x: [batch, seq_len, embedding_dim] complex
        seq_len = x.size(1)
        # Add complex positional encoding
        real = x.real + self.pe_real[:seq_len, :]
        imag = x.imag + self.pe_imag[:seq_len, :]
        return torch.complex(real, imag)

class ComplexLayerNorm(nn.Module):
    """
    LayerNorm on real and imaginary parts separately.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm_real = nn.LayerNorm(dim, eps=eps)
        self.norm_imag = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return torch.complex(
            self.norm_real(x.real),
            self.norm_imag(x.imag)
        )

###############################################################################
#                       ENTANGLED QUANTUM ATTENTION                           #
###############################################################################

class EntangledInterferenceLayer(nn.Module):
    """
    Quantum-inspired multi-head attention with:
      - Complex Q/K/V
      - Entanglement matrix across heads
      - Quantum noise injection
      - Interference-based scoring (real part + cross terms)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, use_entanglement=True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_entanglement = use_entanglement

        self.q_real = nn.Linear(dim, dim)
        self.q_imag = nn.Linear(dim, dim)
        self.k_real = nn.Linear(dim, dim)
        self.k_imag = nn.Linear(dim, dim)
        self.v_real = nn.Linear(dim, dim)
        self.v_imag = nn.Linear(dim, dim)
        self.out_real = nn.Linear(dim, dim)
        self.out_imag = nn.Linear(dim, dim)

        if use_entanglement:
            # Initialize entanglement matrix with slight off-diagonal
            ent_init = torch.eye(num_heads)
            offdiag = (torch.randn(num_heads, num_heads) * 0.1)
            mask = (torch.rand(num_heads, num_heads) < 0.2)
            ent_init = ent_init + offdiag * mask
            self.entanglement_matrix = nn.Parameter(ent_init)
        else:
            self.register_buffer('entanglement_matrix', torch.eye(num_heads))

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, dim] complex
        """
        B, L, D = x.shape

        # 1) Split into real & imag and do linear projections
        qr = self.q_real(x.real) + 1j * self.q_real(x.imag)
        qi = self.q_imag(x.real) + 1j * self.q_imag(x.imag)
        # Actually, let's keep it simpler:
        q = torch.complex(
            self.q_real(x.real), 
            self.q_imag(x.imag)
        )
        k = torch.complex(
            self.k_real(x.real),
            self.k_imag(x.imag)
        )
        v = torch.complex(
            self.v_real(x.real),
            self.v_imag(x.imag)
        )
        # Add quantum noise
        q = add_quantum_noise(q)
        k = add_quantum_noise(k)
        v = add_quantum_noise(v)

        # 2) Reshape for multi-head
        def reshape_heads(t):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # 3) Possibly entangle heads
        if self.use_entanglement:
            # entanglement_matrix: [num_heads, num_heads]
            # q: [B, num_heads, L, head_dim]
            # We'll do a matrix multiply across the head dimension:
            # shape: q -> (B, L, num_heads, head_dim) if we transpose
            # simpler approach: batch-wise multiply
            # We'll treat it as q -> q x ent_matrix in head dimension
            # So q: (B, num_heads, L, head_dim) => we want (num_heads, num_heads)
            # We'll do an einsum:
            q = torch.einsum("bhld,hh->bhld", q, self.entanglement_matrix)
            k = torch.einsum("bhld,hh->bhld", k, self.entanglement_matrix)
        
        # 4) Attention scores using real part of (q * k.conj())
        #    (plus small imaginary cross terms if we want, but let's keep real)
        scores = (q * k.conj()).sum(dim=-1).real * self.scale  # [B, num_heads, L, L]

        if mask is not None:
            # mask: [B, L], broadcast to [B, 1, 1, L]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5) Weighted sum
        out = torch.einsum("bhal,bhld->bhad", attn_weights, v)
        # out: [B, num_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)

        # 6) Final linear output (split real & imag)
        out_r = self.out_real(out.real)
        out_i = self.out_imag(out.imag)
        out_c = torch.complex(out_r, out_i)
        out_c = self.resid_dropout(out_c)
        return out_c, attn_weights

###############################################################################
#                          WAVEFUNCTION COLLAPSE                              #
###############################################################################

class AdvancedWaveFunctionCollapse(nn.Module):
    """
    Takes complex output and collapses it to vocab logits.
    Using an "interference" approach: real^2 + imag^2 + cross-terms.
    """
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.collapse = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x: [B, L, dim] complex
        # We'll compute amplitude^2 + optional cross-phase
        # amplitude^2
        amplitude_sq = x.real**2 + x.imag**2
        # optional cross-phase (sin(real) * cos(imag)) just as an example
        cross_term = 0.3 * (torch.sin(x.real) * torch.cos(x.imag)).mean(dim=-1, keepdim=True)
        # combine
        combined = amplitude_sq + cross_term
        logits = self.collapse(combined)
        return logits

###############################################################################
#                              MEMORY MODULE                                  #
###############################################################################

class SimpleQuantumMemory(nn.Module):
    """
    A minimal differentiable memory for complex representations:
    - We store memory as a parameter
    - We read via attention
    - We do a simple gated write
    """
    def __init__(self, mem_size, dim):
        super().__init__()
        self.mem_size = mem_size
        self.dim = dim
        # Initialize memory as complex
        self.memory = nn.Parameter(
            torch.complex(
                torch.randn(mem_size, dim)*0.01,
                torch.randn(mem_size, dim)*0.01
            )
        )
        self.read_proj = nn.Linear(dim, dim)
        self.write_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim*2, 1)

    def forward(self, x, do_write=True):
        """
        x: [B, L, dim] complex
        do_write: whether to write to memory (training mode)
        Returns read output of shape [B, L, dim]
        """
        B, L, D = x.shape
        # 1) read
        # address: compare x with memory
        # use real part of dot product for simplicity
        # x => [B, L, dim], memory => [mem_size, dim]
        # scores => [B, L, mem_size]
        x_real = x.real
        mem_real = self.memory.conj().real  # [mem_size, dim]
        scores = torch.einsum("bld,md->blm", x_real, mem_real)
        # softmax over mem_size
        attn_weights = F.softmax(scores, dim=-1)
        mem_expanded = self.memory.unsqueeze(0).unsqueeze(0)  # [1,1,mem_size,dim]
        # Weighted sum => [B,L,dim]
        read_vals = torch.einsum("blm,mn->bln", attn_weights, self.memory)  # shape [B,L,dim] complex
        read_out = torch.complex(
            self.read_proj(read_vals.real),
            self.read_proj(read_vals.imag)
        )
        # 2) write
        if do_write and self.training:
            # pick last token for gating
            last_tok = x[:, -1, :].detach()  # [B, dim]
            g = torch.sigmoid(
                self.gate(torch.cat([last_tok.real, last_tok.imag], dim=-1))
            ).mean()  # average gate across batch
            # write update
            update_val = torch.complex(
                self.write_proj(last_tok.real),
                self.write_proj(last_tok.imag)
            ).mean(dim=0)  # average across batch
            # apply
            with torch.no_grad():
                self.memory[0] = (1-g)*self.memory[0] + g*update_val
        return read_out

###############################################################################
#                     HIERARCHICAL INTERFERENCE MODULE                        #
###############################################################################

class HierarchicalInterferenceModule(nn.Module):
    """
    Multi-scale approach:
      - word-level, phrase-level, sentence-level
      - each scale is an EntangledInterferenceLayer repeated a few times
      - pool & unpool
    """
    def __init__(self, dim, heads=8, depth=6, dropout=0.1):
        super().__init__()
        # Let's split depth among word, phrase, sentence
        word_depth = depth // 3
        phrase_depth = depth // 3
        sentence_depth = depth - word_depth - phrase_depth

        self.word_layers = nn.ModuleList([
            EntangledInterferenceLayer(dim, heads, dropout) for _ in range(word_depth)
        ])
        self.phrase_layers = nn.ModuleList([
            EntangledInterferenceLayer(dim, heads, dropout) for _ in range(phrase_depth)
        ])
        self.sentence_layers = nn.ModuleList([
            EntangledInterferenceLayer(dim, heads, dropout) for _ in range(sentence_depth)
        ])

        self.norm = ComplexLayerNorm(dim)

    def pool(self, x, size=2):
        # simple average pool
        real = F.avg_pool1d(x.real.transpose(1,2), size, stride=size).transpose(1,2)
        imag = F.avg_pool1d(x.imag.transpose(1,2), size, stride=size).transpose(1,2)
        return torch.complex(real, imag)

    def unpool(self, x, size=2, target_len=None):
        # nearest neighbor upsample
        B, L, D = x.shape
        # repeat_interleave in the time dimension
        up = x.repeat_interleave(size, dim=1)
        # optionally slice to target_len
        if target_len is not None and up.size(1) > target_len:
            up = up[:, :target_len, :]
        return up

    def forward(self, x):
        # word-level
        out = x
        for layer in self.word_layers:
            normed = self.norm(out)
            delta, _ = layer(normed)
            out = out + delta
        word_out = out

        # phrase-level
        phrase_in = self.pool(word_out, size=2)
        phrase_out = phrase_in
        for layer in self.phrase_layers:
            normed = self.norm(phrase_out)
            delta, _ = layer(normed)
            phrase_out = phrase_out + delta
        # unpool back
        up_phrase = self.unpool(phrase_out, size=2, target_len=word_out.size(1))
        out = word_out + up_phrase

        # sentence-level
        sent_in = self.pool(out, size=2)
        sent_out = sent_in
        for layer in self.sentence_layers:
            normed = self.norm(sent_out)
            delta, _ = layer(normed)
            sent_out = sent_out + delta
        up_sentence = self.unpool(sent_out, size=2, target_len=out.size(1))
        final = self.norm(out + up_sentence)
        return final

###############################################################################
#                               MAIN MODEL                                    #
###############################################################################

class AdvancedSFIN(nn.Module):
    """
    The main SFIN model integrating:
      - Complex embedding + positional encoding
      - Hierarchical interference module
      - Quantum memory
      - Final wavefunction collapse
    """
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, dropout=0.1, max_seq_len=256, mem_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = ComplexEmbedding(vocab_size, dim)
        self.pos_enc = ComplexPositionalEncoding(dim, max_len=max_seq_len)
        self.input_norm = ComplexLayerNorm(dim)
        self.hier = HierarchicalInterferenceModule(dim, heads, depth, dropout)
        self.memory = SimpleQuantumMemory(mem_size, dim)
        self.final_norm = ComplexLayerNorm(dim)
        self.collapse = AdvancedWaveFunctionCollapse(dim, vocab_size)

    def forward(self, input_ids, mask=None, do_write=True):
        """
        input_ids: [B, L] integer tokens
        mask: [B, L] optional attention mask
        """
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.input_norm(x)

        # hierarchical interference
        h = self.hier(x)

        # memory read/write
        mem_out = self.memory(h, do_write=do_write)
        h = h + mem_out
        h = self.final_norm(h)

        # collapse
        logits = self.collapse(h)
        return logits

###############################################################################
#                         DATASET & DATALOADER                                #
###############################################################################

class SFINDataset(Dataset):
    """
    Basic dataset that tokenizes input texts and returns (input_ids, attention_mask, labels).
    """
    def __init__(self, texts, tokenizer, max_length=256, min_length=8):
        self.tokenizer = tokenizer
        self.examples = []
        for txt in tqdm(texts, desc="Creating SFINDataset"):
            if len(txt.split()) < min_length:
                continue
            enc = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            self.examples.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": enc["input_ids"].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

###############################################################################
#                    TRAINING, EVALUATION & GENERATION                        #
###############################################################################

def compute_metrics(preds, targets):
    preds = preds.flatten()
    targets = targets.flatten()
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def evaluate(model, dataloader, loss_fn, fp16=True):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if fp16:
                with autocast():
                    logits = model(input_ids, mask=attn_mask, do_write=False)
                    active = attn_mask.view(-1) == 1
                    active_logits = logits.view(-1, model.vocab_size)[active]
                    active_labels = labels.view(-1)[active]
                    loss = loss_fn(active_logits, active_labels)
            else:
                logits = model(input_ids, mask=attn_mask, do_write=False)
                active = attn_mask.view(-1) == 1
                active_logits = logits.view(-1, model.vocab_size)[active]
                active_labels = labels.view(-1)[active]
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

def train_model(model, train_dl, eval_dl=None, epochs=3, lr=5e-5, warmup_steps=100,
                fp16=True, grad_accum_steps=1, max_grad_norm=1.0,
                scheduler_type="linear", weight_decay=0.01, log_interval=10,
                early_stop_patience=3, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = len(train_dl)*epochs//grad_accum_steps
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler() if fp16 else None
    loss_fn = nn.CrossEntropyLoss()
    global_step = 0
    best_eval_loss = float("inf")
    early_stop_counter = 0

    model.train()
    if config["use_tensorboard"]:
        tb_writer = SummaryWriter(os.path.join(checkpoint_dir, "runs"))
    logger.info("Starting training...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        start_t = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                if fp16:
                    with autocast():
                        logits = model(input_ids, mask=attn_mask, do_write=True)
                        active = attn_mask.view(-1) == 1
                        active_logits = logits.view(-1, model.vocab_size)[active]
                        active_labels = labels.view(-1)[active]
                        loss = loss_fn(active_logits, active_labels)/grad_accum_steps
                else:
                    logits = model(input_ids, mask=attn_mask, do_write=True)
                    active = attn_mask.view(-1) == 1
                    active_logits = logits.view(-1, model.vocab_size)[active]
                    active_labels = labels.view(-1)[active]
                    loss = loss_fn(active_logits, active_labels)/grad_accum_steps

                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step+1) % grad_accum_steps == 0:
                    if fp16:
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
                        tb_writer.add_scalar("train/loss", loss.item()*grad_accum_steps, global_step)

                epoch_loss += loss.item()*grad_accum_steps
            except Exception as e:
                logger.error(f"Error at step {step}: {str(e)}")
                optimizer.zero_grad()
                continue

        logger.info(f"Epoch {epoch+1} completed, avg loss={epoch_loss/len(train_dl):.4f}, time={time.time()-start_t:.2f}s")

        # Evaluate
        if eval_dl is not None:
            metrics = evaluate(model, eval_dl, loss_fn, fp16)
            logger.info(f"Eval metrics at epoch {epoch+1}: {metrics}")
            if metrics["loss"] < best_eval_loss:
                best_eval_loss = metrics["loss"]
                early_stop_counter = 0
                # save best
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "loss": best_eval_loss,
                    "metrics": metrics
                }, best_path)
                logger.info(f"New best model saved to {best_path}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

        # Save epoch checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"sfin_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "loss": epoch_loss/len(train_dl)
        }, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

    if config["use_tensorboard"]:
        tb_writer.close()

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.1):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated, do_write=False)
            next_logits = logits[:, -1, :] / temperature
            # repetition penalty
            for token_id in generated.view(-1).unique():
                next_logits[:, token_id] /= repetition_penalty

            # top-k
            if top_k > 0:
                vals, idx = torch.topk(next_logits, top_k)
                filtered = torch.full_like(next_logits, float('-inf'))
                filtered.scatter_(1, idx, vals)
                next_logits = filtered

            # top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_idx = cum_probs > top_p
                remove_idx[..., 0] = 0
                for b in range(next_logits.size(0)):
                    next_logits[b, sorted_idx[b][remove_idx[b]]] = -float('inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

###############################################################################
#                            EXPLAINABILITY                                   #
###############################################################################

class ExplainabilityTools:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def model_parameter_summary(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def explain_generation(self, prompt, max_length=30):
        """
        Simple token-by-token analysis (no advanced attention hooking).
        """
        self.model.eval()
        tokens_info = []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()
        for i in range(max_length):
            with torch.no_grad():
                logits = self.model(generated, do_write=False)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_str = self.tokenizer.decode(next_token[0])
                tokens_info.append({"token": token_str, "prob": probs[0, next_token].item()})
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        final_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return final_text, tokens_info

###############################################################################
#                               MAIN                                          #
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "generate", "explain"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # Initialize tokenizer
    logger.info("Loading tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size}")

    # Build model
    model = AdvancedSFIN(
        vocab_size=vocab_size,
        dim=512,
        depth=6,
        heads=8,
        dropout=0.1,
        max_seq_len=256,
        mem_size=32
    ).to(device)

    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Checkpoint loaded.")

    # Load dataset
    dataset_name = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
    logger.info(f"Loading dataset: {dataset_name}")
    raw_ds = load_dataset(dataset_name, split="train_sft")
    texts = []
    for ex in raw_ds:
        if "text" in ex:
            texts.append(ex["text"])
        elif "instruction" in ex:
            texts.append(ex["instruction"])
        elif "conversation" in ex:
            texts.append(ex["conversation"])
        else:
            # fallback: combine all strings
            combined = " ".join(str(v) for v in ex.values() if isinstance(v, str))
            if combined:
                texts.append(combined)
    texts = [t for t in texts if len(t.split()) >= 8]
    random.shuffle(texts)
    split_idx = int(0.9*len(texts))
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    logger.info(f"Train samples: {len(train_texts)}, Eval samples: {len(eval_texts)}")

    # Dataloaders
    train_dataset = SFINDataset(train_texts, tokenizer, max_length=256)
    eval_dataset = SFINDataset(eval_texts, tokenizer, max_length=256)
    batch_size = 4 if torch.cuda.is_available() else 1
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dl = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    if args.mode == "train":
        logger.info("Starting training...")
        train_model(
            model,
            train_dl,
            eval_dl=eval_dl,
            epochs=args.epochs,
            lr=args.lr,
            fp16=torch.cuda.is_available(),
            grad_accum_steps=1,
            max_grad_norm=1.0,
            scheduler_type="linear",
            weight_decay=0.01,
            log_interval=10,
            early_stop_patience=3,
            checkpoint_dir="checkpoints_improved_sfin"
        )
    elif args.mode == "evaluate":
        logger.info("Evaluating...")
        loss_fn = nn.CrossEntropyLoss()
        metrics = evaluate(model, eval_dl, loss_fn, fp16=torch.cuda.is_available())
        logger.info(f"Eval metrics: {metrics}")
    elif args.mode == "generate":
        logger.info("Generating text...")
        prompt = "Quantum entanglement suggests"
        generated = generate_text(
            model, tokenizer, prompt=prompt,
            max_length=60, temperature=0.85, top_p=0.9, top_k=40
        )
        logger.info(f"Prompt: {prompt}\nGenerated: {generated}")
    elif args.mode == "explain":
        logger.info("Explainability mode...")
        explainer = ExplainabilityTools(model, tokenizer)
        summary = explainer.model_parameter_summary()
        logger.info(f"Model parameter summary: {summary}")
        sample_prompt = "Language emerges from complex interference"
        text, tokens_info = explainer.explain_generation(sample_prompt, max_length=20)
        logger.info(f"Generated text: {text}")
        for i, tok in enumerate(tokens_info):
            logger.info(f"Token {i+1}: {tok}")

    logger.info("Done.")

if __name__ == "__main__":
    main()
