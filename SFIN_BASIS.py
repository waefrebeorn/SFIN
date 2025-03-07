"""
Advanced Semantic Field Interference Network (SFIN) - Complete Implementation
- Incorporates quantum entanglement in attention, quantum noise injection,
  adaptive interference strength, curriculum collapse scheduling, multi-scale interference,
  memory augmentation, quantum-inspired evaluation metrics, and explainable interference.
- Uses the Everyday Conversations for Smol LLMs (2.2k) dataset.
- Designed to run on standard hardware (no 4050-specific optimizations).
- Added cross-modal branches, automated hyperparameter tuning and enhanced explainability.
"""

import os
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
import optuna
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sfin_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SFIN")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")

# Global configuration for logging and visualization
config = {
    "log_attention_maps": True,
    "save_attention_heatmaps": True,
    "plot_gradient_flow": True,
    "adaptive_training": True,
    "use_tensorboard": True,
    "interference_visualizations": True,
    "explainability_mode": "all"  # "attention", "entropy", "gradients", "all"
}

# TensorBoard setup
if config["use_tensorboard"]:
    writer = SummaryWriter("runs/sfin_experiment")

###############################################################################
#                     QUANTUM & HELPER FUNCTIONS                              #
###############################################################################

def add_quantum_noise(tensor, noise_prob=0.05, noise_scale=0.1, noise_type="phase_and_amplitude"):
    """
    Inject quantum-inspired noise:
    - Phase noise: flips sign (simulates phase flip)
    - Amplitude noise: adds small Gaussian perturbations
    - Measurement noise: randomly zeros out elements (simulates measurement collapse)
    - Entanglement noise: correlates noise across dimensions
    
    Args:
        tensor: Input tensor
        noise_prob: Probability of applying noise
        noise_scale: Scale of the noise
        noise_type: "phase_only", "amplitude_only", "phase_and_amplitude", 
                   "measurement", "entanglement"
    """
    if noise_scale <= 0:
        return tensor
    
    # Skip noise for non-training
    if not tensor.requires_grad:
        return tensor
        
    if noise_type == "phase_only":
        # Only flip signs (phase noise)
        mask = (torch.rand_like(tensor) < noise_prob).float()
        flip = torch.where(mask > 0, -1.0, 1.0)
        return tensor * flip
    
    elif noise_type == "amplitude_only":
        # Only add amplitude noise
        mask = (torch.rand_like(tensor) < noise_prob).float()
        noise = torch.randn_like(tensor) * noise_scale * mask
        return tensor + noise
    
    elif noise_type == "measurement":
        # Simulate measurement: randomly zero out elements
        mask = (torch.rand_like(tensor) < noise_prob).float()
        return tensor * (1 - mask)
    
    elif noise_type == "entanglement":
        # Correlated noise across dimensions (to simulate entanglement)
        if len(tensor.shape) <= 1:
            return add_quantum_noise(tensor, noise_prob, noise_scale, "phase_and_amplitude")
            
        # Generate correlated noise for each slice
        shape = tensor.shape
        reshaped = tensor.view(shape[0], -1)
        noise_matrix = torch.randn(reshaped.size(1), reshaped.size(1)).to(tensor.device) * noise_scale
        # Ensure it's symmetric for valid correlation
        noise_matrix = (noise_matrix + noise_matrix.T) / 2
        # Apply correlated noise selectively
        mask = (torch.rand(shape[0]) < noise_prob).float().to(tensor.device)
        noise = torch.matmul(mask.unsqueeze(1) * torch.randn(shape[0], 1).to(tensor.device), 
                             torch.ones(1, reshaped.size(1)).to(tensor.device))
        noise = torch.matmul(noise, noise_matrix)
        return tensor + noise.view(shape)
    
    else:  # Default: phase_and_amplitude
        # Both phase and amplitude noise
        phase_mask = (torch.rand_like(tensor) < noise_prob).float()
        amp_mask = (torch.rand_like(tensor) < noise_prob).float()
        
        flip = torch.where(phase_mask > 0, -1.0, 1.0)
        noise = torch.randn_like(tensor) * noise_scale * amp_mask
        
        return tensor * flip + noise

def compute_von_neumann_entropy(attn_weights, average_heads=True, normalized=True):
    """
    Compute a quantum-inspired metric: the von Neumann entropy of the attention weights.
    
    Args:
        attn_weights: Attention weight matrix of shape [batch, heads, seq_len, seq_len]
        average_heads: Whether to average over attention heads
        normalized: Whether to normalize by maximum possible entropy
        
    Returns:
        Entropy value(s)
    """
    eps = 1e-8
    batch_size, num_heads = attn_weights.shape[0], attn_weights.shape[1]
    
    if average_heads:
        # For each example, average over heads
        entropies = torch.zeros(batch_size, device=attn_weights.device)
        
        for i in range(batch_size):
            if normalized:
                # Make sure it's a valid density matrix (symmetric, trace=1)
                density = torch.mean(attn_weights[i], dim=0)  # Average over heads
                # Normalize by trace
                density = density / (torch.trace(density) + eps)
            else:
                density = torch.mean(attn_weights[i], dim=0)
                
            # Compute eigenvalues
            try:
                eigvals = torch.linalg.eigvalsh(density)  # Use eigvalsh for symmetric matrices
                # Eliminate negative eigenvalues (numerical errors)
                eigvals = eigvals.clamp(min=eps)
                # Compute entropy
                entropies[i] = -torch.sum(eigvals * torch.log(eigvals))
            except:
                # Fallback to diagonal elements if eigendecomposition fails
                diag_elements = torch.diagonal(density).clamp(min=eps)
                entropies[i] = -torch.sum(diag_elements * torch.log(diag_elements))
                
        # Normalize by log(dimension) if requested
        if normalized:
            max_entropy = torch.log(torch.tensor(attn_weights.size(-1), dtype=torch.float, 
                                                 device=attn_weights.device))
            entropies = entropies / max_entropy
            
        return entropies.mean()
    
    else:
        # Return entropy per head
        entropies = torch.zeros(batch_size, num_heads, device=attn_weights.device)
        
        for i in range(batch_size):
            for h in range(num_heads):
                if normalized:
                    density = attn_weights[i, h]
                    density = density / (torch.trace(density) + eps)
                else:
                    density = attn_weights[i, h]
                    
                try:
                    eigvals = torch.linalg.eigvalsh(density)
                    eigvals = eigvals.clamp(min=eps)
                    entropies[i, h] = -torch.sum(eigvals * torch.log(eigvals))
                except:
                    diag_elements = torch.diagonal(density).clamp(min=eps)
                    entropies[i, h] = -torch.sum(diag_elements * torch.log(diag_elements))
                    
        if normalized:
            max_entropy = torch.log(torch.tensor(attn_weights.size(-1), dtype=torch.float, 
                                                 device=attn_weights.device))
            entropies = entropies / max_entropy
            
        return entropies.mean(dim=0)  # Average over batch

def compute_coherence_metric(real_tensor, imag_tensor, method="cosine_similarity"):
    """
    Computes a coherence metric between real and imaginary components.
    
    Args:
        real_tensor: Real part of the representation
        imag_tensor: Imaginary part of the representation
        method: "cosine_similarity", "mutual_information", or "correlation"
        
    Returns:
        Coherence score: 0 means no coherence, 1 means perfect coherence
    """
    if method == "cosine_similarity":
        real_norm = F.normalize(real_tensor, p=2, dim=-1)
        imag_norm = F.normalize(imag_tensor, p=2, dim=-1)
        similarity = torch.abs(torch.sum(real_norm * imag_norm, dim=-1))
        return similarity.mean()
        
    elif method == "correlation":
        # Compute Pearson correlation
        real_flat = real_tensor.view(-1)
        imag_flat = imag_tensor.view(-1)
        
        real_mean = real_flat.mean()
        imag_mean = imag_flat.mean()
        
        real_std = real_flat.std() + 1e-8
        imag_std = imag_flat.std() + 1e-8
        
        correlation = torch.mean((real_flat - real_mean) * (imag_flat - imag_mean)) / (real_std * imag_std)
        return torch.abs(correlation)
        
    else:  # Default to simple magnitude ratio
        real_magnitude = torch.norm(real_tensor, p=2)
        imag_magnitude = torch.norm(imag_tensor, p=2)
        ratio = torch.min(real_magnitude, imag_magnitude) / torch.max(real_magnitude, imag_magnitude)
        return ratio

def plot_attention_heatmap(attention_weights, filename="attention_heatmap.png", head_idx=0):
    """
    Plots and saves attention weight heatmaps for visualizing the model's focus.
    
    Args:
        attention_weights: Tensor of shape [batch, heads, seq_len, seq_len]
        filename: Output file name
        head_idx: Which attention head to plot
    """
    plt.figure(figsize=(10, 8))
    
    # Select first example and specified head
    attn = attention_weights[0, head_idx].cpu().numpy()
    
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title(f"Attention Weights (Head {head_idx})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    
    # Save the figure
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(os.path.join("visualizations", filename))
    plt.close()

def compute_metrics(predictions, targets):
    """
    Computes various metrics for model evaluation.
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    # Convert to class predictions if needed
    if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
        predictions = torch.argmax(predictions, dim=-1)
    
    # Flatten if needed
    predictions_flat = predictions.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(targets_flat, predictions_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_flat, predictions_flat, average='macro', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def gradient_entropy(model):
    """
    Computes the entropy of gradients across model parameters
    as a metric for learning dynamics.
    
    Returns:
        Entropy value
    """
    grad_entropy = 0.0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Normalize gradients
            grad_flat = param.grad.view(-1).abs().detach().cpu().numpy()
            if len(grad_flat) > 0:
                total_params += len(grad_flat)
                # Bin the gradients and compute entropy
                try:
                    hist, _ = np.histogram(grad_flat, bins=20, density=True)
                    hist = hist[hist > 0]  # Remove zeros
                    grad_entropy += entropy(hist)
                except:
                    continue
    
    if total_params > 0:
        return grad_entropy / (total_params ** 0.5)  # Scale by sqrt of total params
    else:
        return 0.0

###############################################################################
#                        MODEL COMPONENTS                                     #
###############################################################################

class ComplexEmbedding(nn.Module):
    """
    Complex-valued embedding with improved initialization and adaptive scaling.
    """
    def __init__(self, vocab_size, embedding_dim, scale=0.02, adaptive_scale=True):
        super().__init__()
        self.real_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.imag_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.adaptive_scale = adaptive_scale
        
        # Initialize with careful scaling
        nn.init.normal_(self.real_embedding.weight, mean=0.0, std=scale)
        nn.init.normal_(self.imag_embedding.weight, mean=0.0, std=scale * 0.5)
        
        # Optional learnable scaling factors
        if adaptive_scale:
            self.real_scale = nn.Parameter(torch.ones(1))
            self.imag_scale = nn.Parameter(torch.ones(1) * 0.5)
            # Optional frequency component for token-specific phase shifting
            self.frequency = nn.Parameter(torch.ones(embedding_dim) * 0.1)
        
    def forward(self, x):
        real = self.real_embedding(x)
        imag = self.imag_embedding(x)
        
        if self.adaptive_scale:
            # Apply learnable scaling
            real = real * self.real_scale
            imag = imag * self.imag_scale
            
            # Optional: Apply token-specific phase shifting
            # This simulates different frequency components for different tokens
            phases = torch.outer(
                torch.arange(x.size(1), device=x.device).float(),
                self.frequency
            ).unsqueeze(0)
            
            # Apply phase rotation in complex plane
            old_real, old_imag = real.clone(), imag.clone()
            real = old_real * torch.cos(phases) - old_imag * torch.sin(phases)
            imag = old_real * torch.sin(phases) + old_imag * torch.cos(phases)
        
        return real, imag

class PositionalEncoding(nn.Module):
    """
    Complex-valued positional encoding using sine/cosine functions
    with improved phase relations and learnable components.
    """
    def __init__(self, dim, max_len=5000, phase_shift=True, learnable=True):
        super().__init__()
        self.dim = dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        
        # Base positional encodings
        pe_real = torch.zeros(max_len, dim)
        pe_real[:, 0::2] = torch.sin(position * div_term)
        pe_real[:, 1::2] = torch.cos(position * div_term)
        
        pe_imag = torch.zeros(max_len, dim)
        if phase_shift:
            pe_imag[:, 0::2] = torch.cos(position * div_term)
            pe_imag[:, 1::2] = -torch.sin(position * div_term)
        else:
            pe_imag[:, 0::2] = torch.sin(position * div_term + math.pi/4)
            pe_imag[:, 1::2] = torch.cos(position * div_term + math.pi/4)
            
        self.register_buffer('pe_real_base', pe_real)
        self.register_buffer('pe_imag_base', pe_imag)
        
        # Optional learnable components
        self.learnable = learnable
        if learnable:
            # Learnable scaling and shift factors
            self.real_scale = nn.Parameter(torch.ones(1, dim))
            self.imag_scale = nn.Parameter(torch.ones(1, dim))
            self.real_shift = nn.Parameter(torch.zeros(1, dim))
            self.imag_shift = nn.Parameter(torch.zeros(1, dim))
            
            # Learnable frequency factors for each dimension
            self.frequency_factors = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        real, imag = x
        seq_len = real.size(1)
        
        if self.learnable:
            # Apply learnable transformations to base positional encodings
            position_indices = torch.arange(seq_len, device=real.device).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.dim, 2, device=real.device).float() * 
                                (-math.log(10000.0) / self.dim))
            
            # Apply frequency modulation
            div_term = div_term * self.frequency_factors[0::2]
            
            # Compute actual positional encodings
            pe_real = torch.zeros(seq_len, self.dim, device=real.device)
            pe_real[:, 0::2] = torch.sin(position_indices * div_term)
            pe_real[:, 1::2] = torch.cos(position_indices * div_term * 
                                         self.frequency_factors[1::2])
            
            pe_imag = torch.zeros(seq_len, self.dim, device=real.device)
            pe_imag[:, 0::2] = torch.cos(position_indices * div_term)
            pe_imag[:, 1::2] = -torch.sin(position_indices * div_term * 
                                          self.frequency_factors[1::2])
            
            # Apply learnable scaling and shifting
            pe_real = pe_real * self.real_scale + self.real_shift
            pe_imag = pe_imag * self.imag_scale + self.imag_shift
            
            return real + pe_real, imag + pe_imag
        else:
            # Use base positional encodings
            return real + self.pe_real_base[:seq_len, :], imag + self.pe_imag_base[:seq_len, :]

class ComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex inputs with learnable complex coupling.
    """
    def __init__(self, dim, eps=1e-5, coupled=True):
        super().__init__()
        self.real_norm = nn.LayerNorm(dim, eps=eps)
        self.imag_norm = nn.LayerNorm(dim, eps=eps)
        self.coupled = coupled
        
        if coupled:
            # Learnable coupling parameter
            self.coupling = nn.Parameter(torch.tensor(0.1))
            # Cross-component gain and bias
            self.cross_gain_ri = nn.Parameter(torch.zeros(dim))
            self.cross_gain_ir = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        real, imag = x
        
        if self.coupled:
            # Compute standard normalization
            real_normed = self.real_norm(real)
            imag_normed = self.imag_norm(imag)
            
            # Apply cross-component coupling (approximating complex normalization)
            coupling = torch.sigmoid(self.coupling)  # Keep it in [0, 1]
            
            # Mix normalized components
            real_out = real_normed + coupling * self.cross_gain_ri * imag_normed
            imag_out = imag_normed + coupling * self.cross_gain_ir * real_normed
            
            return real_out, imag_out
        else:
            # Apply normalization separately
            return self.real_norm(real), self.imag_norm(imag)

class EntangledInterferenceLayer(nn.Module):
    """
    Enhanced interference layer with quantum entanglement between attention heads,
    adaptive interference strength, and quantum noise injection.
    """
    def __init__(self, dim, heads=8, dropout=0.1, interference_type="quantum",
                 use_entanglement=True, noise_scale=0.1, return_attention=True,
                 use_rotary=True, adaptive_attention=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.interference_type = interference_type
        self.use_entanglement = use_entanglement
        self.noise_scale = noise_scale
        self.return_attention = return_attention
        self.use_rotary = use_rotary
        self.adaptive_attention = adaptive_attention
        
        # Phase shifts and entanglement parameters:
        self.phase_shifts = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        
        # Entanglement matrix couples pairs of heads
        if use_entanglement:
            # Initialize with a sparse structure to encourage specialization
            entangle_init = torch.eye(heads)
            # Add some off-diagonal elements
            offdiag_mask = torch.rand(heads, heads) < 0.2
            offdiag_values = torch.randn(heads, heads) * 0.1
            entangle_init = entangle_init + offdiag_mask * offdiag_values
            self.entanglement_matrix = nn.Parameter(entangle_init)
        else:
            self.register_buffer('entanglement_matrix', torch.eye(heads))
            
        # Query, Key, Value projections for real and imaginary parts
        self.q_real = nn.Linear(dim, dim)
        self.k_real = nn.Linear(dim, dim)
        self.v_real = nn.Linear(dim, dim)
        self.q_imag = nn.Linear(dim, dim)
        self.k_imag = nn.Linear(dim, dim)
        self.v_imag = nn.Linear(dim, dim)
        
        # Output projections
        self.out_real = nn.Linear(dim, dim)
        self.out_imag = nn.Linear(dim, dim)
        
        # Rotary embeddings
        if use_rotary:
            self.rotary_dim = min(self.head_dim, 32)  # Limit rotary dimension
            # Use base frequencies as in original RoPE
            self.rotary_freqs = 10000.0 ** (-torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
            
            # Make these learnable for adaptive behavior
            if adaptive_attention:
                self.rotary_freqs = nn.Parameter(self.rotary_freqs)
        
        # Collapse gate for controlling superposition
        self.collapse_gate = nn.Linear(dim, dim)
        
        # Adaptive interference strength via parameterized gating
        self.interference_strength = nn.Parameter(torch.ones(1))
        
        # Optional: attention temperature parameter
        if adaptive_attention:
            self.attention_temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('attention_temperature', torch.ones(1))
            
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Attention bias for relative position emphasis
        seq_len = 512  # Maximum expected sequence length
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        )
        self.register_buffer('masked_bias', torch.tensor(-1e4))
        
    def _apply_rotary_pos_emb(self, q, k, seq_len):
        """Apply rotary position embeddings to queries and keys"""
        if not self.use_rotary:
            return q, k
            
        # Only apply to a subset of dimensions
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        
        # Get position indices
        position = torch.arange(seq_len, device=q.device).float().unsqueeze(1)
        
        # Compute sin and cos for position encoding
        freqs = self.rotary_freqs.to(q.device)
        sincos_dim = self.rotary_dim // 2
        
        sin = torch.sin(position * freqs[:sincos_dim])
        cos = torch.cos(position * freqs[:sincos_dim])
        
        # Reshape for easier rotation: separate each pair of dimensions
        q_rot2 = q_rot.view(*q_rot.shape[:-1], -1, 2)
        k_rot2 = k_rot.view(*k_rot.shape[:-1], -1, 2)
        
        sin = sin.view(seq_len, 1, 1, -1)
        cos = cos.view(seq_len, 1, 1, -1)
        
        q_rot2_transformed = torch.cat([
            q_rot2[..., 0:1] * cos - q_rot2[..., 1:2] * sin,
            q_rot2[..., 0:1] * sin + q_rot2[..., 1:2] * cos
        ], dim=-1)
        
        k_rot2_transformed = torch.cat([
            k_rot2[..., 0:1] * cos - k_rot2[..., 1:2] * sin,
            k_rot2[..., 0:1] * sin + k_rot2[..., 1:2] * cos
        ], dim=-1)
        
        q_rot = q_rot2_transformed.view(*q_rot.shape)
        k_rot = k_rot2_transformed.view(*k_rot.shape)
        
        q_out = torch.cat([q_rot, q_pass], dim=-1)
        k_out = torch.cat([k_rot, k_pass], dim=-1)
        
        return q_out, k_out
        
    def forward(self, x, mask=None, layer_past=None, return_present=False):
        real, imag = x  # (batch, seq_len, dim)
        batch_size, seq_len, _ = real.shape
        
        # Project to Q, K, V for both components
        q_r = self.q_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        k_r = self.k_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        v_r = self.v_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        q_i = self.q_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        k_i = self.k_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        v_i = self.v_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        
        # Apply quantum noise injection on Q and K
        q_r = add_quantum_noise(q_r, noise_scale=self.noise_scale)
        q_i = add_quantum_noise(q_i, noise_scale=self.noise_scale)
        k_r = add_quantum_noise(k_r, noise_scale=self.noise_scale)
        k_i = add_quantum_noise(k_i, noise_scale=self.noise_scale)
        
        # Apply rotary position embeddings if enabled
        q_r, k_r = self._apply_rotary_pos_emb(q_r, k_r, seq_len)
        q_i, k_i = self._apply_rotary_pos_emb(q_i, k_i, seq_len)
        
        # Entangle attention heads using the entanglement matrix:
        if self.use_entanglement:
            q_r = torch.einsum("bshd,hh->bshd", q_r, self.entanglement_matrix)
            q_i = torch.einsum("bshd,hh->bshd", q_i, self.entanglement_matrix)
            k_r = torch.einsum("bshd,hh->bshd", k_r, self.entanglement_matrix)
            k_i = torch.einsum("bshd,hh->bshd", k_i, self.entanglement_matrix)
            
        # triple check this area as generation was resumed here from claude 3.7
        # Apply phase shifts (rotate in complex plane)
        phase_cos = torch.cos(self.phase_shifts).unsqueeze(0).unsqueeze(0)
        phase_sin = torch.sin(self.phase_shifts).unsqueeze(0).unsqueeze(0)
        q_r_shifted = q_r * phase_cos - q_i * phase_sin
        q_i_shifted = q_r * phase_sin + q_i * phase_cos
        k_r_shifted = k_r * phase_cos - k_i * phase_sin
        k_i_shifted = k_r * phase_sin + k_i * phase_cos
        
        q_r = q_r_shifted.transpose(1, 2)
        q_i = q_i_shifted.transpose(1, 2)
        k_r = k_r_shifted.transpose(1, 2)
        k_i = k_i_shifted.transpose(1, 2)
        v_r = v_r.transpose(1, 2)
        v_i = v_i.transpose(1, 2)
        
        present = torch.stack((k_r, v_r, k_i, v_i))
        
        if layer_past is not None:
            past_k_r, past_v_r, past_k_i, past_v_i = layer_past
            k_r = torch.cat((past_k_r, k_r), dim=-2)
            v_r = torch.cat((past_v_r, v_r), dim=-2)
            k_i = torch.cat((past_k_i, k_i), dim=-2)
            v_i = torch.cat((past_v_i, v_i), dim=-2)
            
        scale = 1.0 / math.sqrt(self.head_dim)
        
        if self.interference_type == "quantum":
            attn_r = torch.matmul(q_r, k_r.transpose(-2, -1)) + torch.matmul(q_i, k_i.transpose(-2, -1))
            attn_i = torch.matmul(q_i, k_r.transpose(-2, -1)) - torch.matmul(q_r, k_i.transpose(-2, -1))
            attn_r = attn_r * scale
            attn_i = attn_i * scale
            
            attn_mag = torch.sqrt(attn_r**2 + attn_i**2 + 1e-6)
            
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(1)
                attn_mag = attn_mag.masked_fill(mask == 0, -1e9)
                
            if self.adaptive_attention:
                attn_mag = attn_mag / self.attention_temperature
                
            attn_weights = F.softmax(attn_mag * torch.sigmoid(self.interference_strength), dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            if self.return_attention:
                self.saved_attn_weights = attn_weights.detach()
                
            out_r = torch.matmul(attn_weights, v_r)
            out_i = torch.matmul(attn_weights, v_i)
            
        else:
            attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
            
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(1)
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
                
            if self.adaptive_attention:
                attn_scores = attn_scores / self.attention_temperature
                
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            if self.return_attention:
                self.saved_attn_weights = attn_weights.detach()
                
            out_r = torch.matmul(attn_weights, v_r)
            out_i = torch.matmul(attn_weights, v_i)
        
        out_r = out_r.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out_i = out_i.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        
        if hasattr(self, 'collapse_alpha'):
            alpha = self.collapse_alpha  # value in [0, 1]
        else:
            alpha = 0.5  # default
        
        collapse_factor = torch.sigmoid(self.collapse_gate(real))
        out_r = out_r * (alpha * collapse_factor + (1 - alpha))
        out_i = out_i * ((1 - alpha) * (1 - collapse_factor) + alpha)
        
        out_r = self.out_real(out_r)
        out_i = self.out_imag(out_i)
        out_r = self.resid_dropout(out_r)
        out_i = self.resid_dropout(out_i)
        
        if return_present:
            return (out_r, out_i), present
        else:
            return (out_r, out_i)

class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion module to combine information from different modalities.
    This enables the model to work with multi-modal inputs, even if only trained on text.
    """
    def __init__(self, text_dim, other_dim, fusion_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.other_dim = other_dim
        self.fusion_dim = fusion_dim
        
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.other_proj = nn.Linear(other_dim, fusion_dim)
        
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout)
        self.text_norm = nn.LayerNorm(fusion_dim)
        self.other_norm = nn.LayerNorm(fusion_dim)
        
        self.fusion_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        
        self.output_proj = nn.Linear(fusion_dim, text_dim)
        
    def forward(self, text_features, other_features, text_mask=None, other_mask=None):
        batch_size, seq_len, _ = text_features.shape
        _, other_seq_len, _ = other_features.shape
        
        text_proj = self.text_proj(text_features)  # [batch, seq_len, fusion_dim]
        other_proj = self.other_proj(other_features)  # [batch, other_seq_len, fusion_dim]
        
        if text_mask is not None:
            text_key_padding_mask = ~text_mask.bool()
        else:
            text_key_padding_mask = None
            
        if other_mask is not None:
            other_key_padding_mask = ~other_mask.bool()
        else:
            other_key_padding_mask = None
            
        text_proj_norm = self.text_norm(text_proj)
        other_proj_norm = self.other_norm(other_proj)
        
        text_proj_t = text_proj_norm.transpose(0, 1)
        other_proj_t = other_proj_norm.transpose(0, 1)
        
        attn_output, _ = self.cross_attn(
            query=text_proj_t,
            key=other_proj_t,
            value=other_proj_t,
            key_padding_mask=other_key_padding_mask
        )
        
        attn_output = attn_output.transpose(0, 1)
        
        gate_input = torch.cat([text_proj, attn_output], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(gate_input))
        
        fused = text_proj * (1 - gate) + attn_output * gate
        
        output = self.output_proj(fused)
        
        return output

class AdvancedWaveFunctionCollapse(nn.Module):
    """
    Advanced collapse mechanism with curriculum learning support
    and multiple collapse strategies.
    """
    def __init__(self, dim, vocab_size, collapse_type="squared_magnitude", 
                 use_mixtures=True, num_mixtures=2):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.collapse_type = collapse_type
        self.use_mixtures = use_mixtures
        self.num_mixtures = num_mixtures
        
        self.real_collapse = nn.Linear(dim, vocab_size)
        self.imag_collapse = nn.Linear(dim, vocab_size)
        self.interference_collapse = nn.Linear(dim, vocab_size)
        
        if collapse_type == "entanglement":
            rank = min(dim // 4, 64)
            self.entangle_u = nn.Parameter(torch.randn(dim, rank) * 0.02)
            self.entangle_v = nn.Parameter(torch.randn(rank, dim) * 0.02)
            
        if use_mixtures:
            self.mixture_weights = nn.Linear(dim, num_mixtures)
            self.mixture_components = nn.ModuleList([
                nn.Linear(dim, vocab_size) for _ in range(num_mixtures)
            ])
        
        self.register_buffer('real_scale', torch.ones(1))
        self.register_buffer('imag_scale', torch.ones(1))
        self.register_buffer('interference_scale', torch.ones(1))
        
    def forward(self, x):
        real, imag = x
        batch_size, seq_len, _ = real.shape
        
        real_logits = self.real_collapse(real) * self.real_scale
        imag_logits = self.imag_collapse(imag) * self.imag_scale
        
        if self.collapse_type == "squared_magnitude":
            density = real_logits**2 + imag_logits**2
            
        elif self.collapse_type == "interference":
            interference_term = self.interference_collapse(real * imag) * self.interference_scale
            density = real_logits**2 + imag_logits**2 + interference_term
            
        elif self.collapse_type == "entanglement":
            if hasattr(self, 'entangle_u'):
                entangled = torch.matmul(real, torch.matmul(self.entangle_u, self.entangle_v))
            else:
                entangled = torch.matmul(real, self.entanglement_weights)
                
            entanglement_term = torch.sum(entangled * imag, dim=-1, keepdim=True)
            density = real_logits**2 + imag_logits**2 + entanglement_term
            
        elif self.collapse_type == "born_rule":
            amplitude = torch.complex(real_logits, imag_logits)
            density = torch.abs(amplitude)**2
            
        else:  # "simple"
            density = real_logits + imag_logits
        
        if self.use_mixtures:
            mix_weights = F.softmax(self.mixture_weights(real), dim=-1)
            
            mixture_logits = 0
            for i, component in enumerate(self.mixture_components):
                component_logits = component(real + imag)
                component_weight = mix_weights[:, :, i:i+1]
                mixture_logits += component_weight * component_logits
                
            if hasattr(self, 'mixture_alpha'):
                alpha = self.mixture_alpha  # In [0, 1]
            else:
                alpha = 0.5
                
            density = (1 - alpha) * density + alpha * mixture_logits
        
        return F.relu(density) + 1e-10

class MemoryModule(nn.Module):
    """
    Enhanced memory module with attention-based read/write operations,
    inspired by Neural Turing Machines and Differentiable Neural Computers.
    """
    def __init__(self, mem_size, input_dim, mem_dim=None, num_heads=1):
        super().__init__()
        self.mem_size = mem_size
        self.input_dim = input_dim
        self.mem_dim = mem_dim if mem_dim is not None else input_dim
        self.num_heads = num_heads
        
        self.register_buffer('has_been_initialized', torch.zeros(1, dtype=torch.bool))
        self.register_parameter('memory', nn.Parameter(torch.randn(mem_size, self.mem_dim) * 0.01))
        
        self.register_buffer('usage', torch.zeros(mem_size))
        
        self.input_proj = nn.Linear(input_dim, self.mem_dim)
        
        self.query_gen = nn.Linear(input_dim, self.mem_dim * num_heads)
        
        self.erase_gen = nn.Linear(input_dim, self.mem_dim)
        self.write_gen = nn.Linear(input_dim, self.mem_dim)
        self.read_proj = nn.Linear(self.mem_dim, input_dim)
        
        self.memory_gate = nn.Linear(input_dim + self.mem_dim, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.memory, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        for layer in [self.query_gen, self.erase_gen, self.write_gen, self.read_proj, self.memory_gate]:
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
    
    def _address_memory(self, query):
        batch_size = query.size(0)
        query = self.query_gen(query).view(batch_size, self.num_heads, self.mem_dim)
        
        query_norm = F.normalize(query, dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        
        scores = torch.matmul(query_norm, memory_norm.t())
        
        weights = F.softmax(scores, dim=-1)
        
        return weights
    
    def read(self, query):
        batch_size = query.size(0)
        
        weights = self._address_memory(query)  # [batch, heads, mem_size]
        
        read_values = torch.matmul(weights, self.memory)
        
        read_values = read_values.view(batch_size, -1)[:, :self.mem_dim]
        output = self.read_proj(read_values)
        
        gate_input = torch.cat([query, read_values], dim=-1)
        gate = torch.sigmoid(self.memory_gate(gate_input))
        gated_output = output * gate
        
        return gated_output
    
    def write(self, input_values):
        batch_size = input_values.size(0)
        
        input_proj = self.input_proj(input_values)
        
        write_weights = self._address_memory(input_values)  # [batch, heads, mem_size]
        
        write_weights = write_weights.max(dim=1)[0]  # [batch, mem_size]
        
        erase_vec = torch.sigmoid(self.erase_gen(input_values))  # [batch, mem_dim]
        write_vec = self.write_gen(input_values)  # [batch, mem_dim]
        
        if batch_size > 0:
            w = write_weights[0].unsqueeze(1)  # [mem_size, 1]
            e = erase_vec[0].unsqueeze(0)  # [1, mem_dim]
            a = write_vec[0].unsqueeze(0)  # [1, mem_dim]
            
            memory_erased = self.memory * (1 - torch.matmul(w, e))
            memory_written = memory_erased + torch.matmul(w, a)
            
            self.usage = torch.clamp(self.usage + write_weights[0], max=1.0)
            
            self.memory.data = memory_written
        
        return input_proj
    
    def reset_memory(self):
        nn.init.kaiming_uniform_(self.memory, a=math.sqrt(5))
        self.usage.zero_()

class HierarchicalInterferenceModule(nn.Module):
    """
    Implements multi-scale interference at different hierarchical levels.
    """
    def __init__(self, dim, heads=8, depth=2, dropout=0.1, interference_type="quantum"):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        
        self.word_scale = nn.ModuleList([
            EntangledInterferenceLayer(
                dim, heads, dropout, interference_type, 
                noise_scale=0.1 / (i+1)
            ) 
            for i in range(depth)
        ])
        
        self.phrase_scale = nn.ModuleList([
            EntangledInterferenceLayer(
                dim, heads, dropout, interference_type,
                noise_scale=0.05 / (i+1)
            )
            for i in range(depth)
        ])
        
        self.sentence_scale = nn.ModuleList([
            EntangledInterferenceLayer(
                dim, heads, dropout, interference_type,
                noise_scale=0.02 / (i+1)
            )
            for i in range(max(1, depth//2))
        ])
        
        self.word_norm = nn.ModuleList([ComplexLayerNorm(dim) for _ in range(depth)])
        self.phrase_norm = nn.ModuleList([ComplexLayerNorm(dim) for _ in range(depth)])
        self.sentence_norm = nn.ModuleList([ComplexLayerNorm(dim) for _ in range(max(1, depth//2))])
        
        self.word_to_phrase = nn.Linear(dim, dim)
        self.phrase_to_sentence = nn.Linear(dim, dim)
        self.sentence_to_output = nn.Linear(dim, dim)
        
        self.scale_fusion = nn.Linear(dim * 3, 3)
        
    def _pool_representations(self, x, window_size, stride=None):
        real, imag = x
        batch_size, seq_len, dim = real.shape
        
        if stride is None:
            stride = window_size
            
        pool_len = (seq_len + stride - 1) // stride
        
        pooled_real = torch.zeros(batch_size, pool_len, dim, device=real.device)
        pooled_imag = torch.zeros(batch_size, pool_len, dim, device=imag.device)
        
        for i in range(0, seq_len, stride):
            end = min(i + window_size, seq_len)
            idx = i // stride
            if idx < pool_len:
                pooled_real[:, idx] = real[:, i:end].mean(dim=1)
                pooled_imag[:, idx] = imag[:, i:end].mean(dim=1)
                
        return pooled_real, pooled_imag
    
    def _unpool_representations(self, x, target_len):
        real, imag = x
        batch_size, pooled_len, dim = real.shape
        
        unpooled_real = torch.zeros(batch_size, target_len, dim, device=real.device)
        unpooled_imag = torch.zeros(batch_size, target_len, dim, device=imag.device)
        
        ratio = pooled_len / target_len
        
        for i in range(target_len):
            src_idx = min(int(i * ratio), pooled_len - 1)
            unpooled_real[:, i] = real[:, src_idx]
            unpooled_imag[:, i] = imag[:, src_idx]
            
        return unpooled_real, unpooled_imag
        
    def forward(self, x):
        real, imag = x
        batch_size, seq_len, _ = real.shape
        
        word_real, word_imag = real, imag
        for layer, norm in zip(self.word_scale, self.word_norm):
            normed = norm((word_real, word_imag))
            out_real, out_imag = layer(normed)
            word_real = word_real + out_real
            word_imag = word_imag + out_imag
            
        phrase_window = min(4, seq_len)
        phrase_stride = max(1, phrase_window // 2)
        phrase_real, phrase_imag = self._pool_representations(
            (word_real, word_imag), phrase_window, phrase_stride
        )
        
        for layer, norm in zip(self.phrase_scale, self.phrase_norm):
            normed = norm((phrase_real, phrase_imag))
            out_real, out_imag = layer(normed)
            phrase_real = phrase_real + out_real
            phrase_imag = phrase_imag + out_imag
            
        sent_window = min(4, phrase_real.size(1))
        sent_stride = max(1, sent_window // 2)
        sent_real, sent_imag = self._pool_representations(
            (phrase_real, phrase_imag), sent_window, sent_stride
        )
        
        for layer, norm in zip(self.sentence_scale, self.sentence_norm):
            normed = norm((sent_real, sent_imag))
            out_real, out_imag = layer(normed)
            sent_real = sent_real + out_real
            sent_imag = sent_imag + out_imag
            
        phrase_real_exp, phrase_imag_exp = self._unpool_representations(
            (phrase_real, phrase_imag), seq_len
        )
        sent_real_exp, sent_imag_exp = self._unpool_representations(
            (sent_real, sent_imag), seq_len
        )
        
        word_feat_real = word_real
        word_feat_imag = word_imag
        
        phrase_feat_real = self.word_to_phrase(phrase_real_exp)
        phrase_feat_imag = self.word_to_phrase(phrase_imag_exp)
        
        sent_feat_real = self.phrase_to_sentence(sent_real_exp)
        sent_feat_imag = self.phrase_to_sentence(sent_imag_exp)
        
        fusion_input = torch.cat([
            word_feat_real, phrase_feat_real, sent_feat_real
        ], dim=-1)
        
        scale_weights = F.softmax(self.scale_fusion(fusion_input), dim=-1)
        
        fused_real = (
            scale_weights[:, :, 0:1] * word_feat_real +
            scale_weights[:, :, 1:2] * phrase_feat_real +
            scale_weights[:, :, 2:3] * sent_feat_real
        )
        
        fused_imag = (
            scale_weights[:, :, 0:1] * word_feat_imag +
            scale_weights[:, :, 1:2] * phrase_feat_imag +
            scale_weights[:, :, 2:3] * sent_feat_imag
        )
        
        output_real = self.sentence_to_output(fused_real)
        output_imag = self.sentence_to_output(fused_imag)
        
        return output_real, output_imag

class AdvancedSFIN(nn.Module):
    """
    Advanced SFIN integrating multi-scale interference, external memory,
    and cross-modal capabilities.
    """
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, dropout=0.1,
                 interference_type="quantum", collapse_type="squared_magnitude",
                 max_seq_len=256, mem_size=32, use_hierarchical=True,
                 use_cross_modal=False, other_dim=None):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.use_hierarchical = use_hierarchical
        self.use_cross_modal = use_cross_modal
        
        self.embedding = ComplexEmbedding(vocab_size, dim)
        self.pos_encoding = PositionalEncoding(dim, max_len=max_seq_len, learnable=True)
        
        if use_hierarchical:
            self.hierarchical_module = HierarchicalInterferenceModule(
                dim, heads, depth // 3, dropout, interference_type
            )
        else:
            self.interference_layers_word = nn.ModuleList([
                EntangledInterferenceLayer(dim, heads, dropout, interference_type)
                for _ in range(depth//2)
            ])
            self.interference_layers_phrase = nn.ModuleList([
                EntangledInterferenceLayer(dim, heads, dropout, interference_type)
                for _ in range(depth - depth//2)
            ])
            self.norm_layers_word = nn.ModuleList([ComplexLayerNorm(dim) for _ in range(depth//2)])
            self.norm_layers_phrase = nn.ModuleList([ComplexLayerNorm(dim) for _ in range(depth - depth//2)])
        
        if use_cross_modal and other_dim is not None:
            self.cross_modal_fusion = CrossModalFusion(
                text_dim=dim, 
                other_dim=other_dim,
                fusion_dim=dim,
                num_heads=heads//2
            )
        
        self.final_norm = ComplexLayerNorm(dim)
        self.collapse = AdvancedWaveFunctionCollapse(dim, vocab_size, collapse_type)
        self.dropout = nn.Dropout(dropout)
        
        self.memory = MemoryModule(mem_size, dim)
        
        # For curriculum learning on collapse
        self.register_buffer("collapse_alpha", torch.tensor(0.0))
        
        # Learning rate multipliers for different components
        self.register_buffer("embed_lr_multiplier", torch.tensor(1.0))
        self.register_buffer("mem_lr_multiplier", torch.tensor(1.0))
        self.register_buffer("attn_lr_multiplier", torch.tensor(1.0))
       
        # Explainability hooks
        self.explainability_mode = config.get("explainability_mode", "all")
        self.saved_attention_maps = {}
        self.saved_entropies = {}
       
        self._init_weights()
       
    def _init_weights(self):
        """Initialize model weights with carefully chosen distributions"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "out" in name or "collapse" in name:
                    nn.init.normal_(module.weight, 0.0, 0.01)
                else:
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
               
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
            if "entanglement_matrix" in name:
                nn.init.orthogonal_(module)
               
    def get_attention_maps(self):
        """Collect attention maps from all interference layers"""
        attention_maps = {}
       
        if self.use_hierarchical:
            for i, layer in enumerate(self.hierarchical_module.word_scale):
                if hasattr(layer, 'saved_attn_weights'):
                    attention_maps[f"word_scale_{i}"] = layer.saved_attn_weights
           
            for i, layer in enumerate(self.hierarchical_module.phrase_scale):
                if hasattr(layer, 'saved_attn_weights'):
                    attention_maps[f"phrase_scale_{i}"] = layer.saved_attn_weights
                   
            for i, layer in enumerate(self.hierarchical_module.sentence_scale):
                if hasattr(layer, 'saved_attn_weights'):
                    attention_maps[f"sentence_scale_{i}"] = layer.saved_attn_weights
        else:
            for i, layer in enumerate(self.interference_layers_word):
                if hasattr(layer, 'saved_attn_weights'):
                    attention_maps[f"word_{i}"] = layer.saved_attn_weights
                   
            for i, layer in enumerate(self.interference_layers_phrase):
                if hasattr(layer, 'saved_attn_weights'):
                    attention_maps[f"phrase_{i}"] = layer.saved_attn_weights
       
        return attention_maps
                   
    def forward(self, x, other_features=None, return_attention=False):
        complex_emb = self.embedding(x)
        complex_emb = self.pos_encoding(complex_emb)
        real, imag = complex_emb
        real = self.dropout(real)
        imag = self.dropout(imag)
        complex_emb = (real, imag)
       
        if self.use_hierarchical:
            complex_emb = self.hierarchical_module(complex_emb)
        else:
            for layer, norm in zip(self.interference_layers_word, self.norm_layers_word):
                normed = norm(complex_emb)
                out = layer(normed)
                r, i = complex_emb
                complex_emb = (r + out[0], i + out[1])
           
            pooled_real = torch.mean(complex_emb[0], dim=1, keepdim=True)
            pooled_imag = torch.mean(complex_emb[1], dim=1, keepdim=True)
            pooled = (pooled_real, pooled_imag)
           
            for layer, norm in zip(self.interference_layers_phrase, self.norm_layers_phrase):
                normed = norm(pooled)
                out = layer(normed)
                pr, pi = pooled
                pooled = (pr + out[0], pi + out[1])
               
            fused_real = torch.cat([complex_emb[0], pooled[0].expand_as(complex_emb[0])], dim=-1)
            fused_imag = torch.cat([complex_emb[1], pooled[1].expand_as(complex_emb[1])], dim=-1)
           
            fusion_proj = nn.Linear(fused_real.size(-1), self.dim).to(fused_real.device)
            fused_real = fusion_proj(fused_real)
            fused_imag = fusion_proj(fused_imag)
            complex_emb = (fused_real, fused_imag)
       
        if self.use_cross_modal and other_features is not None and hasattr(self, 'cross_modal_fusion'):
            real_fused = self.cross_modal_fusion(complex_emb[0], other_features)
            complex_emb = (real_fused, complex_emb[1])
       
        mem_read = self.memory.read(complex_emb[0])
        complex_emb = (complex_emb[0] + mem_read, complex_emb[1])
       
        complex_emb = self.final_norm(complex_emb)
        logits = self.collapse(complex_emb)
       
        if self.training:
            self.memory.write(complex_emb[0])
       
        if return_attention or self.explainability_mode in ["attention", "all"]:
            self.saved_attention_maps = self.get_attention_maps()
           
            if self.explainability_mode in ["entropy", "all"]:
                self.saved_entropies = {
                    name: compute_von_neumann_entropy(attn)
                    for name, attn in self.saved_attention_maps.items()
                }
       
        return logits

    def generate(self, input_ids, max_length=100, temperature=0.8, top_p=0.9, top_k=50,
                 repetition_penalty=1.1, do_sample=True, bos_token_id=None, pad_token_id=None,
                 eos_token_id=None, use_cache=True, num_return_sequences=1,
                 min_length=10, length_penalty=1.0, no_repeat_ngram_size=3):
        """
        Enhanced auto-regressive generation with various decoding strategies.
        """
        batch_size = input_ids.shape[0]
       
        if bos_token_id is None:
            bos_token_id = 0
        if eos_token_id is None:
            eos_token_id = 0
        if pad_token_id is None:
            pad_token_id = 0
           
        attn_mask = torch.ones_like(input_ids)
       
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        cur_len = input_ids.shape[1]
       
        self.memory.reset_memory()
       
        past_key_values = None
       
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            attn_mask = attn_mask.repeat_interleave(num_return_sequences, dim=0)
            unfinished_sequences = unfinished_sequences.repeat_interleave(num_return_sequences, dim=0)
            batch_size *= num_return_sequences
       
        while cur_len < max_length and torch.any(unfinished_sequences).item():
            with torch.no_grad():
                if use_cache and past_key_values is not None:
                    logits = self(input_ids[:, -1:], past_key_values=past_key_values)
                else:
                    logits = self(input_ids)
               
                next_token_logits = logits[:, -1, :]
               
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
               
                if repetition_penalty > 1.0:
                    for batch_idx in range(batch_size):
                        for prev_token in input_ids[batch_idx].unique():
                            next_token_logits[batch_idx, prev_token] /= repetition_penalty
               
                if no_repeat_ngram_size > 0:
                    pass
               
                if cur_len < min_length:
                    next_token_logits[:, eos_token_id] = -float('inf')
               
                if top_k > 0:
                    next_token_logits = self._top_k_top_p_filtering(
                        next_token_logits, top_k=top_k, top_p=top_p if do_sample else 1.0
                    )
               
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
               
                tokens_to_add = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
               
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attn_mask = torch.cat([attn_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)
               
                unfinished_sequences = unfinished_sequences.mul(
                    (tokens_to_add != eos_token_id).long()
                )
               
                cur_len += 1
               
                if torch.all(unfinished_sequences == 0).item():
                    break
       
        return input_ids
   
    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float('inf')):
        top_k = min(top_k, logits.size(-1))
       
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
           
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
           
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
           
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
           
        return logits

###############################################################################
#                         DATASET & DATALOADER                                #
###############################################################################

class EnhancedSFINDataset(Dataset):
    """
    Enhanced dataset for SFIN with better preprocessing and augmentation.
    """
    def __init__(self, texts, tokenizer, max_length=256, min_length=8,
                 augment=False, augment_prob=0.1):
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
           
            encodings = tokenizer(text, truncation=True, max_length=max_length,
                                  padding="max_length", return_tensors="pt")
           
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
       
        if len(positions_to_mask) > 0:
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
            item["input_ids"], item["attention_mask"] = self._augment_text(
                item["input_ids"], item["attention_mask"]
            )
           
        return item

###############################################################################
#                  TRAINING, EVALUATION, & HYPERPARAMETER TUNING              #
###############################################################################

def evaluate(model, dataloader, loss_fn, fp16=True):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
   
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
           
            if fp16:
                with autocast():
                    logits = model(input_ids)
                    active_loss = attn_mask.view(-1) == 1
                    active_logits = logits.view(-1, model.vocab_size)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fn(active_logits, active_labels)
                   
                    preds = torch.argmax(active_logits, dim=-1)
                    all_preds.append(preds.cpu())
                    all_labels.append(active_labels.cpu())
            else:
                logits = model(input_ids)
                active_loss = attn_mask.view(-1) == 1
                active_logits = logits.view(-1, model.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
               
                preds = torch.argmax(active_logits, dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(active_labels.cpu())
               
            total_loss += loss.item()
   
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(dataloader)
   
    return metrics

def plot_gradient_flow(model, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
   
    avg_grads = []
    layers = []
    max_grads = []
   
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
   
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
   
    plt.savefig(os.path.join(output_dir, "gradient_flow.png"))
    plt.close()

def save_attention_heatmaps(model, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
   
    if hasattr(model, 'saved_attention_maps'):
        for name, attn_map in model.saved_attention_maps.items():
            if len(attn_map.shape) == 4:
                for head_idx in range(min(4, attn_map.shape[1])):
                    plot_attention_heatmap(
                        attn_map, 
                        filename=f"attn_{name}_head{head_idx}.png",
                        head_idx=head_idx
                    )

def train_model(model, train_dataloader, eval_dataloader=None, epochs=3, lr=5e-5,
                warmup_steps=100, fp16=True, log_interval=10, save_interval=200,
                eval_interval=100, checkpoint_dir="checkpoints",
                scheduler_type="linear", weight_decay=0.01,
                gradient_accumulation_steps=1, max_grad_norm=1.0,
                early_stopping_patience=3, adaptive_training=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
   
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
   
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
   
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
   
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr/100
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
   
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler() if fp16 else None
   
    global_step = 0
    best_eval_loss = float("inf")
    early_stopping_counter = 0
    model.train()
   
    if config["use_tensorboard"]:
        tb_writer = SummaryWriter(os.path.join(checkpoint_dir, "runs"))
   
    logger.info("Starting training...")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Total optimization steps: {total_steps}")
   
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
       
        if hasattr(model, 'collapse_alpha'):
            model.collapse_alpha = torch.tensor(min(1.0, epoch / (epochs * 0.8))).to(device)
            logger.info(f"Setting collapse_alpha to {model.collapse_alpha.item():.2f}")
       
        if adaptive_training:
            noise_scale = max(0.01, 0.1 * (1 - epoch / epochs))
            for layer in model.modules():
                if hasattr(layer, 'noise_scale'):
                    layer.noise_scale = noise_scale
            logger.info(f"Setting noise_scale to {noise_scale:.3f}")
       
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
           
            if fp16:
                with autocast():
                    logits = model(input_ids)
                    active_loss = attn_mask.view(-1) == 1
                    active_logits = logits.view(-1, model.vocab_size)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fn(active_logits, active_labels)
                    loss = loss / gradient_accumulation_steps
            else:
                logits = model(input_ids)
                active_loss = attn_mask.view(-1) == 1
                active_logits = logits.view(-1, model.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss = loss / gradient_accumulation_steps
           
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
               
            epoch_loss += loss.item() * gradient_accumulation_steps
           
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if fp16:
                    scaler.unscale_(optimizer)
                   
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
               
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                   
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
               
                if global_step % log_interval == 0:
                    lr_val = scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step}: loss = {loss.item() * gradient_accumulation_steps:.4f}, lr = {lr_val:.6f}")
                   
                    if config["use_tensorboard"]:
                        tb_writer.add_scalar("train/loss", loss.item() * gradient_accumulation_steps, global_step)
                        tb_writer.add_scalar("train/lr", lr_val, global_step)
                       
                        grad_norm = 0.0
                        param_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                                param_norm += p.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        param_norm = param_norm ** 0.5
                       
                        tb_writer.add_scalar("train/grad_norm", grad_norm, global_step)
                        tb_writer.add_scalar("train/param_norm", param_norm, global_step)
                       
                        grad_entropy_val = gradient_entropy(model)
                        tb_writer.add_scalar("train/grad_entropy", grad_entropy_val, global_step)
               
                if global_step % save_interval == 0 and global_step > 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"sfin_step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item() * gradient_accumulation_steps,
                    }, ckpt_path)
                    logger.info(f"Checkpoint saved to {ckpt_path}")
               
                if eval_dataloader is not None and global_step % eval_interval == 0 and global_step > 0:
                    metrics = evaluate(model, eval_dataloader, loss_fn, fp16)
                    eval_loss = metrics["loss"]
                   
                    logger.info(f"Evaluation at step {global_step}:")
                    logger.info(f"  Loss: {eval_loss:.4f}")
                    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  F1: {metrics['f1']:.4f}")
                   
                    if config["use_tensorboard"]:
                        for key, value in metrics.items():
                            tb_writer.add_scalar(f"eval/{key}", value, global_step)
                   
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        early_stopping_counter = 0
                        best_path = os.path.join(checkpoint_dir, "sfin_best_model.pt")
                        torch.save({
                            "step": global_step,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "loss": best_eval_loss,
                            "metrics": metrics,
                        }, best_path)
                        logger.info(f"New best model saved with eval loss: {best_eval_loss:.4f}")
                    else:
                        early_stopping_counter += 1
                        logger.info(f"No improvement over best eval loss. Counter: {early_stopping_counter}/{early_stopping_patience}")
                       
                        if early_stopping_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {global_step} steps")
                            break
                   
                    if config["plot_gradient_flow"] and global_step % (eval_interval * 5) == 0:
                        plot_gradient_flow(model)
                       
                    if config["save_attention_heatmaps"] and global_step % (eval_interval * 5) == 0:
                        with torch.no_grad():
                            _ = model(input_ids, return_attention=True)
                        save_attention_heatmaps(model)
                   
                    model.train()
       
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        logger.info(f"Average loss: {epoch_loss/len(train_dataloader):.4f}")
       
        epoch_ckpt = os.path.join(checkpoint_dir, f"sfin_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_loss/len(train_dataloader),
        }, epoch_ckpt)
        logger.info(f"Epoch checkpoint saved to {epoch_ckpt}")
       
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
   
    if eval_dataloader is not None:
        logger.info("Performing final evaluation...")
        final_metrics = evaluate(model, eval_dataloader, loss_fn, fp16)
        logger.info(f"Final evaluation results:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
   
    final_model_path = os.path.join(checkpoint_dir, "sfin_final_model.pt")
    torch.save({
        "epochs": epochs,
        "model_state_dict": model.state_dict(),
        "final_metrics": final_metrics if eval_dataloader is not None else None,
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
   
    if config["use_tensorboard"]:
        tb_writer.close()
   
    return final_metrics if eval_dataloader is not None else {"loss": epoch_loss/len(train_dataloader)}

def run_hyperparameter_optimization(train_data, eval_data, tokenizer, vocab_size, 
                                  n_trials=20, study_name="sfin_optimization"):
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
   
    def objective(trial):
        dim = trial.suggest_categorical("dim", [256, 384, 512, 768])
        depth = trial.suggest_int("depth", 3, 8)
        heads = trial.suggest_categorical("heads", [4, 8, 12, 16])
        dropout = trial.suggest_float("dropout", 0.05, 0.2)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
        interference_type = trial.suggest_categorical("interference_type", ["quantum", "classical"])
        collapse_type = trial.suggest_categorical("collapse_type", ["squared_magnitude", "interference", "entanglement"])
        use_hierarchical = trial.suggest_categorical("use_hierarchical", [True, False])
       
        model = AdvancedSFIN(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            interference_type=interference_type,
            collapse_type=collapse_type,
            use_hierarchical=use_hierarchical
        ).to(device)
       
        train_dataset = EnhancedSFINDataset(train_data, tokenizer, max_length=256)
        eval_dataset = EnhancedSFINDataset(eval_data, tokenizer, max_length=256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
       
        metrics = train_model(
            model, 
            train_loader, 
            eval_loader,
            epochs=1,
            lr=learning_rate,
            checkpoint_dir=f"checkpoints/trial_{trial.number}",
            log_interval=50,
            eval_interval=100,
            save_interval=500
        )
       
        return metrics["loss"]
   
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
   
    study.optimize(objective, n_trials=n_trials)
   
    logger.info("Hyperparameter optimization completed")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best loss: {study.best_trial.value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")
   
    os.makedirs("optuna", exist_ok=True)
    with open(f"optuna/{study_name}_results.json", "w") as f:
        json.dump({
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "best_trial": study.best_trial.number,
            "n_trials": n_trials
        }, f, indent=2)
   
    return study.best_trial.params

###############################################################################
#                           TEXT GENERATION                                   #
###############################################################################

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8,
                 top_p=0.9, top_k=50, repetition_penalty=1.1, num_return_sequences=1,
                 control_tokens=None, beam_size=None, explain_generation=False):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
   
    if control_tokens is not None:
        control_tensor = torch.tensor(control_tokens).unsqueeze(0).to(device)
        input_ids = torch.cat([control_tensor, input_ids], dim=1)
   
    if beam_size is not None and beam_size > 1:
        return _generate_beam_search(
            model, tokenizer, input_ids, beam_size, max_length, temperature,
            repetition_penalty
        )
   
    generated_sequences = []
    attention_maps = {} if explain_generation else None
   
    for _ in range(num_return_sequences):
        cur_ids = input_ids.clone()
        past_key_values = None
       
        for i in range(max_length):
            with torch.no_grad():
                outputs = model(cur_ids, return_attention=explain_generation)
                logits = outputs
               
                if explain_generation and hasattr(model, 'saved_attention_maps'):
                    step_attentions = {
                        f"{name}_step{i}": attn 
                        for name, attn in model.saved_attention_maps.items()
                    }
                    attention_maps.update(step_attentions)
               
                next_logits = logits[:, -1, :]
               
                if temperature > 0:
                    next_logits = next_logits / temperature
               
                for token_id in cur_ids.view(-1).unique():
                    next_logits[:, token_id] /= repetition_penalty
               
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(next_logits, top_k)
                    filtered_logits = torch.full_like(next_logits, float('-inf'))
                    filtered_logits.scatter_(1, topk_idx, topk_vals)
                    next_logits = filtered_logits
               
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                   
                    sorted_idx_to_remove = cumulative_probs > top_p
                    sorted_idx_to_remove[..., 0] = 0
                   
                    sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                   
                    for batch_idx in range(next_logits.size(0)):
                        indices_to_remove = sorted_idx[batch_idx][sorted_idx_to_remove[batch_idx]]
                        next_logits[batch_idx, indices_to_remove] = float('-inf')
               
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
               
                cur_ids = torch.cat([cur_ids, next_token], dim=1)
               
                if next_token.item() == tokenizer.eos_token_id:
                    break
       
        generated_sequences.append(cur_ids)
   
    decoded = []
    for seq in generated_sequences:
        text = tokenizer.decode(seq[0], skip_special_tokens=True)
        decoded.append(text)
   
    if explain_generation:
        return decoded, attention_maps
    else:
        return decoded

def _generate_beam_search(model, tokenizer, input_ids, beam_size, max_length, 
                         temperature=1.0, repetition_penalty=1.0):
    batch_size = input_ids.shape[0]
    vocab_size = model.vocab_size
   
    scores = torch.zeros(batch_size, beam_size, device=device)
    sequences = input_ids.repeat_interleave(beam_size, dim=0)
   
    done = [False for _ in range(batch_size * beam_size)]
   
    with torch.no_grad():
        logits = model(sequences)
        next_logits = logits[:, -1, :] / temperature
       
        for i, seq in enumerate(sequences):
            for token_id in seq.unique():
                next_logits[i, token_id] /= repetition_penalty
       
        log_probs = F.log_softmax(next_logits, dim=-1)
        top_scores, top_tokens = log_probs.topk(beam_size, dim=-1)
       
        new_sequences = []
        new_scores = []
       
        for batch_idx in range(batch_size):
            batch_start = batch_idx * beam_size
            curr_seq = sequences[batch_start:batch_start+1]
           
            for beam_idx in range(beam_size):
                token = top_tokens[batch_start, beam_idx].unsqueeze(0)
                score = top_scores[batch_start, beam_idx].item()
               
                new_seq = torch.cat([curr_seq, token.unsqueeze(0)], dim=1)
                new_sequences.append(new_seq)
                new_scores.append(score)
       
        sequences = torch.cat(new_sequences, dim=0)
        scores = torch.tensor(new_scores, device=device).view(batch_size, beam_size)
       
        for step in range(1, max_length):
            active_batch_size = sum(1 for d in done if not d)
            if active_batch_size == 0:
                break
               
            active_sequences = torch.cat([seq.unsqueeze(0) for seq, d in zip(sequences, done) if not d], dim=0)
            logits = model(active_sequences)
            next_logits = logits[:, -1, :] / temperature
           
            for i, seq in enumerate(active_sequences):
                for token_id in seq.unique():
                    next_logits[i, token_id] /= repetition_penalty
           
            log_probs = F.log_softmax(next_logits, dim=-1)
           
            active_idx = 0
            new_sequences = []
            new_scores = []
            new_done = []
           
            for batch_idx in range(batch_size):
                batch_start = batch_idx * beam_size
               
                batch_candidates = []
               
                for beam_idx in range(beam_size):
                    idx = batch_start + beam_idx
                   
                    if done[idx]:
                        new_sequences.append(sequences[idx].unsqueeze(0))
                        new_scores.append(scores[batch_idx, beam_idx])
                        new_done.append(True)
                        continue
                   
                    curr_log_probs = log_probs[active_idx]
                    curr_score = scores[batch_idx, beam_idx]
                    active_idx += 1
                   
                    topk_probs, topk_tokens = curr_log_probs.topk(beam_size)
                   
                    for k in range(beam_size):
                        token = topk_tokens[k].unsqueeze(0)
                        log_prob = topk_probs[k].item()
                        score = curr_score + log_prob
                       
                        batch_candidates.append({
                            "seq": torch.cat([sequences[idx].unsqueeze(0), token.unsqueeze(0)], dim=1),
                            "score": score,
                            "done": token.item() == tokenizer.eos_token_id
                        })
               
                batch_candidates.sort(key=lambda x: -x["score"])
                selected = batch_candidates[:beam_size]
               
                for candidate in selected:
                    new_sequences.append(candidate["seq"])
                    new_scores.append(candidate["score"])
                    new_done.append(candidate["done"])
           
            sequences = torch.cat(new_sequences, dim=0)
            scores = torch.tensor(new_scores, device=device).view(batch_size, beam_size)
            done = new_done
   
    result = []
    for batch_idx in range(batch_size):
        batch_start = batch_idx * beam_size
        best_idx = scores[batch_idx].argmax().item()
        best_seq = sequences[batch_start + best_idx]
        result.append(best_seq.unsqueeze(0))
   
    result = torch.cat(result, dim=0)
   
    decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in result]
    return decoded

class ExplainabilityTools:
    """
    Tools for understanding and visualizing SFIN behavior.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
       
    def visualize_attention(self, text_input, layer_name=None, head_idx=0, output_dir="visualizations"):
        os.makedirs(output_dir, exist_ok=True)
       
        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(device)
        input_tokens = [self.tokenizer.decode([id]) for id in input_ids[0]]
       
        with torch.no_grad():
            _ = self.model(input_ids, return_attention=True)
       
        if hasattr(self.model, 'saved_attention_maps'):
            attention_maps = self.model.saved_attention_maps
           
            if layer_name is not None:
                attention_maps = {k: v for k, v in attention_maps.items() if layer_name in k}
           
            for name, attn in attention_maps.items():
                head_attn = attn[0, head_idx].cpu().numpy()
               
                plt.figure(figsize=(10, 8))
                plt.imshow(head_attn, cmap='viridis')
                plt.colorbar()
                plt.title(f"Attention: {name}, Head {head_idx}")
               
                plt.xticks(range(len(input_tokens)), input_tokens, rotation=90)
                plt.yticks(range(len(input_tokens)), input_tokens)
               
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"attn_{name}_head{head_idx}.png"))
                plt.close()
               
            return attention_maps
        else:
            print("No attention maps found. Make sure return_attention=True in forward pass.")
            return None
       
    def analyze_generation(self, prompt, max_length=30, temperature=0.8, top_p=0.9):
        self.model.eval()
        tokenized = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_length = tokenized.shape[1]
       
        token_explanations = []
        all_attention_maps = {}
        current_text = prompt
       
        for i in range(max_length):
            with torch.no_grad():
                outputs = self.model(tokenized, return_attention=True)
                logits = outputs
               
                next_token_logits = logits[:, -1, :] / temperature
               
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
               
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0
               
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
               
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('inf')
               
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
               
                next_token_str = self.tokenizer.decode(next_token[0])
               
                if hasattr(self.model, 'saved_attention_maps'):
                    step_maps = {f"{name}_step{i}": attn.detach().cpu() 
                                for name, attn in self.model.saved_attention_maps.items()}
                    all_attention_maps.update(step_maps)
               
                topk = 5
                topk_tokens = torch.topk(probs, topk, dim=-1)
                alternatives = []
               
                for j in range(topk):
                    token_id = topk_tokens.indices[0, j].item()
                    token_prob = topk_tokens.values[0, j].item()
                    token_str = self.tokenizer.decode([token_id])
                    alternatives.append((token_str, token_prob))
               
                explanation = {
                    "position": i + input_length,
                    "token": next_token_str,
                    "token_id": next_token.item(),
                    "probability": probs[0, next_token].item(),
                    "alternatives": alternatives,
                    "attention": self.model.saved_attention_maps if hasattr(self.model, 'saved_attention_maps') else None
                }
               
                token_explanations.append(explanation)
               
                tokenized = torch.cat((tokenized, next_token), dim=1)
                current_text += next_token_str
               
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
       
        token_importance = self._compute_token_importance(all_attention_maps)
       
        for i, explanation in enumerate(token_explanations):
            if i < len(token_importance):
                explanation["importance"] = token_importance[i]
       
        return current_text, token_explanations, all_attention_maps
       
    def _compute_token_importance(self, attention_maps):
        attention_matrices = []
       
        for name, attn in attention_maps.items():
            if "word" in name or "phrase" in name:
                avg_attn = attn.mean(dim=1)
                attention_matrices.append(avg_attn)
       
        if not attention_matrices:
            return []
           
        importance = torch.zeros_like(attention_matrices[0][0, -1, :])
       
        for attn in attention_matrices:
            importance += attn[0, -1, :]
       
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)
       
        return importance.cpu().numpy().tolist()
       
    def explain_model_parameters(self):
        info = {}
       
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
       
        info["parameters"] = {
            "total": total_params,
            "trainable": trainable_params
        }
       
        entanglement_metrics = {}
       
        for name, module in self.model.named_modules():
            if isinstance(module, EntangledInterferenceLayer) and hasattr(module, 'entanglement_matrix'):
                try:
                    eigvals = torch.linalg.eigvalsh(module.entanglement_matrix.detach())
                    entanglement_metrics[name] = {
                        "eigenvalues": eigvals.cpu().numpy().tolist(),
                        "max_eigenvalue": eigvals.max().item(),
                        "min_eigenvalue": eigvals.min().item(),
                        "trace": torch.trace(module.entanglement_matrix).item()
                    }
                except:
                    pass
       
        info["entanglement"] = entanglement_metrics
       
        if hasattr(self.model, 'embedding'):
            real_norm = torch.norm(self.model.embedding.real_embedding.weight.detach(), dim=1)
            imag_norm = torch.norm(self.model.embedding.imag_embedding.weight.detach(), dim=1)
           
            info["embedding"] = {
                "real_norm_mean": real_norm.mean().item(),
                "real_norm_std": real_norm.std().item(),
                "imag_norm_mean": imag_norm.mean().item(),
                "imag_norm_std": imag_norm.std().item(),
                "ratio_mean": (real_norm / (imag_norm + 1e-8)).mean().item()
            }
       
        return info

###############################################################################
#                           MAIN FUNCTION                                     #
###############################################################################

def main(mode="train", use_hyperopt=False, generation_examples=True, load_checkpoint=None):
    """
    Main function to run training, evaluation, or generation.
    
    Args:
        mode: "train", "evaluate", "generate", or "explain"
        use_hyperopt: Whether to use hyperparameter optimization
        generation_examples: Whether to include text generation examples
        load_checkpoint: Path to checkpoint to load (None = train from scratch)
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
   
    logger.info(f"Initialized tokenizer with vocabulary size: {vocab_size}")
   
    model_args = {
        "vocab_size": vocab_size,
        "dim": 768,
        "depth": 6,
        "heads": 12,
        "dropout": 0.1,
        "interference_type": "quantum",
        "collapse_type": "squared_magnitude",
        "max_seq_len": 256,
        "mem_size": 32,
        "use_hierarchical": True
    }
   
    logger.info("Loading dataset...")
    dataset_raw = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train")
    texts = [ex["text"] for ex in dataset_raw if "text" in ex]
    logger.info(f"Loaded {len(texts)} conversations.")
   
    split_idx = int(0.9 * len(texts))
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
   
    if load_checkpoint:
        logger.info(f"Loading model from checkpoint: {load_checkpoint}")
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model = AdvancedSFIN(**model_args).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.info("Creating new model with default parameters")
        model = AdvancedSFIN(**model_args).to(device)
   
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
   
    batch_size = 4
   
    if mode in ["train", "evaluate"]:
        train_dataset = EnhancedSFINDataset(train_texts, tokenizer, max_length=256)
        eval_dataset = EnhancedSFINDataset(eval_texts, tokenizer, max_length=256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
       
        logger.info(f"Created dataloaders with batch size {batch_size}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
   
    if use_hyperopt and mode == "train":
        logger.info("Running hyperparameter optimization")
        best_params = run_hyperparameter_optimization(
            train_texts, eval_texts, tokenizer, vocab_size, n_trials=10
        )
       
        model_args.update(best_params)
        model = AdvancedSFIN(**model_args).to(device)
        logger.info("Created model with optimized hyperparameters")
   
    if mode == "train":
        logger.info("Starting model training")
        train_model(
            model, 
            train_loader, 
            eval_dataloader=eval_loader, 
            epochs=3, 
            lr=5e-5,
            warmup_steps=100, 
            fp16=True, 
            log_interval=10, 
            save_interval=200, 
            eval_interval=100,
            adaptive_training=True
        )
   
    elif mode == "evaluate":
        logger.info("Evaluating model")
        loss_fn = nn.CrossEntropyLoss()
        metrics = evaluate(model, eval_loader, loss_fn)
       
        logger.info("Evaluation results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
   
    elif mode == "generate":
        logger.info("Running text generation examples")
       
        prompts = [
            "Semantic fields create",
            "The quantum properties of language allow",
            "In the future, AI will",
            "The relationship between consciousness and computation is"
        ]
       
        for prompt in prompts:
            logger.info(f"\nGenerating from prompt: '{prompt}'")
           
            generated = generate_text(
                model, tokenizer, prompt, max_length=100, 
                temperature=0.8, top_p=0.9, top_k=50
            )
           
            logger.info("Generated text (temperature=0.8):")
            for text in generated:
                logger.info(text)
           
            generated_beam = generate_text(
                model, tokenizer, prompt, max_length=100,
                temperature=0.7, beam_size=3
            )
           
            logger.info("\nGenerated text (beam search, beam_size=3):")
            for text in generated_beam:
                logger.info(text)
   
    elif mode == "explain":
        logger.info("Running model explainability tools")
        explainer = ExplainabilityTools(model, tokenizer)
       
        model_info = explainer.explain_model_parameters()
       
        logger.info("Model parameter analysis:")
        logger.info(f"Total parameters: {model_info['parameters']['total']:,}")
        logger.info(f"Trainable parameters: {model_info['parameters']['trainable']:,}")
       
        if 'embedding' in model_info:
            logger.info("\nEmbedding analysis:")
            logger.info(f"Real norm mean: {model_info['embedding']['real_norm_mean']:.4f}")
            logger.info(f"Imaginary norm mean: {model_info['embedding']['imag_norm_mean']:.4f}")
            logger.info(f"Real/Imaginary ratio: {model_info['embedding']['ratio_mean']:.4f}")
       
        sample_text = "The quantum properties of language emerge through semantic interference."
        logger.info(f"\nVisualizing attention for input: '{sample_text}'")
       
        attention_maps = explainer.visualize_attention(sample_text)
       
        sample_prompt = "Semantic fields interact through"
        logger.info(f"\nAnalyzing generation process for prompt: '{sample_prompt}'")
       
        generated_text, token_explanations, _ = explainer.analyze_generation(
            sample_prompt, max_length=20
        )
       
        logger.info(f"Generated text: {generated_text}")
        logger.info("\nToken-by-token explanation:")
       
        for i, explanation in enumerate(token_explanations):
            token = explanation["token"]
            prob = explanation["probability"]
           
            alternatives = ", ".join([f"{t} ({p:.2f})" for t, p in explanation["alternatives"][:3]])
           
            logger.info(f"Token {i+1}: '{token}' (p={prob:.4f})")
            logger.info(f"  Alternatives: {alternatives}")
           
            if "importance" in explanation:
                logger.info(f"  Importance: {explanation['importance']:.4f}")
           
            logger.info("")
   
    else:
        logger.info(f"Unknown mode: {mode}")
   
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
   
    parser = argparse.ArgumentParser(description="Advanced SFIN Training and Evaluation")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "evaluate", "generate", "explain"],
                        help="Mode to run the script in")
    parser.add_argument("--hyperopt", action="store_true", 
                        help="Run hyperparameter optimization")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to load")
   
    args = parser.parse_args()
   
    main(mode=args.mode, use_hyperopt=args.hyperopt, 
         generation_examples=True, load_checkpoint=args.checkpoint)
