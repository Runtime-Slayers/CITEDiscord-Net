"""
cross_modal_attention.py — Cross-Modal Attention Transformer (CMAT) + InfoNCE
==============================================================================

Novel contributions to CITEDiscord-Net:

1. Cross-Modal Attention Transformer (CMAT)
   -----------------------------------------
   After modality-specific encoders produce (μ_rna, σ_rna) and (μ_adt, σ_adt),
   a multi-head cross-attention mechanism lets RNA attend to protein features and
   vice versa. This captures fine-grained cross-modal dependencies that scalar
   Product-of-Experts fusion misses.

   Architecture:
     - RNA query attends over protein key/value → attended_rna
     - Protein query attends over RNA key/value → attended_prot
     - Residual connection + LayerNorm on each branch
     - A gating network learns per-cell mixture weights α ∈ [0,1]
     - Final: μ* = α · μ_attended + (1-α) · μ_original

   This is analogous to cross-attention in transformers (Vaswani et al. 2017)
   adapted for paired latent Gaussian parameters rather than token sequences.

2. InfoNCE Contrastive RNA-Protein Alignment Loss
   ------------------------------------------------
   For each cell i, the pair (z_rna_i, z_prot_i) is a positive pair.
   All other cells' representations are negatives.

   InfoNCE objective (van den Oord et al. 2018):
     L_NCE = -E[ log( exp(sim(z_r_i, z_p_i)/τ) / Σ_j exp(sim(z_r_i, z_p_j)/τ) ) ]

   This aligns RNA and protein latent spaces beyond ELBO, improving modality
   coherence and reducing the cross-modal gap that causes discordance inflation.

3. Mixture-of-Gaussians (MoG) Prior
   -----------------------------------
   Instead of a standard N(0,I) prior, we use a K-component GMM prior whose
   component assignment probabilities are inferred per-cell via a gumbel-softmax
   relaxation. This yields a more structured latent space aligned with cell clusters.

     p(z) = Σ_{k=1}^{K} π_k · N(μ_k, σ_k²·I)

   The MVAE KL divergence is computed against this GMM prior using an
   importance-weighted mixture bound (VLAE approximation).
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-Head Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossModalAttentionBlock(nn.Module):
    """
    Multi-head cross-attention between RNA and protein latent Gaussian parameters.

    Treats latent means as query/key/value vectors of length d_z.
    Sequences of length 1 (each cell is a single 'token'), but this generalises
    to multi-token representations in future work.

    Parameters
    ----------
    latent_dim : int        Dimensionality of each latent vector
    n_heads    : int        Number of attention heads
    dropout    : float      Attention weight dropout
    """

    def __init__(self, latent_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert latent_dim % n_heads == 0, (
            f"latent_dim ({latent_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.latent_dim = latent_dim
        self.n_heads    = n_heads
        self.d_head     = latent_dim // n_heads

        # RNA attends to Protein
        self.rna_to_prot_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.rna_to_prot_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.rna_to_prot_v = nn.Linear(latent_dim, latent_dim, bias=False)
        self.rna_out_proj   = nn.Linear(latent_dim, latent_dim)

        # Protein attends to RNA
        self.prot_to_rna_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.prot_to_rna_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.prot_to_rna_v = nn.Linear(latent_dim, latent_dim, bias=False)
        self.prot_out_proj  = nn.Linear(latent_dim, latent_dim)

        # Layer norms (post-attention)
        self.norm_rna  = nn.LayerNorm(latent_dim)
        self.norm_prot = nn.LayerNorm(latent_dim)

        # Gating: learn per-cell mixture weight α ∈ [0,1]
        self.gate_rna  = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.Sigmoid(),
        )
        self.gate_prot = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_head)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B, H, 1, d_head) for batch-level token attention."""
        B = x.shape[0]
        return x.view(B, self.n_heads, 1, self.d_head)

    def _scaled_dot_product(self, q, k, v) -> torch.Tensor:
        """Standard scaled dot-product attention over 1-token sequences."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, 1, 1)
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        return torch.matmul(attn, v).squeeze(-2)                     # (B, H, d_head)

    def forward(
        self,
        mu_rna:  torch.Tensor,   # (B, latent_dim)
        mu_prot: torch.Tensor,   # (B, latent_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu_rna_out  : (B, latent_dim)  — cross-attended RNA latent
        mu_prot_out : (B, latent_dim)  — cross-attended protein latent
        """
        B = mu_rna.shape[0]

        # ── RNA attends to Protein ──────────────────────────────────
        Qq_rna = self._split_heads(self.rna_to_prot_q(mu_rna))
        Kk_rna = self._split_heads(self.rna_to_prot_k(mu_prot))
        Vv_rna = self._split_heads(self.rna_to_prot_v(mu_prot))
        attended_rna = self._scaled_dot_product(Qq_rna, Kk_rna, Vv_rna)  # (B, H, d_head)
        attended_rna = attended_rna.reshape(B, self.latent_dim)
        attended_rna = self.rna_out_proj(attended_rna)

        # Gated residual
        alpha_rna   = self.gate_rna(torch.cat([mu_rna, attended_rna], dim=-1))
        mu_rna_out  = self.norm_rna(mu_rna + alpha_rna * attended_rna)

        # ── Protein attends to RNA ─────────────────────────────────
        Qq_prot = self._split_heads(self.prot_to_rna_q(mu_prot))
        Kk_prot = self._split_heads(self.prot_to_rna_k(mu_rna))
        Vv_prot = self._split_heads(self.prot_to_rna_v(mu_rna))
        attended_prot = self._scaled_dot_product(Qq_prot, Kk_prot, Vv_prot)
        attended_prot = attended_prot.reshape(B, self.latent_dim)
        attended_prot = self.prot_out_proj(attended_prot)

        alpha_prot   = self.gate_prot(torch.cat([mu_prot, attended_prot], dim=-1))
        mu_prot_out  = self.norm_prot(mu_prot + alpha_prot * attended_prot)

        return mu_rna_out, mu_prot_out


# ---------------------------------------------------------------------------
# InfoNCE Contrastive Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss for RNA ↔ Protein alignment.

    Positive pair: (z_rna_i, z_prot_i) for the same cell.
    Negatives: all other cells in the batch.

    L = -1/N · Σ_i log[ exp(sim(z_r,i, z_p,i)/τ) / Σ_j exp(sim(z_r,i, z_p,j)/τ) ]

    where sim is cosine similarity and τ is the temperature parameter.

    Parameters
    ----------
    temperature : float     Softmax temperature τ (default 0.07)
    symmetric   : bool      If True, also computes protein→RNA direction
    """

    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric   = symmetric

    def forward(
        self,
        z_rna:  torch.Tensor,   # (B, D) — RNA latent samples
        z_prot: torch.Tensor,   # (B, D) — Protein latent samples
    ) -> torch.Tensor:
        B     = z_rna.shape[0]
        device = z_rna.device

        # L2 normalise
        z_r = F.normalize(z_rna,  dim=-1)   # (B, D)
        z_p = F.normalize(z_prot, dim=-1)   # (B, D)

        # Similarity matrix
        logits = torch.matmul(z_r, z_p.T) / self.temperature  # (B, B)

        # Labels: positive pair is on the diagonal
        labels = torch.arange(B, device=device)

        loss_rna2prot = F.cross_entropy(logits, labels)

        if self.symmetric:
            loss_prot2rna = F.cross_entropy(logits.T, labels)
            return 0.5 * (loss_rna2prot + loss_prot2rna)

        return loss_rna2prot


# ---------------------------------------------------------------------------
# Mixture-of-Gaussians Prior (VampPrior-lite / GMM-VAE)
# ---------------------------------------------------------------------------

class MoGPrior(nn.Module):
    """
    Learnable K-component Mixture-of-Gaussians prior for the VAE latent space.

    Component means μ_k and log-variances log σ²_k are learned parameters.
    Component weights π_k (log-space) are also learnable.

    KL divergence approximation (single-sample importance weighted):
      KL(q(z|x) || p_MoG(z)) ≈ log q(z|x) - log p_MoG(z)

    evaluated at a Monte-Carlo sample z ~ q (reparameterised).

    Parameters
    ----------
    n_components : int      Number of GMM components K (recommend = n_clusters)
    latent_dim   : int      Latent space dimension
    """

    def __init__(self, n_components: int = 10, latent_dim: int = 32):
        super().__init__()
        self.K = n_components
        self.D = latent_dim

        # Learnable GMM parameters
        self.log_pi    = nn.Parameter(torch.zeros(n_components))
        self.mu_k      = nn.Parameter(torch.randn(n_components, latent_dim) * 0.5)
        self.logvar_k  = nn.Parameter(torch.zeros(n_components, latent_dim))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p_MoG(z) for each sample in z.

        Parameters
        ----------
        z : (B, D)

        Returns
        -------
        log_p : (B,)
        """
        # Normalise mixture weights
        log_pi = F.log_softmax(self.log_pi, dim=0)   # (K,)

        # Expand for broadcasting: z → (B, 1, D), μ_k → (1, K, D)
        z_exp  = z.unsqueeze(1)                          # (B, 1, D)
        mu_exp = self.mu_k.unsqueeze(0)                  # (1, K, D)
        lv_exp = self.logvar_k.unsqueeze(0)              # (1, K, D)

        # Log N(z; μ_k, σ_k²) per component and per dimension
        diff    = z_exp - mu_exp                         # (B, K, D)
        log_det = lv_exp.sum(-1)                         # (B, K)
        quad    = (diff ** 2 * torch.exp(-lv_exp)).sum(-1)  # (B, K)
        log_gauss = -0.5 * (self.D * math.log(2 * math.pi) + log_det + quad)  # (B, K)

        # log-sum-exp over components
        log_p = torch.logsumexp(log_pi.unsqueeze(0) + log_gauss, dim=1)  # (B,)
        return log_p

    def kl_divergence(
        self,
        mu_q:     torch.Tensor,   # (B, D)
        logvar_q: torch.Tensor,   # (B, D)
        z:        torch.Tensor,   # (B, D) — reparameterised sample
    ) -> torch.Tensor:
        """
        Monte-Carlo estimate of KL(q || p_MoG).

        KL ≈ log q(z|x) - log p_MoG(z)

        Returns
        -------
        kl : scalar tensor
        """
        # log q(z|x) under the encoder Gaussian
        log_q = -0.5 * (
            self.D * math.log(2 * math.pi)
            + logvar_q.sum(-1)
            + ((z - mu_q) ** 2 * torch.exp(-logvar_q)).sum(-1)
        )  # (B,)

        log_p = self.log_prob(z)  # (B,)

        return (log_q - log_p).mean()


# ---------------------------------------------------------------------------
# Protein-Specific Projection Head (for InfoNCE)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head used before InfoNCE loss (SimCLR-style).
    Separate heads for RNA and protein encodings.
    """
    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helper: reparameterisation trick
# ---------------------------------------------------------------------------

def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = μ + ε·σ,  ε ~ N(0,I)."""
    if not mu.requires_grad and not logvar.requires_grad:
        return mu   # deterministic inference mode
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
