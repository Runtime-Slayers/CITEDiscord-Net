"""
mvae.py — Product-of-Experts Multi-Modal VAE (MVAE)
======================================================

Architecture (Wu & Goodman, 2018 — adapted for CITE-seq)
---------------------------------------------------------

Two modality-specific encoders produce independent Gaussian posteriors:
  - RNAEncoder  : HVG → (μ_rna, log σ²_rna)
  - ProteinEncoder : ADT → (μ_adt, log σ²_adt)

Product-of-Experts (PoE) fusion:
  The joint posterior q(z | x_rna, x_adt) ∝ p(z) · q(z|x_rna) · q(z|x_adt)
  Under Gaussians this reduces to combining precisions:
    Λ_joint = Λ_prior + Λ_rna + Λ_adt
    μ_joint = Σ_joint · (Λ_prior·μ_prior + Λ_rna·μ_rna + Λ_adt·μ_adt)

This principled fusion is superior to concatenation because:
  1. Missing modalities degrade gracefully (drop one expert)
  2. Confident modality contributes more to joint latent
  3. Uncertainty is propagated explicitly

Decoders reconstruct each modality independently.
β-VAE weighting on KL term improves disentanglement (Higgins et al. 2017).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Novel hybrid components ──────────────────────────────────────────
from src.cross_modal_attention import (
    CrossModalAttentionBlock,
    InfoNCELoss,
    MoGPrior,
    ProjectionHead,
    reparameterise as cmat_reparameterise,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

def _mlp(dims: list[int], dropout: float, batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if batch_norm and i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i+1]))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class RNAEncoder(nn.Module):
    """Maps RNA Pearson-residuals → (μ, log σ²) in latent space."""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        self.body = _mlp([input_dim] + hidden_dims, dropout, batch_norm)
        self.mu_head    = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor):
        h  = self.body(x)
        return self.mu_head(h), self.logvar_head(h)


class ProteinEncoder(nn.Module):
    """Maps CLR-normalised ADT → (μ, log σ²) in latent space."""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        self.body = _mlp([input_dim] + hidden_dims, dropout, batch_norm)
        self.mu_head    = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.mu_head(h), self.logvar_head(h)


class RNADecoder(nn.Module):
    """Reconstructs RNA from latent z."""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        self.net = _mlp([latent_dim] + hidden_dims + [output_dim], dropout, batch_norm)

    def forward(self, z: torch.Tensor):
        return self.net(z)    # scale-free (Pearson residual target)


class ProteinDecoder(nn.Module):
    """Reconstructs ADT from latent z."""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        self.net = _mlp([latent_dim] + hidden_dims + [output_dim], dropout, batch_norm)

    def forward(self, z: torch.Tensor):
        return self.net(z)


# ---------------------------------------------------------------------------
# Product-of-Experts fusion
# ---------------------------------------------------------------------------

def product_of_experts(
    mus: list[torch.Tensor],
    logvars: list[torch.Tensor],
    prior_mu: Optional[torch.Tensor] = None,
    prior_logvar: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine Gaussian posteriors via Product-of-Experts.

    All inputs have shape (batch, latent_dim).
    Includes a standard-Normal prior expert.

    Returns
    -------
    mu_joint     : (batch, latent_dim)
    logvar_joint : (batch, latent_dim)
    """
    batch, latent = mus[0].shape
    device = mus[0].device

    # Prior N(0,1) — precision = 1
    if prior_mu is None:
        prior_mu     = torch.zeros(batch, latent, device=device)
    if prior_logvar is None:
        prior_logvar = torch.zeros(batch, latent, device=device)

    all_mus     = [prior_mu]    + list(mus)
    all_logvars = [prior_logvar] + list(logvars)

    # Precision = 1 / σ²
    prec = [torch.exp(-lv) for lv in all_logvars]   # (B, L) each
    prec_sum = torch.stack(prec, dim=0).sum(dim=0)   # (B, L)

    # Weighted sum of means
    mu_weighted = sum(p * m for p, m in zip(prec, all_mus))  # (B, L)

    mu_joint     = mu_weighted / (prec_sum + 1e-8)
    logvar_joint = -torch.log(prec_sum + 1e-8)

    return mu_joint, logvar_joint


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample z ~ N(μ, σ²) using reparameterisation trick."""
    if not torch.is_grad_enabled():
        return mu
    std = torch.exp(0.5 * logvar).clamp(min=1e-4, max=4.0)
    eps = torch.randn_like(std)
    return mu + std * eps


# ---------------------------------------------------------------------------
# Full MVAE
# ---------------------------------------------------------------------------

class MVAE(nn.Module):
    """
    Multi-Modal VAE for CITE-seq data.

    Accepts RNA + protein simultaneously or either modality alone.
    Joint representation learned via Product-of-Experts latent fusion.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mv = cfg["mvae"]
        rna_in  = mv["rna_input_dim"]
        adt_in  = mv["adt_input_dim"]
        lat     = mv["latent_dim"]
        rna_h   = mv["rna_hidden_dims"]
        adt_h   = mv["adt_hidden_dims"]
        dec_h   = mv["decoder_hidden_dims"]
        do      = mv.get("dropout", 0.1)
        bn      = mv.get("batch_norm", True)

        self.latent_dim = lat
        self.beta       = mv.get("beta", 4.0)

        # Core modality-specific encoders and decoders
        self.rna_enc   = RNAEncoder(rna_in, rna_h, lat, do, bn)
        self.adt_enc   = ProteinEncoder(adt_in, adt_h, lat, do, bn)
        self.rna_dec   = RNADecoder(lat, dec_h, rna_in, do, bn)
        self.adt_dec   = ProteinDecoder(lat, dec_h, adt_in, do, bn)

        # ── Novel Component 1: Cross-Modal Attention Transformer ───
        # Lets RNA attend to protein and vice-versa before PoE fusion
        cmat_cfg  = mv.get("cmat", {})
        n_heads   = cmat_cfg.get("n_heads", max(1, lat // 8))
        # Ensure latent_dim divisible by n_heads
        while lat % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.cmat = CrossModalAttentionBlock(
            latent_dim=lat, n_heads=n_heads, dropout=do
        )
        self.use_cmat = cmat_cfg.get("enabled", True)

        # ── Novel Component 2: Mixture-of-Gaussians Prior ─────────
        n_mog_components = mv.get("mog_components", 10)
        self.mog_prior   = MoGPrior(n_components=n_mog_components, latent_dim=lat)
        self.use_mog     = mv.get("use_mog_prior", True)

        # ── Novel Component 3: Projection heads for InfoNCE ────────
        proj_dim          = mv.get("proj_dim", 64)
        self.rna_proj_head  = ProjectionHead(lat, proj_dim)
        self.prot_proj_head = ProjectionHead(lat, proj_dim)
        self.info_nce       = InfoNCELoss(
            temperature=mv.get("infonce_temperature", 0.07),
            symmetric=True,
        )
        self.lambda_nce = mv.get("lambda_nce", 0.1)

    def encode(
        self,
        x_rna: Optional[torch.Tensor],
        x_adt: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both modalities with:
          1. Individual Gaussian encoders (μ_rna, μ_prot)
          2. CMAT cross-attention refinement (μ*_rna, μ*_prot)
          3. PoE fusion → (μ_joint, logvar_joint)
        """
        mus, logvars = [], []
        mu_r_raw = mu_a_raw = None
        lv_r_raw = lv_a_raw = None

        if x_rna is not None:
            mu_r_raw, lv_r_raw = self.rna_enc(x_rna)
        if x_adt is not None:
            mu_a_raw, lv_a_raw = self.adt_enc(x_adt)

        # ── Cross-Modal Attention ────────────────────────────────────
        if self.use_cmat and mu_r_raw is not None and mu_a_raw is not None:
            mu_r, mu_a = self.cmat(mu_r_raw, mu_a_raw)
            # Preserve original logvars (CMAT only refines means)
        else:
            mu_r, mu_a = mu_r_raw, mu_a_raw

        if mu_r is not None:
            mus.append(mu_r);      logvars.append(lv_r_raw)
        if mu_a is not None:
            mus.append(mu_a);      logvars.append(lv_a_raw)

        if not mus:
            raise ValueError("At least one modality must be provided")

        # Store per-modality means for InfoNCE loss (accessed during training)
        self._mu_rna_last  = mu_r
        self._mu_adt_last  = mu_a

        return product_of_experts(mus, logvars)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rna_dec(z), self.adt_dec(z)

    def forward(
        self,
        x_rna: Optional[torch.Tensor],
        x_adt: Optional[torch.Tensor],
    ):
        mu, logvar = self.encode(x_rna, x_adt)
        z          = reparameterise(mu, logvar)
        rna_hat, adt_hat = self.decode(z)
        return rna_hat, adt_hat, mu, logvar, z

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def elbo_loss(
        self,
        x_rna: torch.Tensor,
        x_adt: torch.Tensor,
        rna_hat: torch.Tensor,
        adt_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        warmup_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Hybrid ELBO with three novel loss terms:

          L = L_recon + β·warmup·L_KL + λ_NCE·L_InfoNCE

          L_recon = MSE(rna_hat, x_rna) + MSE(adt_hat, x_adt)

          L_KL    = KL(q(z|x) || p(z))
                    where p(z) is either N(0,I) [standard] or MoG [novel]

          L_InfoNCE = InfoNCE(proj_rna(μ_rna), proj_prot(μ_adt))
                      aligns RNA and protein latent spaces contrastively
        """
        # Reconstruction losses (MSE on Pearson-residual + CLR targets)
        recon_rna = F.mse_loss(rna_hat, x_rna, reduction="mean")
        recon_adt = F.mse_loss(adt_hat, x_adt, reduction="mean")

        # KL divergence
        if self.use_mog and z is not None:
            # MoG prior: MC estimate of KL(q || p_MoG)
            kl = self.mog_prior.kl_divergence(mu, logvar, z)
        else:
            # Standard isotropic Gaussian prior (analytical)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        # InfoNCE contrastive alignment loss
        loss_nce = torch.tensor(0.0, device=mu.device)
        if (self._mu_rna_last is not None and
                self._mu_adt_last is not None and
                self.lambda_nce > 0):
            z_r = self.rna_proj_head(self._mu_rna_last)
            z_p = self.prot_proj_head(self._mu_adt_last)
            loss_nce = self.info_nce(z_r, z_p)

        total = (
            recon_rna
            + recon_adt
            + self.beta * warmup_weight * kl
            + self.lambda_nce * loss_nce
        )
        return total, {
            "loss_total":   total.item(),
            "recon_rna":    recon_rna.item(),
            "recon_adt":    recon_adt.item(),
            "kl":           kl.item(),
            "loss_nce":     loss_nce.item(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mvae(
    mvae: MVAE,
    X_rna: np.ndarray,
    X_adt: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> Tuple[MVAE, list[dict]]:
    """
    Train the MVAE end-to-end.

    Includes a linear KL warm-up schedule and cosine LR decay.
    Returns trained model and per-epoch loss history.
    """
    mv = cfg["mvae"]
    lr           = mv["lr"]
    wd           = mv["weight_decay"]
    bs           = mv["batch_size"]
    n_epochs     = mv["n_epochs"]
    warmup_eps   = mv["warmup_epochs"]
    grad_clip    = mv.get("grad_clip", 1.0)

    X_rna_t = torch.from_numpy(X_rna).float()
    X_adt_t = torch.from_numpy(X_adt).float()
    dataset  = TensorDataset(X_rna_t, X_adt_t)
    loader   = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

    mvae = mvae.to(device)
    optim = torch.optim.Adam(mvae.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=n_epochs - warmup_eps, eta_min=lr / 10
    )

    history: list[dict] = []
    for epoch in range(1, n_epochs + 1):
        # KL warm-up
        kl_weight = min(1.0, epoch / max(warmup_eps, 1))

        mvae.train()
        epoch_stats: dict[str, float] = {}
        n_batches = 0

        for x_rna_b, x_adt_b in loader:
            x_rna_b = x_rna_b.to(device)
            x_adt_b = x_adt_b.to(device)

            optim.zero_grad()
            rna_hat, adt_hat, mu, logvar, z = mvae(x_rna_b, x_adt_b)
            loss, stats = mvae.elbo_loss(
                x_rna_b, x_adt_b, rna_hat, adt_hat,
                mu, logvar, z, warmup_weight=kl_weight
            )
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(mvae.parameters(), grad_clip)
            optim.step()

            for k, v in stats.items():
                epoch_stats[k] = epoch_stats.get(k, 0.0) + v
            n_batches += 1

        if epoch > warmup_eps:
            scheduler.step()

        avg = {k: v / n_batches for k, v in epoch_stats.items()}
        avg["epoch"] = epoch
        avg["lr"]    = optim.param_groups[0]["lr"]
        history.append(avg)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"MVAE epoch {epoch:3d}/{n_epochs} | "
                f"loss={avg['loss_total']:.4f} | "
                f"rna={avg['recon_rna']:.4f} | "
                f"adt={avg['recon_adt']:.4f} | "
                f"kl={avg['kl']:.4f}"
            )

    return mvae, history


@torch.no_grad()
def get_latent(
    mvae: MVAE,
    X_rna: np.ndarray,
    X_adt: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all cells → (μ, z_sampled)  both (n_cells, latent_dim).
    Returns mu (deterministic) and sampled z.
    """
    mvae.eval()
    all_mu, all_z = [], []

    X_rna_t = torch.from_numpy(X_rna).float()
    X_adt_t = torch.from_numpy(X_adt).float()

    for start in range(0, len(X_rna_t), batch_size):
        end  = min(start + batch_size, len(X_rna_t))
        xr   = X_rna_t[start:end].to(device)
        xa   = X_adt_t[start:end].to(device)
        mu, logvar = mvae.encode(xr, xa)
        z = reparameterise(mu, logvar)
        all_mu.append(mu.cpu().numpy())
        all_z.append(z.cpu().numpy())

    return np.vstack(all_mu), np.vstack(all_z)
