"""
cvae.py — Conditional VAE for Protein Abundance Prediction from RNA
======================================================================

Novel contribution:
  • Standard CITE-seq tools predict protein from RNA via linear regression
    or simple DNNs.  This CVAE models the full posterior p(x_adt | x_rna)
    including uncertainty, enabling downstream discordance uncertainty scoring.
  • ESM-2 LLM protein embeddings are injected into the decoder conditioning
    vector, providing biochemical context beyond cell-level RNA signal.

Architecture
-----------
  Encoder  : (x_adt | cond) → (μ_z, log σ²_z)
  Decoder  : (z, cond, esm_embed) → x_adt_hat
  Condition: RNA latent μ from MVAE  (no RNA data leakage at test time)

  At inference, z is sampled from the prior N(0,I) (no encoder needed),
  conditioned on the RNA latent vector alone.
"""
from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class CVAEEncoder(nn.Module):
    """Encodes (x_adt, cond) → (μ, log σ²)."""
    def __init__(self, adt_dim: int, cond_dim: int,
                 hidden: list, latent: int, dropout: float):
        super().__init__()
        in_dim = adt_dim + cond_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.body     = nn.Sequential(*layers)
        self.mu_head  = nn.Linear(prev, latent)
        self.lv_head  = nn.Linear(prev, latent)

    def forward(self, x_adt: torch.Tensor, cond: torch.Tensor):
        inp = torch.cat([x_adt, cond], dim=1)
        h   = self.body(inp)
        return self.mu_head(h), self.lv_head(h)


class CVAEDecoder(nn.Module):
    """
    Decodes (z, cond, esm_embed) → x_adt_hat.

    esm_embed: ESM-2 protein language model embedding summed over the
    protein panel. Provides per-cell biochemical prior information beyond
    what transcriptomics alone can capture.
    """
    def __init__(self, latent: int, cond_dim: int, esm_dim: int,
                 adt_dim: int, hidden: list, dropout: float):
        super().__init__()
        in_dim = latent + cond_dim + esm_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, adt_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor,
                esm_embed: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, cond, esm_embed], dim=1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Full CVAE
# ---------------------------------------------------------------------------

class ProteinCVAE(nn.Module):
    """
    Conditional VAE: predict protein from RNA latent vector.

    At training time: both RNA latent (cond) and true ADT (x_adt) are available.
    At inference:     only RNA latent is given; z is sampled from N(0, I).
    """

    def __init__(self, cfg: dict, n_proteins: int, esm_dim: int = 320):
        super().__init__()
        cv = cfg["cvae"]
        latent   = cv["latent_dim"]
        cond_dim = cv["condition_dim"]
        hidden   = cv["hidden_dims"]
        dropout  = cv.get("dropout", 0.1)

        self.latent_dim = latent
        self.esm_dim    = esm_dim
        self.use_esm    = cfg["cvae"].get("use_esm_features", True) and (esm_dim > 0)

        effective_esm = esm_dim if self.use_esm else 0

        self.encoder = CVAEEncoder(n_proteins, cond_dim, hidden, latent, dropout)
        self.decoder = CVAEDecoder(latent, cond_dim, effective_esm, n_proteins, hidden, dropout)

    def forward(
        self,
        x_adt: torch.Tensor,
        cond: torch.Tensor,
        esm_embed: Optional[torch.Tensor] = None,
    ):
        """Training forward pass."""
        mu, logvar = self.encoder(x_adt, cond)
        z = _reparam(mu, logvar)

        if esm_embed is None:
            esm_embed = torch.zeros(x_adt.shape[0], self.esm_dim, device=x_adt.device)
        if not self.use_esm:
            esm_embed = torch.zeros(x_adt.shape[0], 0, device=x_adt.device)

        x_hat = self.decoder(z, cond, esm_embed)
        return x_hat, mu, logvar

    @torch.no_grad()
    def predict(
        self,
        cond: torch.Tensor,
        esm_embed: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Inference: sample protein predictions from the prior."""
        self.eval()
        B = cond.shape[0]
        preds = []
        if esm_embed is None or not self.use_esm:
            dim = self.esm_dim if self.use_esm else 0
            esm_embed = torch.zeros(B, dim, device=cond.device)
        for _ in range(n_samples):
            z = torch.randn(B, self.latent_dim, device=cond.device)
            preds.append(self.decoder(z, cond, esm_embed))
        return torch.stack(preds, dim=0).mean(dim=0)    # average over samples


def _reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return mu
    std = (0.5 * logvar).exp().clamp(1e-4, 4.0)
    return mu + std * torch.randn_like(std)


# ---------------------------------------------------------------------------
# CVAE Loss Function
# ---------------------------------------------------------------------------

def cvae_loss(
    x_adt: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Standard CVAE ELBO.
    Reconstruction: MSE (ADT data is CLR-normalised continuous)
    KL: analytical Gaussian closed-form
    """
    recon = F.mse_loss(x_hat, x_adt, reduction="mean")
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    total = recon + beta * kl
    return total, {"cvae_loss": total.item(), "cvae_recon": recon.item(), "cvae_kl": kl.item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cvae(
    cvae: ProteinCVAE,
    rna_latent: np.ndarray,   # RNA μ from MVAE  (n_cells, latent_dim)
    X_adt_norm: np.ndarray,   # CLR ADT          (n_cells, n_proteins)
    esm_embed: np.ndarray,    # ESM-2 embeddings (n_proteins, esm_dim) OR None
    cfg: dict,
    device: torch.device,
    is_real: Optional[np.ndarray] = None,
) -> Tuple[ProteinCVAE, list[dict]]:
    """
    Train CVAE on (RNA latent, ADT) pairs.
    Uses only real cells for training (augmented cells filtered out if is_real given).
    """
    cv = cfg["cvae"]
    lr       = cv["lr"]
    bs       = cv["batch_size"]
    n_epochs = cv["n_epochs"]

    # Use only real cells for CVAE training
    mask = is_real if is_real is not None else np.ones(len(rna_latent), dtype=bool)
    rna_t = torch.from_numpy(rna_latent[mask]).float()
    adt_t = torch.from_numpy(X_adt_norm[mask]).float()

    # Build per-cell ESM embedding  (broadcast protein embedding sum across cells)
    n_prot = X_adt_norm.shape[1]
    esm_dim = cvae.esm_dim if cvae.use_esm else 0
    if cvae.use_esm and esm_embed is not None and esm_embed.size > 0:
        # average ESM embedding of all proteins in the panel
        esm_mean = esm_embed.mean(axis=0)  # (esm_dim,)
        esm_t = torch.from_numpy(
            np.tile(esm_mean, (mask.sum(), 1)).astype(np.float32)
        )
    else:
        esm_t = torch.zeros(mask.sum(), 0)

    dataset = TensorDataset(rna_t, adt_t, esm_t)
    loader  = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

    cvae = cvae.to(device)
    optim     = torch.optim.Adam(cvae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)

    history = []
    for epoch in range(1, n_epochs + 1):
        cvae.train()
        epoch_stats: dict[str, float] = {}
        n_batches = 0
        for cond_b, adt_b, esm_b in loader:
            cond_b = cond_b.to(device)
            adt_b  = adt_b.to(device)
            esm_b  = esm_b.to(device) if esm_b.numel() > 0 else None

            optim.zero_grad()
            x_hat, mu, logvar = cvae(adt_b, cond_b, esm_b)
            loss, stats = cvae_loss(adt_b, x_hat, mu, logvar)
            loss.backward()
            optim.step()
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats.get(k, 0.0) + v
            n_batches += 1

        avg = {k: v / n_batches for k, v in epoch_stats.items()}
        avg["epoch"] = epoch
        scheduler.step(avg["cvae_loss"])
        history.append(avg)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"CVAE epoch {epoch:3d}/{n_epochs} | "
                     f"loss={avg['cvae_loss']:.4f} | "
                     f"recon={avg['cvae_recon']:.4f} | "
                     f"kl={avg['cvae_kl']:.4f}")

    return cvae, history


@torch.no_grad()
def predict_protein(
    cvae: ProteinCVAE,
    rna_latent: np.ndarray,
    esm_embed: Optional[np.ndarray],
    device: torch.device,
    n_samples: int = 10,
    batch_size: int = 512,
) -> np.ndarray:
    """Return per-cell predicted protein  (n_cells, n_proteins)."""
    cvae.eval()
    preds = []
    n = len(rna_latent)
    esm_dim = cvae.esm_dim if cvae.use_esm else 0

    if cvae.use_esm and esm_embed is not None and esm_embed.size > 0:
        esm_mean = esm_embed.mean(axis=0).astype(np.float32)
    else:
        esm_mean = np.zeros(esm_dim, dtype=np.float32)

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        cond  = torch.from_numpy(rna_latent[start:end]).float().to(device)
        esm_b = torch.from_numpy(np.tile(esm_mean, (end-start, 1))).float().to(device)
        if esm_dim == 0:
            esm_b = None
        pred  = cvae.predict(cond, esm_b, n_samples=n_samples)
        preds.append(pred.cpu().numpy())

    return np.vstack(preds)
