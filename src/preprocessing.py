"""
preprocessing.py — CITE-seq Quality Control & Normalisation
=============================================================

Steps:
  1. Parse 10X h5 file → split RNA vs ADT matrices
  2. QC filtering (min/max genes, mitochondrial content)
  3. RNA normalisation: log1p + Pearson residuals (analytically regularised)
  4. ADT normalisation: Centred Log-Ratio (CLR) per cell
  5. Highly variable gene (HVG) selection
  6. PCA on RNA
  7. Data augmentation via Gaussian perturbation of the normalised data
     (augmented cells explicitly labelled; combined with real data)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import digamma
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# h5 → numpy parsing
# ---------------------------------------------------------------------------

def load_10x_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, list, list, Optional[np.ndarray]]:
    """
    Parse 10X Genomics CellRanger h5 into RNA and ADT matrices.

    Returns
    -------
    X_rna    : ndarray (n_cells, n_genes)  — raw counts
    X_adt    : ndarray (n_cells, n_proteins) — raw ADT counts
    gene_names  : list[str]
    protein_names : list[str]
    cell_types  : ndarray or None (if known ground truth available)
    """
    with h5py.File(h5_path, "r") as f:
        matrix_grp = f["matrix"]
        feature_type = np.array(matrix_grp["features"]["feature_type"]).astype(str)
        names        = np.array(matrix_grp["features"]["name"]).astype(str)

        data    = np.array(matrix_grp["data"])
        indices = np.array(matrix_grp["indices"])
        indptr  = np.array(matrix_grp["indptr"])
        shape   = tuple(matrix_grp["shape"][:])  # (n_features, n_cells)

        X_full = csr_matrix((data, indices, indptr), shape=shape).T.toarray().astype(np.float32)
        # X_full: (n_cells, n_features)

        rna_mask  = feature_type == "Gene Expression"
        adt_mask  = feature_type == "Antibody Capture"
        gene_names    = list(names[rna_mask])
        protein_names = list(names[adt_mask])

        X_rna = X_full[:, rna_mask]
        X_adt = X_full[:, adt_mask]

        # Try reading cell types (present in fallback data)
        cell_types = None
        if "adt_matrix" in f:
            adt_grp = f["adt_matrix"]
            if "cell_types" in adt_grp:
                cell_types = np.array(adt_grp["cell_types"]).astype(str)
            if "data" in adt_grp and X_adt.sum() == 0:
                # Use the separately stored ADT matrix in the fallback format
                X_adt = np.array(adt_grp["data"])
                if "proteins" in adt_grp:
                    protein_names = list(np.array(adt_grp["proteins"]).astype(str))

    log.info(f"Loaded  RNA: {X_rna.shape}, ADT: {X_adt.shape}")
    return X_rna, X_adt, gene_names, protein_names, cell_types


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def quality_control(
    X_rna: np.ndarray,
    gene_names: list,
    min_genes: int = 200,
    max_genes: int = 6000,
    max_mito_frac: float = 0.20,
) -> np.ndarray:
    """Return boolean mask of cells passing QC."""
    n_genes_per_cell = (X_rna > 0).sum(axis=1)
    total_counts     = X_rna.sum(axis=1)

    mito_idx = [i for i, g in enumerate(gene_names) if g.upper().startswith("MT-")]
    if mito_idx:
        mito_counts  = X_rna[:, mito_idx].sum(axis=1)
        mito_frac    = mito_counts / np.maximum(total_counts, 1)
    else:
        mito_frac = np.zeros(X_rna.shape[0], dtype=np.float32)

    mask = (
        (n_genes_per_cell >= min_genes) &
        (n_genes_per_cell <= max_genes) &
        (mito_frac <= max_mito_frac)
    )
    log.info(f"QC: {mask.sum()}/{len(mask)} cells pass")
    return mask


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def clr_normalize(X_adt: np.ndarray) -> np.ndarray:
    """
    Centred Log-Ratio (CLR) normalisation per cell.

    CLR(x_j) = log(x_j / geometric_mean(x + 1))

    Standard pre-processing for ADT data in CITE-seq (Hao et al. 2021).
    """
    X_clr = np.log1p(X_adt)
    row_means = X_clr.mean(axis=1, keepdims=True)
    return (X_clr - row_means).astype(np.float32)


def pearson_residuals(X_rna: np.ndarray, theta: float = 100.0, clip: float = 30.0) -> np.ndarray:
    """
    Analytic Pearson residuals (Lause, Berens & Kobak 2021).

    Analytically regularised residuals under a negative-binomial model.
    Variance-stabilises read counts without arbitrary log-scaling choices.

    theta : NB overdispersion parameter (100 ≈ Poisson-like for scRNA-seq)
    clip  : symmetric clipping of residuals to [-clip, +clip]
    """
    # Library-size normalisation to estimate expected counts μ
    total = X_rna.sum(axis=1, keepdims=True)
    gene_means = X_rna.mean(axis=0, keepdims=True)     # (1, G)
    mu = total * (gene_means / (gene_means.sum() + 1e-8))  # (n_cells, G)

    # Pearson residuals: (x - mu) / sqrt(mu + mu^2/theta)
    var   = mu + mu**2 / theta
    resid = (X_rna - mu) / np.sqrt(var + 1e-8)
    resid = np.clip(resid, -clip, clip)
    return resid.astype(np.float32)


# ---------------------------------------------------------------------------
# Highly Variable Genes
# ---------------------------------------------------------------------------

def select_hvg(X_raw: np.ndarray, n_top: int = 2000, min_cells: int = 10) -> np.ndarray:
    """
    Select highly variable genes by normalised dispersion (Satija lab method).

    Returns boolean index array of length n_genes.
    """
    # Filter by min cells
    expressed = (X_raw > 0).sum(axis=0)  # (n_genes,)
    valid = expressed >= min_cells

    # Log-normalise per cell then compute per-gene dispersion
    lib_size = X_raw.sum(axis=1, keepdims=True)
    X_norm   = np.log1p(X_raw / (lib_size + 1e-8) * 1e4)

    gene_mean = X_norm.mean(axis=0)           # (G,)
    gene_var  = X_norm.var(axis=0, ddof=1)    # (G,)
    disp       = gene_var / (gene_mean + 1e-8)  # coefficient of variation

    # Normalise dispersion within bins of mean expression
    n_bins = 20
    mean_bins = np.percentile(gene_mean[valid], np.linspace(0, 100, n_bins + 1))
    mean_bins[-1] += 1e-6
    disp_norm = np.zeros_like(disp)
    for b in range(n_bins):
        in_bin = valid & (gene_mean >= mean_bins[b]) & (gene_mean < mean_bins[b+1])
        if in_bin.sum() > 0:
            mu_d  = disp[in_bin].mean()
            std_d = disp[in_bin].std() + 1e-8
            disp_norm[in_bin] = (disp[in_bin] - mu_d) / std_d

    # Rank by normalised dispersion, return top n_top
    disp_norm[~valid] = -np.inf
    hvg_idx = np.argsort(disp_norm)[::-1][:n_top]
    mask = np.zeros(X_raw.shape[1], dtype=bool)
    mask[hvg_idx] = True
    log.info(f"HVG selection: {mask.sum()} genes (of {(valid).sum()} expressed)")
    return mask


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(X: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, PCA]:
    """Fit PCA and return (scores, fitted_pca)."""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)
    var_expl = pca.explained_variance_ratio_.cumsum()[-1]
    log.info(f"PCA {n_components} PCs explain {var_expl*100:.1f}% variance")
    return scores.astype(np.float32), pca


# ---------------------------------------------------------------------------
# Data Augmentation (biologically informed VAE-style perturbation)
# ---------------------------------------------------------------------------

def augment_cells(
    X_rna_norm: np.ndarray,
    X_adt_norm: np.ndarray,
    n_augmented: int = 3000,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate augmented cells by:
      1. Sampling neighbour pairs from the normalised manifold
      2. Interpolating + adding small Gaussian perturbations

    This is akin to in-silico cell line expansion (data manifold smoothing).
    Returns augmented RNA, augmented ADT, and boolean is_real flag.

    The augmented data is explicitly labelled so downstream models can
    track which cells are real vs. generated.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_real = X_rna_norm.shape[0]
    # Sample random pairs
    idx_a = rng.integers(0, n_real, size=n_augmented)
    idx_b = rng.integers(0, n_real, size=n_augmented)
    alpha = rng.uniform(0.3, 0.7, size=(n_augmented, 1)).astype(np.float32)

    X_rna_aug = (
        alpha * X_rna_norm[idx_a] +
        (1 - alpha) * X_rna_norm[idx_b] +
        rng.normal(0, noise_std, size=(n_augmented, X_rna_norm.shape[1])).astype(np.float32)
    )
    X_adt_aug = (
        alpha * X_adt_norm[idx_a] +
        (1 - alpha) * X_adt_norm[idx_b] +
        rng.normal(0, noise_std, size=(n_augmented, X_adt_norm.shape[1])).astype(np.float32)
    )

    is_real = np.array([True] * n_real + [False] * n_augmented, dtype=bool)

    X_rna_combined = np.vstack([X_rna_norm, X_rna_aug])
    X_adt_combined = np.vstack([X_adt_norm, X_adt_aug])

    log.info(f"Augmented: {n_real} real + {n_augmented} synthetic = {len(is_real)} total cells")
    return X_rna_combined, X_adt_combined, is_real


# ---------------------------------------------------------------------------
# Master pipeline call
# ---------------------------------------------------------------------------

def run_preprocessing(
    h5_path: Path,
    cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        X_rna_norm     : (n_total, n_hvg)  Pearson-residual RNA
        X_adt_norm     : (n_total, n_prot)  CLR ADT
        X_pca          : (n_total, n_pcs)   PCA of RNA
        is_real        : (n_total,) bool
        gene_names_hvg : list[str]
        protein_names  : list[str]
        cell_types_raw : ndarray or None
        n_real         : int
        n_augmented    : int
    """
    data_cfg = cfg.get("data", {})
    aug_cfg  = cfg.get("data", {})

    # 1. Load
    X_rna, X_adt, gene_names, protein_names, cell_types_raw = load_10x_h5(h5_path)

    # 2. QC
    qc_mask = quality_control(
        X_rna, gene_names,
        min_genes=data_cfg.get("min_genes_per_cell", 200),
        max_genes=data_cfg.get("max_genes_per_cell", 6000),
        max_mito_frac=data_cfg.get("max_mito_fraction", 0.20),
    )
    X_rna = X_rna[qc_mask]
    X_adt = X_adt[qc_mask]
    if cell_types_raw is not None:
        cell_types_raw = cell_types_raw[qc_mask]

    # 3. HVG
    hvg_mask = select_hvg(
        X_rna,
        n_top=data_cfg.get("n_highly_variable_genes", 2000),
        min_cells=data_cfg.get("min_cells_per_gene", 10),
    )
    X_rna_hvg      = X_rna[:, hvg_mask]
    gene_names_hvg  = [g for g, m in zip(gene_names, hvg_mask) if m]

    # 4. Normalise
    X_rna_norm = pearson_residuals(X_rna_hvg)
    X_adt_norm = clr_normalize(X_adt)

    # 5. PCA
    X_pca_real, _ = run_pca(X_rna_norm, n_components=data_cfg.get("n_pca_components", 50))

    # 6. Augmentation
    n_real = X_rna_norm.shape[0]
    n_aug  = aug_cfg.get("n_augmented_cells", 3000)
    noise  = aug_cfg.get("augmentation_noise_std", 0.05)
    X_rna_combined, X_adt_combined, is_real = augment_cells(
        X_rna_norm, X_adt_norm, n_augmented=n_aug, noise_std=noise, rng=rng
    )

    # 7. PCA for combined (recompute scores using fitted PCA of real cells)
    pca_model = PCA(n_components=min(data_cfg.get("n_pca_components", 50),
                                     X_rna_combined.shape[1] - 1), random_state=42)
    X_pca = pca_model.fit_transform(X_rna_combined).astype(np.float32)

    return {
        "X_rna_norm":    X_rna_combined,     # (n_total, n_hvg)
        "X_adt_norm":    X_adt_combined,     # (n_total, n_prot)
        "X_pca":         X_pca,              # (n_total, n_pcs)
        "is_real":       is_real,            # (n_total,)
        "gene_names_hvg": gene_names_hvg,
        "protein_names": protein_names,
        "cell_types_raw": cell_types_raw,    # None or (n_cells,)
        "n_real":        n_real,
        "n_augmented":   n_aug,
    }
