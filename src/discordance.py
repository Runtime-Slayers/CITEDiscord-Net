"""
discordance.py — RNA-Protein Discordance Analysis
===================================================

Systematic identification of cells where RNA and protein levels disagree.

Post-transcriptional regulation (PTR) causes mRNA and protein abundance
to be decoupled.  Discordant cells likely harbour active:
  • miRNA-mediated translational repression
  • Protein degradation / ubiquitination
  • Translational stalling (ER stress, etc.)

Method
------
1. For each protein p and cluster c, compute Spearman correlation
   between RNA (gene g) and CLR-normalised ADT.                      ρ(g, p | c)
2. Generate null distribution via permutation of protein labels within cluster.
3. Z-score discordance = (ρ_obs - μ_null) / σ_null
4. Per-cell discordance score: mean |Z| across all protein-gene pairs for that cell.
5. FDR correction (Benjamini-Hochberg) for gene-protein pair testing.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spearman correlation helper
# ---------------------------------------------------------------------------

def _spearman_vectorised(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Spearman r per column of Y vs. each column of X.

    X : (n_cells, n_genes)
    Y : (n_cells, n_proteins)
    Returns : (n_genes, n_proteins) correlation matrix.
    """
    # Rank transform
    from scipy.stats import rankdata
    X_rank = np.apply_along_axis(rankdata, 0, X)
    Y_rank = np.apply_along_axis(rankdata, 0, Y)

    # Normalise
    X_c = X_rank - X_rank.mean(0, keepdims=True)
    Y_c = Y_rank - Y_rank.mean(0, keepdims=True)

    X_std = (X_c**2).sum(0)**0.5 + 1e-8
    Y_std = (Y_c**2).sum(0)**0.5 + 1e-8

    corr = (X_c.T @ Y_c) / (X_std[:, None] * Y_std[None, :])  # (G, P)
    return corr


# ---------------------------------------------------------------------------
# Per-cluster permutation discordance test
# ---------------------------------------------------------------------------

def compute_discordance(
    X_rna: np.ndarray,               # (n_cells, n_hvg) Pearson residuals
    X_adt: np.ndarray,               # (n_cells, n_prot) CLR
    cluster_labels: np.ndarray,      # (n_cells,) int
    gene_names: list[str],
    protein_names: list[str],
    is_real: Optional[np.ndarray] = None,
    n_permutations: int = 500,
    alpha: float = 0.05,
    min_cells: int = 20,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Main discordance analysis.

    Returns
    -------
    results_df : DataFrame with per-(gene, protein, cluster) statistics
    cell_scores : ndarray (n_cells,)  — per-cell discordance magnitude
    """
    # Work only on real cells for statistical testing
    if is_real is None:
        is_real = np.ones(len(X_rna), dtype=bool)

    unique_clusters = np.unique(cluster_labels[is_real])
    records = []

    cell_discord = np.zeros(len(X_rna), dtype=np.float32)

    for cl in unique_clusters:
        cl_mask = is_real & (cluster_labels == cl)
        n_cl = cl_mask.sum()
        if n_cl < min_cells:
            continue

        Xr = X_rna[cl_mask]   # (n_cl, n_hvg)
        Xa = X_adt[cl_mask]   # (n_cl, n_prot)

        # Take only top 50 genes per protein by variance (speed)
        gene_var = Xr.var(axis=0)
        top_gene_idx = np.argsort(gene_var)[::-1][:50]
        Xr_top = Xr[:, top_gene_idx]
        top_genes = [gene_names[i] for i in top_gene_idx]

        # Observed correlations
        rho_obs = _spearman_vectorised(Xr_top, Xa)  # (50, n_prot)

        # Null distribution via permutation
        rho_null = np.zeros((n_permutations, rho_obs.shape[0], rho_obs.shape[1]))
        rng_perm = np.random.default_rng(int(cl) + 200)
        for k in range(n_permutations):
            perm_idx = rng_perm.permutation(n_cl)
            rho_null[k] = _spearman_vectorised(Xr_top, Xa[perm_idx])

        null_mu  = rho_null.mean(0)    # (50, n_prot)
        null_std = rho_null.std(0) + 1e-6

        Z = (rho_obs - null_mu) / null_std   # (50, n_prot)

        # Per-cell discordance: for each cell, mean |z| weighted by its residuals
        # Cell contribution to discordance = dot product of residuals with Z scores
        # Project cell's position in gene-protein correlation onto discordance axis
        cell_disc_cl = np.abs(Xr_top @ Z).mean(axis=1) / (np.abs(Xr_top).sum(axis=1) + 1e-8)
        cell_discord[cl_mask] += cell_disc_cl.astype(np.float32)

        # Record per-(gene, protein) pairs exceeding threshold
        sig_mask = np.abs(Z) > 2.576   # ~99% CI before FDR
        for gi, gn in enumerate(top_genes):
            for pi, pn in enumerate(protein_names):
                if sig_mask[gi, pi]:
                    records.append({
                        "gene":       gn,
                        "protein":    pn,
                        "cluster":    int(cl),
                        "n_cells":    n_cl,
                        "rho_obs":    float(rho_obs[gi, pi]),
                        "rho_null_mean": float(null_mu[gi, pi]),
                        "z_score":    float(Z[gi, pi]),
                    })

    if not records:
        log.warning("No significant discordant gene-protein pairs found. "
                    "Try increasing n_permutations or lowering alpha.")
        results_df = pd.DataFrame(columns=["gene","protein","cluster","n_cells",
                                            "rho_obs","rho_null_mean","z_score","fdr_padj"])
        return results_df, cell_discord

    results_df = pd.DataFrame(records)

    # BH FDR correction
    from scipy.stats import norm
    p_vals = 2 * norm.sf(np.abs(results_df["z_score"].values))
    results_df["p_value"] = p_vals
    results_df["fdr_padj"] = _bh_correction(p_vals)
    results_df = results_df.sort_values("z_score", key=np.abs, ascending=False)

    n_sig = (results_df["fdr_padj"] < alpha).sum()
    log.info(f"Discordance: {len(results_df)} tested, {n_sig} significant at FDR={alpha}")

    # Normalize cell scores to [0, 1]
    mx = cell_discord.max()
    if mx > 0:
        cell_discord = cell_discord / mx

    return results_df.reset_index(drop=True), cell_discord


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    padj  = np.empty(n)
    padj[order] = p_values[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        padj[order[i]] = min(padj[order[i]], padj[order[i + 1]])
    return np.clip(padj, 0, 1)


# ---------------------------------------------------------------------------
# Cell-level discordance summary
# ---------------------------------------------------------------------------

def summarise_discordance(
    cell_discord: np.ndarray,
    cluster_labels: np.ndarray,
    protein_names: list[str],
    results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Summarise discordance per cluster: mean score, top discordant proteins.
    """
    rows = []
    for cl in np.unique(cluster_labels):
        cl_mask = cluster_labels == cl
        mean_disc = cell_discord[cl_mask].mean()
        top_prots = (
            results_df[results_df["cluster"] == int(cl)]
            .sort_values("z_score", key=np.abs, ascending=False)
            ["protein"].unique()[:3].tolist()
        ) if len(results_df) > 0 else []
        rows.append({
            "cluster":       int(cl),
            "n_cells":       int(cl_mask.sum()),
            "mean_discord":  float(mean_disc),
            "top_proteins":  ", ".join(top_prots),
        })
    return pd.DataFrame(rows).sort_values("mean_discord", ascending=False)
