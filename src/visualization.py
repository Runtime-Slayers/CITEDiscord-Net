"""
visualization.py — Publication-Quality Figures for CITEDiscord-Net
====================================================================

Generates 14 figures covering:
  1.  QC violin plots (gates applied)
  2.  HVG mean-variance scatter
  3.  MVAE training loss curves (RNA, ADT, KL)
  4.  UMAP coloured by Leiden cluster
  5.  UMAP coloured by cell type annotation
  6.  UMAP coloured by per-cell discordance score
  7.  UMAP coloured by real vs. augmented cells
  8.  RNA-protein observed vs. predicted scatter (CVAE)
  9.  Per-protein prediction Spearman r (bar chart)
  10. Top discordant gene-protein pairs (signed Z heatmap per cluster)
  11. Cell communication chord / circos diagram
  12. Communication matrix heatmap
  13. ESM-2 protein embedding UMAP
  14. Cell-type marker gene dot plot
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PALETTE = [
    "#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261",
    "#264653","#8338EC","#06D6A0","#118AB2","#FFB703",
    "#FB5607","#3A86FF","#8AC926","#FF006E","#FFBE0B",
]


def _get_colors(n: int) -> list[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _save(fig, path: Path, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Figure saved: {path.name}")


# ---------------------------------------------------------------------------
# 1. QC metrics violin
# ---------------------------------------------------------------------------

def plot_qc_violins(pipeline, out_dir: Path, dpi: int = 150):
    """Violin plots: n_genes_per_cell, total_counts, % mito."""
    data = pipeline.data
    X_rna_raw = None
    # Load via h5 if available
    try:
        import h5py
        with h5py.File(pipeline.h5_path, "r") as f:
            # Attempt to read a subset of raw counts for QC
            mat = f["matrix"]
            feat_type = np.array(mat["features"]["feature_type"]).astype(str)
            rna_mask  = feat_type == "Gene Expression"
            data_arr  = np.array(mat["data"])
            idx_arr   = np.array(mat["indices"])
            ptr_arr   = np.array(mat["indptr"])
            shape     = tuple(mat["shape"][:])
            from scipy.sparse import csr_matrix
            X_full = csr_matrix((data_arr, idx_arr, ptr_arr), shape=shape)
            X_rna_raw = np.asarray(X_full[:, rna_mask].T.sum(axis=1)).flatten()
    except Exception:
        pass

    n_cells = data["n_real"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Cell QC Metrics", fontsize=14, fontweight="bold")

    # Genes per cell (derived from HVG-filtered data)
    n_genes = (data["X_rna_norm"][:n_cells] != 0).sum(axis=1)
    axes[0].violinplot(n_genes, positions=[1], showmedians=True)
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["Cells"])
    axes[0].set_ylabel("# HVGs detected")
    axes[0].set_title("Genes per Cell (HVG subset)")

    # ADT total counts
    adt_total = data["X_adt_norm"][:n_cells].sum(axis=1)
    axes[1].violinplot(adt_total, positions=[1], showmedians=True)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Cells"])
    axes[1].set_ylabel("Sum(CLR ADT)")
    axes[1].set_title("CLR-ADT total per cell")

    plt.tight_layout()
    _save(fig, out_dir / "01_qc_violins.pdf", dpi)


# ---------------------------------------------------------------------------
# 2. HVG selection scatter
# ---------------------------------------------------------------------------

def plot_hvg_scatter(pipeline, out_dir: Path, dpi: int = 150):
    x = pipeline.data["X_rna_norm"][:pipeline.data["n_real"]].mean(axis=0)
    y = pipeline.data["X_rna_norm"][:pipeline.data["n_real"]].var(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=1, alpha=0.3, color="#457B9D", label="HVG")
    ax.set_xlabel("Mean Pearson residual")
    ax.set_ylabel("Variance")
    ax.set_title("Highly Variable Genes (Pearson Residual space)")
    ax.legend(markerscale=5)
    plt.tight_layout()
    _save(fig, out_dir / "02_hvg_scatter.pdf", dpi)


# ---------------------------------------------------------------------------
# 3. MVAE loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(pipeline, out_dir: Path, dpi: int = 150):
    if not hasattr(pipeline, "mvae_history") or not pipeline.mvae_history:
        return

    hist = pd.DataFrame(pipeline.mvae_history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("MVAE Training Loss", fontsize=14, fontweight="bold")

    for ax, col, label, colour in zip(
        axes,
        ["loss_total", "kl", "recon_rna"],
        ["Total ELBO", "KL Divergence", "RNA Reconstruction MSE"],
        ["#E63946", "#457B9D", "#2A9D8F"],
    ):
        ax.plot(hist["epoch"], hist[col], color=colour, lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / "03_mvae_loss_curves.pdf", dpi)


# ---------------------------------------------------------------------------
# 4. UMAP — cluster
# ---------------------------------------------------------------------------

def plot_umap_clusters(pipeline, out_dir: Path, dpi: int = 150):
    emb = pipeline.umap_emb[:pipeline.data["n_real"]]
    cl  = pipeline.cluster_labels[:pipeline.data["n_real"]]
    unique = sorted(np.unique(cl))
    colors = _get_colors(len(unique))

    fig, ax = plt.subplots(figsize=(8, 7))
    for c, colour in zip(unique, colors):
        mask = cl == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.5,
                   color=colour, label=f"Cluster {c}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP — Leiden Clusters (real cells)")
    ax.legend(markerscale=4, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    _save(fig, out_dir / "04_umap_clusters.pdf", dpi)


# ---------------------------------------------------------------------------
# 5. UMAP — cell type
# ---------------------------------------------------------------------------

def plot_umap_celltypes(pipeline, out_dir: Path, dpi: int = 150):
    emb = pipeline.umap_emb[:pipeline.data["n_real"]]
    ct  = pipeline.cell_types
    unique = sorted(set(ct))
    colors = _get_colors(len(unique))

    fig, ax = plt.subplots(figsize=(9, 7))
    for c, colour in zip(unique, colors):
        mask = ct == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.5,
                   color=colour, label=c)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP — Predicted Cell Types")
    ax.legend(markerscale=4, loc="upper right", fontsize=7, ncol=1)
    plt.tight_layout()
    _save(fig, out_dir / "05_umap_celltypes.pdf", dpi)


# ---------------------------------------------------------------------------
# 6. UMAP — discordance score
# ---------------------------------------------------------------------------

def plot_umap_discordance(pipeline, out_dir: Path, dpi: int = 150):
    emb   = pipeline.umap_emb[:pipeline.data["n_real"]]
    score = pipeline.cell_discord[:pipeline.data["n_real"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=score, s=2,
                    cmap="YlOrRd", vmin=0, vmax=score.max())
    plt.colorbar(sc, ax=ax, label="Discordance Score")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("RNA-Protein Discordance Score per Cell")
    plt.tight_layout()
    _save(fig, out_dir / "06_umap_discordance.pdf", dpi)


# ---------------------------------------------------------------------------
# 7. UMAP — real vs. augmented
# ---------------------------------------------------------------------------

def plot_umap_real_vs_aug(pipeline, out_dir: Path, dpi: int = 150):
    emb  = pipeline.umap_emb
    real = pipeline.data["is_real"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb[~real, 0], emb[~real, 1], s=1, alpha=0.3,
               color="#F4A261", label="Augmented")
    ax.scatter(emb[real, 0],  emb[real, 1],  s=1, alpha=0.5,
               color="#457B9D", label="Real")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Real vs. VAE-Augmented Cells in Joint Latent Space")
    ax.legend(markerscale=5)
    plt.tight_layout()
    _save(fig, out_dir / "07_umap_real_vs_aug.pdf", dpi)


# ---------------------------------------------------------------------------
# 8. CVAE: observed vs. predicted protein
# ---------------------------------------------------------------------------

def plot_protein_scatter(pipeline, out_dir: Path, dpi: int = 150):
    real = pipeline.data["is_real"]
    obs  = pipeline.data["X_adt_norm"][real].flatten()
    pred = pipeline.protein_pred[real].flatten()

    fig, ax = plt.subplots(figsize=(7, 6))
    # Subsample for readability
    idx = np.random.default_rng(0).choice(len(obs), size=min(10000, len(obs)), replace=False)
    ax.hexbin(obs[idx], pred[idx], gridsize=60, cmap="Blues", mincnt=1)
    ax.set_xlabel("Observed CLR ADT")
    ax.set_ylabel("CVAE Predicted ADT")
    ax.set_title("CVAE Protein Prediction (all proteins × real cells)")
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(obs, pred)
        ax.text(0.05, 0.92, f"Spearman ρ = {r:.3f}", transform=ax.transAxes,
                fontsize=11, color="black")
    except Exception:
        pass
    plt.tight_layout()
    _save(fig, out_dir / "08_protein_pred_scatter.pdf", dpi)


# ---------------------------------------------------------------------------
# 9. Per-protein Spearman bar
# ---------------------------------------------------------------------------

def plot_per_protein_spearman(pipeline, out_dir: Path, dpi: int = 150):
    from scipy.stats import spearmanr
    real   = pipeline.data["is_real"]
    obs    = pipeline.data["X_adt_norm"][real]
    pred   = pipeline.protein_pred[real]
    prots  = pipeline.data["protein_names"]
    n_p    = min(obs.shape[1], len(prots))

    rho_vals = []
    for j in range(n_p):
        r, _ = spearmanr(obs[:, j], pred[:, j])
        rho_vals.append(float(r))

    fig, ax = plt.subplots(figsize=(max(8, n_p * 0.8), 5))
    colours = ["#2A9D8F" if r > 0.3 else "#E63946" for r in rho_vals]
    ax.bar(prots[:n_p], rho_vals, color=colours)
    ax.axhline(0.3, ls="--", color="gray", alpha=0.7, label="ρ = 0.3 threshold")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Protein"); ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Protein CVAE Prediction Accuracy")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    _save(fig, out_dir / "09_per_protein_spearman.pdf", dpi)


# ---------------------------------------------------------------------------
# 10. Discordance heatmap (cluster × top proteins)
# ---------------------------------------------------------------------------

def plot_discordance_heatmap(pipeline, out_dir: Path, dpi: int = 150):
    df = pipeline.discord_df
    if df is None or len(df) == 0:
        return

    pivot = df.pivot_table(
        index="cluster", columns="protein",
        values="z_score", aggfunc="mean"
    ).fillna(0)

    # Top 10 most discordant proteins
    top_prots = pivot.abs().max(axis=0).nlargest(min(10, pivot.shape[1])).index
    pivot = pivot[top_prots]

    fig, ax = plt.subplots(figsize=(max(8, len(top_prots)), max(5, len(pivot) * 0.6)))
    import matplotlib.cm as cm
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-4, vmax=4)
    plt.colorbar(im, ax=ax, label="Mean Z-score")
    ax.set_xticks(range(len(top_prots)))
    ax.set_xticklabels(list(top_prots), rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"Cluster {c}" for c in pivot.index], fontsize=9)
    ax.set_title("RNA-Protein Discordance Z-score (per cluster × protein)")
    plt.tight_layout()
    _save(fig, out_dir / "10_discordance_heatmap.pdf", dpi)


# ---------------------------------------------------------------------------
# 11. Communication matrix heatmap
# ---------------------------------------------------------------------------

def plot_comm_matrix(pipeline, out_dir: Path, dpi: int = 150):
    mat   = pipeline.comm_matrix
    names = [f"Cl{i}" for i in range(mat.shape[0])]
    if mat is None or mat.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, mat.shape[0] * 0.8),
                                    max(6, mat.shape[0] * 0.8)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Interaction Strength")
    ax.set_xticks(range(mat.shape[0])); ax.set_xticklabels(names, rotation=45)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(names)
    ax.set_title("Cell–Cell Communication Matrix\n(Sender → Receiver)")
    ax.set_xlabel("Receiver"); ax.set_ylabel("Sender")
    plt.tight_layout()
    _save(fig, out_dir / "11_comm_matrix.pdf", dpi)


# ---------------------------------------------------------------------------
# 12. Top LR pairs bar chart
# ---------------------------------------------------------------------------

def plot_top_lr_pairs(pipeline, out_dir: Path, dpi: int = 150, top_n: int = 15):
    df = pipeline.comm_df
    if df is None or len(df) == 0:
        return
    top = df.nlargest(top_n, "interaction_score")
    labels = [f"{r['ligand']}→{r['receptor']}\n({r['sender']}→{r['receiver']})"
              for _, r in top.iterrows()]
    scores = top["interaction_score"].values

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.5)))
    colours = _get_colors(top_n)
    ax.barh(range(len(labels)), scores[::-1], color=colours[::-1])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("Interaction Score"); ax.set_title("Top L-R Pair Interactions")
    plt.tight_layout()
    _save(fig, out_dir / "12_top_lr_pairs.pdf", dpi)


# ---------------------------------------------------------------------------
# 13. ESM-2 embedding UMAP
# ---------------------------------------------------------------------------

def plot_esm2_umap(pipeline, out_dir: Path, dpi: int = 150):
    emb_mat = pipeline.esm_embeddings
    if emb_mat is None or emb_mat.shape[0] < 3:
        return
    prots = pipeline.data["protein_names"]

    try:
        from sklearn.decomposition import PCA
        if emb_mat.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            proj = pca.fit_transform(emb_mat)
        else:
            proj = emb_mat[:, :2]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(proj[:, 0], proj[:, 1], s=60, c=_get_colors(len(prots))[:len(prots)])
        for i, p in enumerate(prots[:len(proj)]):
            ax.annotate(p, (proj[i, 0], proj[i, 1]), fontsize=8, alpha=0.8)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title("ESM-2 Protein Language Model Embeddings (PCA)")
        plt.tight_layout()
        _save(fig, out_dir / "13_esm2_protein_pca.pdf", dpi)
    except Exception as e:
        log.warning(f"ESM-2 plot skipped: {e}")


# ---------------------------------------------------------------------------
# 14. Marker gene dot plot
# ---------------------------------------------------------------------------

def plot_marker_dotplot(pipeline, out_dir: Path, dpi: int = 150):
    from src.clustering import PBMC_MARKERS
    data      = pipeline.data
    cl_labels = pipeline.cluster_labels[:data["n_real"]]
    X_rna     = data["X_rna_norm"][:data["n_real"]]
    gene_idx  = {g.upper(): i for i, g in enumerate(data["gene_names_hvg"])}

    clusters = sorted(np.unique(cl_labels))
    # Use at most 5 markers per type
    all_markers = []
    for mgs in PBMC_MARKERS.values():
        for g in mgs[:2]:
            if g.upper() in gene_idx and g not in all_markers:
                all_markers.append(g)
    all_markers = all_markers[:20]

    if not all_markers:
        return

    mean_exp  = np.zeros((len(clusters), len(all_markers)))
    frac_exp  = np.zeros((len(clusters), len(all_markers)))
    for ci, cl in enumerate(clusters):
        c_mask = cl_labels == cl
        for gi, g in enumerate(all_markers):
            gi_idx = gene_idx.get(g.upper())
            if gi_idx is not None:
                vals = X_rna[c_mask, gi_idx]
                mean_exp[ci, gi] = vals.mean()
                frac_exp[ci, gi] = (vals > 0).mean()

    fig, ax = plt.subplots(figsize=(max(10, len(all_markers)), max(4, len(clusters) * 0.6)))
    for ci in range(len(clusters)):
        for gi in range(len(all_markers)):
            size  = frac_exp[ci, gi] * 200
            color_val = mean_exp[ci, gi]
            ax.scatter(gi, ci, s=size, c=[[color_val]], cmap="Reds",
                       vmin=0, vmax=mean_exp.max(), edgecolors="gray", linewidths=0.3)

    ax.set_xticks(range(len(all_markers)))
    ax.set_xticklabels(all_markers, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters])
    ax.set_title("Marker Gene Dot Plot\n(size ∝ fraction expressing, colour ∝ mean expression)")
    plt.tight_layout()
    _save(fig, out_dir / "14_marker_dot_plot.pdf", dpi)


# ---------------------------------------------------------------------------
# Master call
# ---------------------------------------------------------------------------

def generate_all_figures(pipeline, results_dir: Path, dpi: int = 150):
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    log.info(f"Generating figures in {fig_dir}")

    steps = [
        (plot_qc_violins,           "QC violins"),
        (plot_hvg_scatter,          "HVG scatter"),
        (plot_loss_curves,          "MVAE loss curves"),
        (plot_umap_clusters,        "UMAP clusters"),
        (plot_umap_celltypes,       "UMAP cell types"),
        (plot_umap_discordance,     "UMAP discordance"),
        (plot_umap_real_vs_aug,     "UMAP real vs. augmented"),
        (plot_protein_scatter,      "Protein prediction scatter"),
        (plot_per_protein_spearman, "Per-protein Spearman"),
        (plot_discordance_heatmap,  "Discordance heatmap"),
        (plot_comm_matrix,          "Communication matrix"),
        (plot_top_lr_pairs,         "Top LR pairs"),
        (plot_esm2_umap,            "ESM-2 embedding PCA"),
        (plot_marker_dotplot,       "Marker dot plot"),
    ]

    for fn, name in steps:
        try:
            fn(pipeline, fig_dir, dpi=dpi)
        except Exception as e:
            log.warning(f"Figure '{name}' skipped: {e}")

    log.info(f"All figures written to {fig_dir}")
