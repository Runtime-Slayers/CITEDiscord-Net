"""
clustering.py — Leiden Clustering + UMAP on MVAE Latent Space
==============================================================

Leiden clustering (Traag et al. 2019) finds communities in a k-NN graph
built on the MVAE joint latent space.  Compared to k-means:
  • Resolution-parameterised modularity optimisation
  • No need to specify k clusters a priori
  • More biologically meaningful partitions for scRNA-seq data

Cell-type annotation uses canonical PBMC marker signatures after clustering.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# k-NN graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(
    latent: np.ndarray,
    n_neighbors: int = 15,
) -> Tuple[list, list, list]:
    """
    Build sparse k-NN graph (COO format) from MVAE latent vectors.

    Returns
    -------
    sources, targets, weights  (for igraph construction)
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    nn.fit(latent)
    distances, indices = nn.kneighbors(latent)

    sources, targets, weights = [], [], []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for d, j in zip(dists[1:], idxs[1:]):   # skip self
            sources.append(i)
            targets.append(j)
            weights.append(float(np.exp(-d)))    # Gaussian kernel weight

    return sources, targets, weights


# ---------------------------------------------------------------------------
# Leiden clustering
# ---------------------------------------------------------------------------

def leiden_clustering(
    latent: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 0.5,
) -> np.ndarray:
    """
    Leiden community detection on the k-NN graph.

    Returns
    -------
    labels : int ndarray (n_cells,)
    """
    try:
        import igraph as ig
        import leidenalg

        sources, targets, weights = build_knn_graph(latent, n_neighbors)
        n_nodes = latent.shape[0]
        G = ig.Graph(n=n_nodes, edges=list(zip(sources, targets)), directed=False)
        G.es["weight"] = weights

        partition = leidenalg.find_partition(
            G,
            leidenalg.RBConfigurationVertexPartition,
            weights=G.es["weight"],
            resolution_parameter=resolution,
            seed=42,
        )
        labels = np.array(partition.membership, dtype=np.int32)
        n_cl   = len(set(labels))
        log.info(f"Leiden: {n_cl} clusters (resolution={resolution})")
        return labels

    except ImportError:
        log.warning("leidenalg or igraph unavailable. Falling back to k-means.")
        from sklearn.cluster import KMeans
        n_clusters = max(2, int(np.sqrt(latent.shape[0] / 20)))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return km.fit_predict(latent).astype(np.int32)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(
    latent: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    n_components: int = 2,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    UMAP dimensionality reduction on MVAE latent space.

    Returns (n_cells, 2) embedding.
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
            verbose=False,
        )
        embedding = reducer.fit_transform(latent)
        log.info(f"UMAP complete: {embedding.shape}")
        return embedding.astype(np.float32)
    except ImportError:
        log.warning("umap-learn unavailable. Using PCA for 2D embedding.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(latent).astype(np.float32)


# ---------------------------------------------------------------------------
# Cell-type annotation from marker genes
# ---------------------------------------------------------------------------

# Canonical PBMC marker signatures (Zheng et al. 2017 + 10X references)
PBMC_MARKERS: dict[str, list[str]] = {
    "T_CD4_Naive":    ["CD4", "CCR7", "SELL", "TCF7", "LEF1"],
    "T_CD4_Memory":   ["CD4", "IL7R", "S100A4", "AQP3"],
    "T_CD8_Effector": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1"],
    "T_CD8_Naive":    ["CD8A", "CCR7", "TCF7", "SELL"],
    "NK":             ["NCAM1", "NKG7", "KLRB1", "GNLY", "FGFBP2"],
    "B_cell":         ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1"],
    "Monocyte_CD14":  ["CD14", "LYZ", "CST3", "FCN1", "S100A8"],
    "Monocyte_CD16":  ["FCGR3A", "MS4A7", "VMO1", "CXCL10"],
    "DC":             ["FCER1A", "CLEC9A", "IL3RA", "LILRA4"],
    "Platelet":       ["PPBP", "PF4", "GP1BA", "TREML1"],
    "HSPC":           ["SPINK2", "PRSS57", "EGFL7"],
}


def annotate_cell_types(
    X_rna: np.ndarray,           # (n_cells, n_hvg) Pearson residuals
    gene_names: list[str],
    cluster_labels: np.ndarray,
    markers: Optional[dict] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Score each cluster against canonical marker gene sets.

    Uses a simple mean expression ranking (no model needed).
    Returns per-cell labels and per-cluster scoring DataFrame.
    """
    if markers is None:
        markers = PBMC_MARKERS

    gene_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    clusters = sorted(np.unique(cluster_labels))
    n_cl     = len(clusters)
    ct_names = list(markers.keys())

    score_mx = np.zeros((n_cl, len(ct_names)), dtype=np.float32)

    for ci, cl in enumerate(clusters):
        cl_mask  = cluster_labels == cl
        X_cl     = X_rna[cl_mask]
        mean_exp = X_cl.mean(axis=0)  # (n_hvg,)

        for ti, (ct, mgs) in enumerate(markers.items()):
            present = [gene_idx[g.upper()] for g in mgs if g.upper() in gene_idx]
            if present:
                score_mx[ci, ti] = mean_exp[present].mean()

    # Assign cell type to each cluster (argmax)
    best_ct_idx = score_mx.argmax(axis=1)
    cl_to_ct = {cl: ct_names[best_ct_idx[ci]] for ci, cl in enumerate(clusters)}

    cell_type_labels = np.array([cl_to_ct[cl] for cl in cluster_labels])

    score_df = pd.DataFrame(
        score_mx,
        index=[f"Cluster_{cl}" for cl in clusters],
        columns=ct_names,
    )
    score_df["predicted_cell_type"] = [ct_names[i] for i in best_ct_idx]
    log.info(f"Cell-type annotation: {len(set(cl_to_ct.values()))} distinct types")

    return cell_type_labels, score_df
