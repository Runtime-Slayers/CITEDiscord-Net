"""
training_pipeline.py — CITEDiscord-Net Full Pipeline Orchestration
====================================================================

Orchestrates all stages in order:
  1.  Data download (real PBMC 10k CITE-seq + CellPhoneDB LR db)
  2.  Preprocessing (QC, CLR, Pearson residuals, HVG, augmentation)
  3.  ESM-2 protein LLM embeddings
  4.  MVAE training (Product-of-Experts multi-modal latent learning)
  5.  Leiden clustering + UMAP on joint latent space
  6.  Cell-type annotation via marker gene scoring
  7.  CVAE training (protein prediction from RNA latent + ESM-2)
  8.  RNA-protein discordance analysis (permutation-based)
  9.  Cell communication inference (from-scratch NicheNet equivalent)
  10. Metrics computation + artefact saving
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

log = logging.getLogger(__name__)


class CITEDiscordPipeline:
    """End-to-end CITEDiscord-Net training and analysis pipeline."""

    def __init__(
        self,
        project_dir: Path,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.project_dir = Path(project_dir)
        self.cfg = self._load_config(config_path)
        self.device  = self._resolve_device(device)
        self.rng     = np.random.default_rng(self.cfg.get("seed", 42))

        # Placeholders — populated during run()
        self.data:          dict = {}
        self.cluster_labels: np.ndarray = None
        self.umap_emb:       np.ndarray = None
        self.cell_types:     np.ndarray = None
        self.mvae_latent:    np.ndarray = None
        self.cell_discord:   np.ndarray = None
        self.discord_df:     pd.DataFrame = None
        self.comm_df:        pd.DataFrame = None
        self.comm_matrix:    np.ndarray   = None
        self.cluster_names:  list[str]    = []
        self.esm_embeddings: np.ndarray   = None
        self.results_dir:    Path = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: Optional[str]) -> dict:
        default = self.project_dir / "configs" / "hyperparams.yaml"
        path    = Path(config_path) if config_path else default
        if path.exists():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            log.info(f"Config: {path}")
            return cfg
        log.warning(f"Config not found at {path}. Using built-in defaults.")
        return {}

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device and device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Results directory
    # ------------------------------------------------------------------

    def _create_results_dir(self) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"citediscord_{ts}"
        rdir = self.project_dir / self.cfg.get("output", {}).get("results_dir", "results") / name
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "figures").mkdir(exist_ok=True)
        (rdir / "manuscript").mkdir(exist_ok=True)
        return rdir

    # ------------------------------------------------------------------
    # Stage 1: Download
    # ------------------------------------------------------------------

    def _stage_download(self):
        log.info("=== Stage 1: Data Download ===")
        from src.data_download import download_pbmc_10k, download_cellphonedb

        raw_dir = self.project_dir / self.cfg.get("data", {}).get("raw_dir", "data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        self.h5_path  = download_pbmc_10k(raw_dir)
        self.lr_csv   = download_cellphonedb(raw_dir)

    # ------------------------------------------------------------------
    # Stage 2: Preprocessing
    # ------------------------------------------------------------------

    def _stage_preprocess(self):
        log.info("=== Stage 2: Preprocessing ===")
        from src.preprocessing import run_preprocessing

        proc_dir = self.project_dir / self.cfg.get("data", {}).get("processed_dir", "data/processed")
        cache   = proc_dir / "preprocessed.npz"

        if cache.exists():
            log.info(f"Loading preprocessed cache from {cache}")
            d = np.load(cache, allow_pickle=True)
            self.data = {k: d[k] for k in d.files}
            # Restore lists
            self.data["gene_names_hvg"] = list(self.data["gene_names_hvg"])
            self.data["protein_names"]  = list(self.data["protein_names"])
            self.data["cell_types_raw"] = (
                self.data["cell_types_raw"] if self.data["cell_types_raw"].ndim > 0 else None
            )
            # Derive n_real / n_augmented from is_real if not stored (backwards-compat)
            if "n_real" not in self.data and "is_real" in self.data:
                self.data["n_real"] = int(self.data["is_real"].sum())
            if "n_augmented" not in self.data and "is_real" in self.data:
                self.data["n_augmented"] = int((~self.data["is_real"]).sum())
        else:
            self.data = run_preprocessing(self.h5_path, self.cfg, rng=self.rng)
            proc_dir.mkdir(parents=True, exist_ok=True)
            saveable = {k: v for k, v in self.data.items()
                        if v is not None and isinstance(v, np.ndarray)}
            np.savez_compressed(cache, **saveable,
                                gene_names_hvg=np.array(self.data["gene_names_hvg"]),
                                protein_names =np.array(self.data["protein_names"]),
                                n_real        =np.int64(self.data["n_real"]),
                                n_augmented   =np.int64(self.data["n_augmented"]))
            log.info(f"Saved preprocessed cache → {cache}")

    # ------------------------------------------------------------------
    # Stage 3: ESM-2 embeddings
    # ------------------------------------------------------------------

    def _stage_esm2(self):
        log.info("=== Stage 3: ESM-2 Protein LLM Embeddings ===")
        from src.llm_embed import compute_protein_embeddings

        device_str = str(self.device)
        self.esm_embeddings = compute_protein_embeddings(
            self.data["protein_names"], self.cfg, device_str=device_str
        )
        log.info(f"ESM-2 embeddings: {self.esm_embeddings.shape}")

    # ------------------------------------------------------------------
    # Stage 4: MVAE training
    # ------------------------------------------------------------------

    def _stage_mvae(self):
        log.info("=== Stage 4: Multi-Modal VAE (Product-of-Experts) Training ===")
        from src.mvae import MVAE, train_mvae, get_latent

        mv = self.cfg.get("mvae", {})
        # Auto-detect input dims
        mv["rna_input_dim"] = self.data["X_rna_norm"].shape[1]
        mv["adt_input_dim"] = self.data["X_adt_norm"].shape[1]
        self.cfg["mvae"]    = mv

        mvae_path = self.results_dir / "mvae_best.pt"

        mvae = MVAE(self.cfg)
        mvae, self.mvae_history = train_mvae(
            mvae, self.data["X_rna_norm"], self.data["X_adt_norm"],
            self.cfg, self.device
        )

        torch.save(mvae.state_dict(), mvae_path)
        log.info(f"MVAE saved → {mvae_path}")

        # Extract latent representations
        self.mvae_mu, self.mvae_latent = get_latent(
            mvae, self.data["X_rna_norm"], self.data["X_adt_norm"], self.device
        )
        np.save(self.results_dir / "mvae_latent.npy", self.mvae_latent)
        np.save(self.results_dir / "mvae_mu.npy", self.mvae_mu)
        self.mvae_model = mvae

    # ------------------------------------------------------------------
    # Stage 5: Clustering + UMAP
    # ------------------------------------------------------------------

    def _stage_cluster(self):
        log.info("=== Stage 5: Leiden Clustering + UMAP ===")
        from src.clustering import leiden_clustering, run_umap, annotate_cell_types

        cl_cfg   = self.cfg.get("clustering", {})
        self.cluster_labels = leiden_clustering(
            self.mvae_mu,    # Use deterministic mu for clustering
            n_neighbors=cl_cfg.get("n_neighbors", 15),
            resolution=cl_cfg.get("leiden_resolution", 0.5),
        )
        self.umap_emb = run_umap(
            self.mvae_mu,
            n_neighbors=cl_cfg.get("n_neighbors", 15),
            min_dist=cl_cfg.get("umap_min_dist", 0.3),
        )

        self.cell_types, self.ct_score_df = annotate_cell_types(
            self.data["X_rna_norm"][:self.data["n_real"]],   # real only
            self.data["gene_names_hvg"],
            self.cluster_labels[:self.data["n_real"]],
        )
        # Fill augmented cells with nearest cluster's label
        all_ct = np.full(len(self.cluster_labels), "Unknown", dtype=object)
        all_ct[:self.data["n_real"]] = self.cell_types
        self.cell_types_all = all_ct

        np.save(self.results_dir / "cluster_labels.npy",  self.cluster_labels)
        np.save(self.results_dir / "umap_embedding.npy",  self.umap_emb)
        np.save(self.results_dir / "cell_types.npy",      self.cell_types_all)
        self.ct_score_df.to_csv(self.results_dir / "cell_type_scores.csv")

    # ------------------------------------------------------------------
    # Stage 6: CVAE training
    # ------------------------------------------------------------------

    def _stage_cvae(self):
        log.info("=== Stage 6: Conditional VAE — Protein Prediction from RNA ===")
        from src.cvae import ProteinCVAE, train_cvae, predict_protein

        n_proteins  = self.data["X_adt_norm"].shape[1]
        esm_dim     = (self.esm_embeddings.shape[1]
                       if self.esm_embeddings is not None else 0)
        self.cfg["cvae"]["condition_dim"] = self.cfg["mvae"]["latent_dim"]
        self.cfg["cvae"]["esm_embed_dim"] = esm_dim

        cvae = ProteinCVAE(self.cfg, n_proteins=n_proteins, esm_dim=esm_dim)
        cvae, self.cvae_history = train_cvae(
            cvae,
            self.mvae_mu,                    # RNA latent = MVAE joint μ
            self.data["X_adt_norm"],
            self.esm_embeddings,
            self.cfg,
            self.device,
            is_real=self.data["is_real"],
        )
        torch.save(cvae.state_dict(), self.results_dir / "cvae_best.pt")

        # Predict protein for all cells
        self.protein_pred = predict_protein(
            cvae, self.mvae_mu, self.esm_embeddings, self.device
        )
        np.save(self.results_dir / "protein_predicted.npy", self.protein_pred)

    # ------------------------------------------------------------------
    # Stage 7: Discordance analysis
    # ------------------------------------------------------------------

    def _stage_discordance(self):
        log.info("=== Stage 7: RNA-Protein Discordance Analysis ===")
        from src.discordance import compute_discordance, summarise_discordance

        dc_cfg = self.cfg.get("discordance", {})
        self.discord_df, self.cell_discord = compute_discordance(
            self.data["X_rna_norm"],
            self.data["X_adt_norm"],
            self.cluster_labels,
            self.data["gene_names_hvg"],
            self.data["protein_names"],
            is_real=self.data["is_real"],
            n_permutations=dc_cfg.get("n_permutations", 500),
            alpha=dc_cfg.get("alpha", 0.05),
            min_cells=dc_cfg.get("min_corr_cells", 20),
        )

        summary_df = summarise_discordance(
            self.cell_discord, self.cluster_labels,
            self.data["protein_names"], self.discord_df,
        )
        self.discord_df.to_csv(self.results_dir / "discordance_pairs.csv", index=False)
        summary_df.to_csv(self.results_dir / "discordance_summary.csv", index=False)
        np.save(self.results_dir / "cell_discord_scores.npy", self.cell_discord)

    # ------------------------------------------------------------------
    # Stage 8: Cell communication (GAT-enhanced hybrid)
    # ------------------------------------------------------------------

    def _stage_cell_comm(self):
        log.info("=== Stage 8: GAT-Enhanced Cell Communication Inference ===")
        from src.cell_communication import load_lr_database, infer_cell_communication
        from src.graph_communication import infer_communication_gat

        lr_db  = load_lr_database(getattr(self, "lr_csv", None))
        cc_cfg = self.cfg.get("cell_communication", {})
        gat_cfg = cc_cfg.get("gat", {})

        use_gat = gat_cfg.get("enabled", True) and (self.mvae_latent is not None)

        if use_gat:
            log.info("Using GAT-enhanced cell communication (novel hybrid mode)")
            try:
                self.comm_df, self.comm_matrix, self.cluster_names = infer_communication_gat(
                    X_rna           = self.data["X_rna_norm"],
                    gene_names      = self.data["gene_names_hvg"],
                    cluster_labels  = self.cluster_labels,
                    mvae_latent     = self.mvae_latent,
                    lr_db           = lr_db,
                    is_real         = self.data["is_real"],
                    k_neighbors     = gat_cfg.get("k_neighbors", 15),
                    n_bootstrap     = cc_cfg.get("n_bootstrap", 200),
                    min_expr_frac   = cc_cfg.get("min_expr_fraction", 0.10),
                    expr_threshold_pct = cc_cfg.get("expr_threshold_percentile", 10.0),
                    gat_hid_dim     = gat_cfg.get("hid_dim", 16),
                    gat_out_dim     = gat_cfg.get("out_dim", 32),
                    gat_n_heads     = gat_cfg.get("n_heads", 4),
                    device          = str(self.device),
                )
            except Exception as e:
                log.warning(f"GAT communication failed ({e}); falling back to classical.")
                use_gat = False

        if not use_gat:
            log.info("Using classical NicheNet-style cell communication (fallback)")
            self.comm_df, self.comm_matrix, self.cluster_names = infer_cell_communication(
                self.data["X_rna_norm"],
                self.data["gene_names_hvg"],
                self.cluster_labels,
                lr_db,
                is_real=self.data["is_real"],
                n_bootstrap=cc_cfg.get("n_bootstrap", 200),
                min_expr_fraction=cc_cfg.get("min_expr_fraction", 0.10),
                expr_threshold_pct=cc_cfg.get("expr_threshold_percentile", 10.0),
            )

        if self.comm_df is not None and len(self.comm_df) > 0:
            self.comm_df.to_csv(self.results_dir / "cell_communication.csv", index=False)
        np.save(self.results_dir / "comm_matrix.npy", self.comm_matrix)

    # ------------------------------------------------------------------
    # Stage 9: Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict:
        from sklearn.metrics import silhouette_score

        metrics: dict = {}

        # MVAE convergence
        if hasattr(self, "mvae_history") and self.mvae_history:
            metrics["mvae_final_loss"]   = self.mvae_history[-1]["loss_total"]
            metrics["mvae_final_kl"]     = self.mvae_history[-1]["kl"]
            metrics["mvae_final_recon"]  = (
                self.mvae_history[-1]["recon_rna"]
                + self.mvae_history[-1]["recon_adt"]
            )

        # CVAE convergence
        if hasattr(self, "cvae_history") and self.cvae_history:
            metrics["cvae_final_loss"] = self.cvae_history[-1]["cvae_loss"]

        # Clustering quality
        try:
            real_mask = self.data["is_real"]
            sil = silhouette_score(
                self.mvae_mu[real_mask],
                self.cluster_labels[real_mask],
                sample_size=min(2000, real_mask.sum()),
                random_state=42,
            )
            metrics["silhouette_score"] = float(sil)
        except Exception:
            pass

        # Protein prediction quality
        try:
            from scipy.stats import spearmanr
            real_mask = self.data["is_real"]
            rho, _  = spearmanr(
                self.data["X_adt_norm"][real_mask].flatten(),
                self.protein_pred[real_mask].flatten(),
            )
            metrics["protein_pred_spearman_r"] = float(rho)
        except Exception:
            pass

        # Discordance stats
        if self.discord_df is not None and len(self.discord_df) > 0:
            metrics["n_discordant_pairs"] = int(
                (self.discord_df.get("fdr_padj", self.discord_df["z_score"]).pipe(
                    lambda s: (s < 0.05).sum() if "fdr_padj" in s.index or "fdr_padj" == s.name else (np.abs(s) > 2.576).sum()
                ))
            )
            metrics["mean_discord_score"] = float(self.cell_discord.mean())

        # Cell comm
        if self.comm_df is not None and len(self.comm_df) > 0:
            metrics["n_significant_interactions"] = int((self.comm_df["fdr"] < 0.1).sum())

        metrics["n_real_cells"]      = int(self.data["n_real"])
        metrics["n_augmented_cells"] = int(self.data["n_augmented"])
        metrics["n_clusters"]        = int(len(np.unique(self.cluster_labels)))
        metrics["n_proteins"]        = int(self.data["X_adt_norm"].shape[1])
        metrics["n_hvg"]             = int(self.data["X_rna_norm"].shape[1])
        metrics["esm2_embed_dim"]    = int(
            self.esm_embeddings.shape[1] if self.esm_embeddings is not None else 0
        )

        return metrics

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self) -> dict:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

        log.info(f"Device: {self.device}")
        self.results_dir = self._create_results_dir()
        # Save config
        with open(self.results_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

        t0 = time.time()

        self._stage_download()
        self._stage_preprocess()
        self._stage_esm2()
        self._stage_mvae()
        self._stage_cluster()
        self._stage_cvae()
        self._stage_discordance()
        self._stage_cell_comm()

        metrics = self._compute_metrics()
        with open(self.results_dir / "all_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        elapsed = time.time() - t0
        metrics["elapsed_seconds"] = round(elapsed, 1)

        # Summary text
        summary = self._format_summary(metrics)
        (self.results_dir / "summary.txt").write_text(summary)
        print(summary)

        return metrics

    def _format_summary(self, m: dict) -> str:
        lines = [
            "=" * 70,
            "  CITEDiscord-Net — Run Summary",
            "=" * 70,
            f"  Real cells:            {m.get('n_real_cells', '?')}",
            f"  Augmented cells:       {m.get('n_augmented_cells', '?')}",
            f"  HVGs:                  {m.get('n_hvg', '?')}",
            f"  Proteins (ADT):        {m.get('n_proteins', '?')}",
            f"  ESM-2 embed dim:       {m.get('esm2_embed_dim', '?')}",
            f"  Leiden clusters:       {m.get('n_clusters', '?')}",
            "",
            f"  MVAE final loss:       {m.get('mvae_final_loss', 'N/A'):.4f}",
            f"  MVAE KL:               {m.get('mvae_final_kl', 'N/A'):.4f}",
            f"  CVAE final loss:       {m.get('cvae_final_loss', 'N/A'):.4f}",
            f"  Silhouette score:      {m.get('silhouette_score', 'N/A')}",
            f"  Protein pred Spearman: {m.get('protein_pred_spearman_r', 'N/A')}",
            "",
            f"  Discordant LR pairs:   {m.get('n_discordant_pairs', 'N/A')}",
            f"  Mean discord score:    {m.get('mean_discord_score', 'N/A'):.4f}",
            f"  Sig. cell comm. pairs: {m.get('n_significant_interactions', 'N/A')}",
            "",
            f"  Total time:            {m.get('elapsed_seconds', 0):.0f}s",
            "=" * 70,
        ]
        return "\n".join(lines)
