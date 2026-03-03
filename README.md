<h1 align="center">
  CITEDiscord-Net
</h1>

<p align="center">
  <b>A Hybrid Deep-Generative Framework for RNA-Protein Discordance Discovery in CITE-seq Data</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://github.com/Runtime-Slayers/CITEDiscord-Net"><img src="https://img.shields.io/badge/GitHub-Runtime--Slayers-181717?logo=github" alt="GitHub"></a>
</p>

---

## Overview

**CITEDiscord-Net** integrates five novel or significantly extended deep learning components for comprehensive CITE-seq data analysis:

| Component | Description |
|-----------|-------------|
| **PoE-MVAE** | Product-of-Experts Multi-Modal VAE encoding RNA (2,000 HVGs) and protein (17 ADTs) into a shared 32-d latent space |
| **CMAT** | Cross-Modal Attention Transformer with bidirectional multi-head attention and learned gating |
| **MoG Prior** | Mixture-of-Gaussians prior (K=10 learnable components) replacing standard N(0,I) |
| **InfoNCE** | Symmetric NT-Xent contrastive alignment between RNA and protein projections |
| **ESM-2 CVAE** | Conditional VAE conditioned on ESM-2 protein language model embeddings (320-d) |
| **GAT Communication** | Graph Attention Network on k-NN latent cell graph for ligand-receptor interaction refinement |

## Architecture

```
Raw CITE-seq ──▶ QC & Filtering ──▶ Pearson Residuals + CLR Normalisation
       │
       ▼
   HVG Selection (2,000 genes) ──▶ VAE Augmentation (+3,000 cells)
       │
       ▼
   ESM-2 Protein Embeddings (320-d per protein)
       │
       ├──▶ PoE-MVAE + CMAT + MoG + InfoNCE (80 epochs)
       │         │
       │         ▼
       │    Joint Latent Space (32-d)
       │         │
       │         ├──▶ Leiden Clustering (6 clusters)
       │         ├──▶ UMAP Visualisation
       │         └──▶ GAT Cell Communication (k-NN graph)
       │
       └──▶ ESM-2 Conditioned CVAE (60 epochs)
                 │
                 ▼
            Protein Prediction (Spearman ρ = 0.454)
                 │
                 ▼
         Discordance Analysis (2,167 pairs)
```

## Key Results

| Metric | Value |
|--------|-------|
| Protein Prediction Spearman ρ | **0.454** |
| Discordant Gene-Protein Pairs | **2,167** |
| Mean Discordance Score | **0.293** |
| Leiden Clusters | **6** |
| Silhouette Score | **0.107** |
| MVAE Final Loss | **2.119** |
| CVAE Final Loss | **1.021** |
| Total Cells (real + augmented) | **11,000** |

## Installation

### Prerequisites

- Python ≥ 3.10
- macOS (MPS), Linux (CUDA), or CPU
- ~4 GB disk space (data + models)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Runtime-Slayers/CITEDiscord-Net.git
cd CITEDiscord-Net

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

### Alternative Setup

```bash
# Use the provided setup script
chmod +x setup.sh
./setup.sh
```

## Usage

```bash
# Full pipeline (auto-detects device: MPS/CUDA/CPU)
python main.py

# Force CPU execution
python main.py --device cpu

# Skip LaTeX generation
python main.py --skip-latex

# Skip visualisation
python main.py --skip-viz

# Custom config
python main.py --config configs/hyperparams.yaml
```

## Project Structure

```
CITEDiscord-Net/
├── main.py                          # Pipeline entry point
├── configs/
│   └── hyperparams.yaml             # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # QC, normalisation, augmentation
│   ├── data_download.py             # 10X Genomics & CellPhoneDB download
│   ├── llm_embed.py                 # ESM-2 protein embeddings
│   ├── mvae.py                      # PoE-MVAE + CMAT + MoG + InfoNCE
│   ├── cross_modal_attention.py     # CMAT, InfoNCE, MoG Prior, ProjectionHead
│   ├── cvae.py                      # ESM-2 conditioned CVAE
│   ├── clustering.py                # Leiden clustering + UMAP
│   ├── discordance.py               # Permutation-based discordance scoring
│   ├── cell_communication.py        # Classical NicheNet-style communication
│   ├── graph_communication.py       # GAT-enhanced cell communication
│   ├── visualization.py             # 13 publication-quality figures
│   ├── latex_generator.py           # Auto LaTeX manuscript generation
│   └── training_pipeline.py         # 10-stage pipeline orchestrator
├── data/
│   ├── raw/                         # Auto-downloaded raw data
│   └── processed/                   # Preprocessed .npz files
├── manuscript/
│   └── CITEDiscord_Net_Journal_Paper.tex  # Full journal paper
├── results/                         # Timestamped output directories
│   └── citediscord_YYYYMMDD_HHMMSS/
│       ├── all_metrics.json         # Quantitative results
│       ├── figures/                 # 13 PDF figures
│       ├── mvae_best.pt             # Trained MVAE weights
│       ├── cvae_best.pt             # Trained CVAE weights
│       ├── mvae_latent.npy          # Joint latent vectors
│       ├── discordance_pairs.csv    # Significant RNA-protein pairs
│       └── ...
├── requirements.txt
├── setup.sh
├── LICENSE
└── README.md
```

## Hyperparameter Configuration

All hyperparameters are configurable via `configs/hyperparams.yaml`:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **MVAE** | `latent_dim` | 32 | Joint latent dimensionality |
| | `beta` | 4.0 | β-VAE KL weight |
| | `n_epochs` | 80 | Training epochs |
| | `lr` | 0.001 | Learning rate |
| **CMAT** | `n_heads` | 4 | Multi-head attention heads |
| | `enabled` | true | Toggle CMAT on/off |
| **MoG** | `mog_components` | 10 | Gaussian mixture components |
| | `use_mog_prior` | true | Toggle MoG vs N(0,I) |
| **InfoNCE** | `lambda_nce` | 0.1 | Contrastive loss weight |
| | `infonce_temperature` | 0.07 | NT-Xent temperature τ |
| **CVAE** | `n_epochs` | 60 | Protein prediction epochs |
| | `esm_embed_dim` | 320 | ESM-2 embedding size |
| **GAT** | `k_neighbors` | 15 | k-NN graph connectivity |
| | `n_heads` | 4 | GAT attention heads |

## Dataset

The pipeline automatically downloads the **10X Genomics PBMC 10k v3** CITE-seq dataset:
- **8,000 cells** after QC filtering (from ~10,000 raw)
- **2,000 highly variable genes** (RNA)
- **17 surface protein markers** (ADT)
- **3,000 augmented cells** via VAE-based interpolation

## Output Figures

The pipeline generates 13 publication-quality PDF figures:

1. **QC Violin Plots** — Gene counts, UMI counts, mitochondrial fraction
2. **HVG Selection** — Mean-variance relationship with selected genes
3. **MVAE Training** — Loss curves (total, reconstruction, KL)
4. **Latent UMAP** — 2D projection coloured by cluster
5. **Cluster Composition** — Cell type proportions per cluster
6. **Augmentation Assessment** — Real vs augmented cell distribution
7. **Protein Prediction** — Predicted vs actual scatter per protein
8. **Spearman Heatmap** — Per-protein correlation coefficients
9. **Discordance Heatmap** — Gene × cluster discordance scores
10. **Cell Discord Distribution** — Per-cell discordance histogram
11. **Marker Expression** — Dot plot of canonical markers by cluster
12. **ESM-2 Embeddings** — t-SNE of protein language model features
13. **Communication Network** — Cell-type interaction heatmap

## Citation

If you use CITEDiscord-Net in your research, please cite:

```bibtex
@article{citediscord2026,
  title={CITEDiscord-Net: A Hybrid Deep-Generative Framework Integrating 
         Cross-Modal Attention, Mixture-of-Gaussians Priors, Contrastive 
         Alignment, and Graph Attention Networks for RNA-Protein Discordance 
         Discovery in CITE-seq Data},
  author={Reddy, Bhavanam Rajendra and Boddu, Saran and 
          Ramanathan, Muthuraman and Likith, Palakurthi K-S-S-Srihari},
  journal={Amrita School of Artificial Intelligence},
  year={2026}
}
```

## Authors

- **Bhavanam Rajendra Reddy** — Amrita School of Artificial Intelligence
- **Boddu Saran** — Amrita School of Artificial Intelligence (Corresponding Author)
- **Muthuraman Ramanathan** — Amrita School of Artificial Intelligence
- **Palakurthi K-S-S-Srihari Likith** — Amrita School of Artificial Intelligence

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was carried out as part of the undergraduate research programme at Amrita Vishwa Vidyapeetham, Coimbatore, India.
