"""Quick end-to-end smoke test for CITEDiscord-Net."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("1. Creating fallback h5 data ...")
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
from src.data_download import _create_fallback_data
h5 = _create_fallback_data(data_dir)
print(f"   h5: {h5}")

print("2. Preprocessing (small subset) ...")
from src.preprocessing import load_10x_h5, quality_control, select_hvg, pearson_residuals, clr_normalize
X_rna, X_adt, gg, pp, ct = load_10x_h5(h5)
print(f"   RNA {X_rna.shape}, ADT {X_adt.shape}")
qc = quality_control(X_rna, gg, min_genes=50, max_genes=20000)
Xr = X_rna[qc][:500]
Xa = X_adt[qc][:500]
hvg = select_hvg(Xr, n_top=200, min_cells=5)
Xr_hvg = Xr[:, hvg]
Xr_norm = pearson_residuals(Xr_hvg)
Xa_norm = clr_normalize(Xa)
print(f"   Preprocessed: RNA {Xr_norm.shape}, ADT {Xa_norm.shape}")

print("3. MVAE mini-train (3 epochs) ...")
cfg = {
    "mvae": {
        "rna_input_dim": Xr_norm.shape[1],
        "adt_input_dim": Xa_norm.shape[1],
        "latent_dim": 16,
        "rna_hidden_dims": [64, 32],
        "adt_hidden_dims": [32, 16],
        "decoder_hidden_dims": [32, 64],
        "beta": 2.0, "dropout": 0.0, "batch_norm": False,
        "lr": 1e-3, "weight_decay": 0, "batch_size": 64,
        "n_epochs": 3, "warmup_epochs": 1, "grad_clip": 1.0, "scheduler": "cosine",
    }
}
from src.mvae import MVAE, train_mvae, get_latent
mvae = MVAE(cfg)
mvae, hist = train_mvae(mvae, Xr_norm, Xa_norm, cfg, torch.device("cpu"))
mu, z = get_latent(mvae, Xr_norm, Xa_norm, torch.device("cpu"))
print(f"   Latent: {mu.shape}, final loss: {hist[-1]['loss_total']:.4f}")

print("4. Leiden clustering ...")
from src.clustering import leiden_clustering
labels = leiden_clustering(mu, n_neighbors=5, resolution=0.5)
print(f"   {len(set(labels))} clusters")

print("5. Discordance analysis ...")
from src.discordance import compute_discordance
is_real = np.ones(len(Xr_norm), dtype=bool)
discord_df, cell_discord = compute_discordance(
    Xr_norm, Xa_norm, labels, [g for g, m in zip(gg, hvg) if m], pp,
    is_real=is_real, n_permutations=10, min_cells=10,
)
print(f"   {len(discord_df)} discordant pairs")

print("6. Cell communication ...")
from src.cell_communication import load_lr_database, infer_cell_communication
lr_db = load_lr_database(None)
comm_df, comm_mat, cluster_names = infer_cell_communication(
    Xr_norm, [g for g, m in zip(gg, hvg) if m], labels, lr_db,
    is_real=is_real, n_bootstrap=5,
)
print(f"   Comm matrix {comm_mat.shape}")

print("\n=== ALL TESTS PASSED ===")
