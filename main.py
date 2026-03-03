#!/usr/bin/env python3
"""
main.py — CITEDiscord-Net Pipeline Orchestrator
=================================================

Single-Cell Multi-Modal CITE-seq Integration:
Discordance Analysis & Cell Communication Inference
with Product-of-Experts MVAE + ESM-2 LLM + Conditional VAE

Architecture summary:
  ─ Real data: 10X Genomics PBMC 10k CITE-seq (publicly available)
  ─ Augmented data: VAE-based cell interpolation (~3k extra cells)
  ─ Hybrid model:
        Product-of-Experts MVAE (RNA encoder + Protein encoder + PoE fusion)
      + ESM-2 protein language model embeddings (HuggingFace)
      + Conditional CVAE (RNA latent → protein prediction)
      + Permutation discordance tester
      + From-scratch NicheNet cell communication engine
  ─ Output: 14 figures, JSON metrics, LaTeX paper, CSV artefacts

Usage:
    python main.py                         # full pipeline, auto device
    python main.py --device cpu            # force CPU
    python main.py --skip-latex            # skip LaTeX generation
    python main.py --skip-viz              # skip figure generation
    python main.py --config configs/hyperparams.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("PYTHONPATH", str(PROJECT_DIR))

log = logging.getLogger(__name__)


def banner(text: str, char: str = "█", width: int = 72) -> None:
    print(f"\n{char * width}\n  {text}\n{char * width}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CITEDiscord-Net: CITE-seq Discordance Analysis & Cell Communication Inference\n"
            "Hybrid model: PoE-MVAE + ESM-2 LLM + CVAE"
        )
    )
    parser.add_argument("--project-dir", default=str(PROJECT_DIR))
    parser.add_argument("--config",      default=None,
                        help="Path to hyperparams YAML")
    parser.add_argument("--device",      default=None,
                        help="Device: auto | cpu | cuda | mps")
    parser.add_argument("--skip-latex",  action="store_true",
                        help="Skip LaTeX manuscript generation")
    parser.add_argument("--skip-viz",    action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    t_global = time.time()
    banner("CITEDiscord-Net Pipeline — CITE-seq Discordance & Cell Communication")

    project_dir = Path(args.project_dir)

    # ── Run pipeline ────────────────────────────────────────────────
    banner("STEP 1/3 — Training Pipeline", "═")
    from src.training_pipeline import CITEDiscordPipeline

    pipeline = CITEDiscordPipeline(
        project_dir=project_dir,
        config_path=args.config,
        device=args.device,
    )
    try:
        metrics = pipeline.run()
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

    results_dir = pipeline.results_dir

    # ── Visualisation ────────────────────────────────────────────────
    if not args.skip_viz:
        banner("STEP 2/3 — Visualisation (14 figures)", "═")
        try:
            from src.visualization import generate_all_figures
            dpi = pipeline.cfg.get("output", {}).get("figure_dpi", 150)
            generate_all_figures(pipeline, results_dir, dpi=dpi)
        except Exception as e:
            log.warning(f"Visualisation failed (non-fatal): {e}")
            log.debug(traceback.format_exc())

    # ── LaTeX manuscript ─────────────────────────────────────────────
    if not args.skip_latex:
        banner("STEP 3/3 — LaTeX Manuscript Generation", "═")
        try:
            from src.latex_generator import generate_latex
            tex_path = generate_latex(metrics, results_dir, project_dir)
            log.info(f"LaTeX written: {tex_path}")

            # Attempt pdflatex compilation (graceful failure if unavailable)
            import subprocess
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", str(tex_path.parent),
                 str(tex_path)],
                capture_output=True, text=True, timeout=120,
            )
            if res.returncode == 0:
                pdf = tex_path.with_suffix(".pdf")
                log.info(f"PDF compiled: {pdf}")
            else:
                log.warning("pdflatex compilation failed (not installed?). "
                            "LaTeX source is still available.")
        except FileNotFoundError:
            log.warning("pdflatex not found — install TeX Live to compile PDF.")
        except Exception as e:
            log.warning(f"LaTeX step failed (non-fatal): {e}")

    # ── Final summary ────────────────────────────────────────────────
    elapsed = time.time() - t_global
    banner("DONE", "─")
    print(f"  Results directory : {results_dir}")
    print(f"  Total time        : {elapsed:.0f}s")
    print()
    print("  Key outputs:")
    print(f"    - all_metrics.json        — quantitative results")
    print(f"    - results/figures/        — 14 publication-quality PDFs")
    print(f"    - manuscript/*.tex        — auto-generated LaTeX paper")
    print(f"    - mvae_best.pt            — trained MVAE weights")
    print(f"    - cvae_best.pt            — trained CVAE weights")
    print(f"    - mvae_latent.npy         — joint latent vectors (n_cells × 32)")
    print(f"    - discordance_pairs.csv   — significant RNA-protein pairs")
    print(f"    - cell_communication.csv  — LR interaction network")
    print()


if __name__ == "__main__":
    main()
