#!/usr/bin/env python3
"""Generate the three public research figures from the canonical result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "public_summary" / "main_results.json"
FIGURE_DIR = ROOT / "figures"
MANIFEST = FIGURE_DIR / "manifest.json"
FIGURE_STEMS = (
    "fig_main_effects",
    "fig_candidate_to_set_gap",
    "fig_state_dependence_decision",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data() -> dict:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if data.get("schema_version") != "public-research-summary-v1":
        raise ValueError("unexpected public summary schema")
    return data


def configure_plotting() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
        }
    )


def save_figure(fig, stem: str) -> None:
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(
        pdf,
        facecolor="white",
        metadata={"Creator": "generate_public_figures.py", "CreationDate": None},
    )


def main_effects(data: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ["gsm8k", "svamp", "asdiv_numeric", "multiarith", "ood_macro"]
    labels = ["GSM8K", "SVAMP", "ASDiv numeric", "MultiArith", "OOD macro"]
    values = np.array(
        [data["downstream_results"][key]["difference_percentage_points"] for key in keys]
    )
    intervals = np.array(
        [data["downstream_results"][key]["ci95_percentage_points"] for key in keys]
    )
    lower = values - intervals[:, 0]
    upper = intervals[:, 1] - values
    y = np.arange(len(keys))[::-1]

    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    ax.axvline(0, color="#2E3440", linewidth=1.1, zorder=1)
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#0072B2",
        ecolor="#56B4E9",
        capsize=4,
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("RDS − matched Random (accuracy percentage points)")
    ax.set_title("Estimated downstream differences remain uncertain")
    ax.text(
        0.99,
        0.02,
        "Dots: point estimates; bars: 95% intervals",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    save_figure(fig, "fig_main_effects")
    plt.close(fig)


def candidate_to_set_gap(data: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    candidate = data["candidate_utility"]
    downstream = data["downstream_results"]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9))

    names = ["Tulu96", "GSM8K-domain48"]
    rhos = [candidate["tulu96"]["partial_spearman"], candidate["gsm8k_domain48"]["partial_spearman"]]
    passed = [candidate["tulu96"]["original_gate_passed"], candidate["gsm8k_domain48"]["original_gate_passed"]]
    colors = ["#009E73" if value else "#B0BEC5" for value in passed]
    axes[0].bar(np.arange(2), rhos, color=colors, width=0.58)
    axes[0].axhline(0.15, color="#D55E00", linestyle="--", linewidth=1, label="frozen ρ gate")
    axes[0].set_xticks(np.arange(2), names, rotation=12)
    axes[0].set_ylabel("Partial Spearman ρ")
    axes[0].set_title("Candidate-level screening")
    axes[0].legend(fontsize=7.5, frameon=False)
    for idx, (rho, label) in enumerate(zip(rhos, ["passed", "did not pass"])):
        axes[0].text(idx, rho + 0.012, f"{rho:.3f}\n{label}", ha="center", fontsize=7.5)

    values = [
        downstream["gsm8k"]["difference_percentage_points"],
        downstream["ood_macro"]["difference_percentage_points"],
    ]
    labels = ["GSM8K", "OOD macro"]
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in values]
    axes[1].bar(np.arange(2), values, color=colors, width=0.58)
    axes[1].axhline(0, color="#2E3440", linewidth=1)
    axes[1].set_xticks(np.arange(2), labels)
    axes[1].set_ylabel("RDS − Random (percentage points)")
    axes[1].set_title("500-example downstream training")
    for idx, value in enumerate(values):
        offset = 0.06 if value >= 0 else -0.11
        axes[1].text(idx, value + offset, f"{value:+.3f}", ha="center", fontsize=8)

    fig.suptitle("A local screening signal did not become a reliable set-level gain", y=1.03, fontweight="bold")
    save_figure(fig, "fig_candidate_to_set_gap")
    plt.close(fig)


def state_dependence_decision(data: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    state = data["state_dependence_v3"]
    fig, ax = plt.subplots(figsize=(6.75, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, width, height, text, color, fontsize=8.5):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1,
            edgecolor=color,
            facecolor=color + "18",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
        return patch

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color="#6B7280"))

    box(3.0, 5.4, 4.0, 1.0, f"Frozen unseen panel\n{state['frozen_panel_count']} candidates; overlap = {state['training_overlap_count']}", "#0072B2")
    box(3.0, 3.7, 4.0, 1.0, f"U0a: fixed-state reliability\n{state['u0a_planned_measurements']} planned measurements", "#E69F00")
    box(0.2, 1.1, 2.8, 1.35, "Unreliable\nStudy measurement uncertainty\nand stop cross-state claims", "#D55E00", 8)
    box(3.6, 1.1, 2.8, 1.35, "Reliable, rankings change\nStudy when examples\nneed revaluation", "#CC79A7", 8)
    box(7.0, 1.1, 2.8, 1.35, "Reliable, rankings stable\nStudy redundancy, conflict,\nand complementarity", "#009E73", 8)
    arrow((5.0, 5.4), (5.0, 4.72))
    arrow((4.1, 3.7), (1.6, 2.48))
    arrow((5.0, 3.7), (5.0, 2.48))
    arrow((5.9, 3.7), (8.4, 2.48))
    ax.text(5.0, 6.75, "State Dependence v3: measure reliability before mechanism", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(5.0, 0.35, "Current status: CPU preflight complete; no GPU result", ha="center", color="#555555", fontsize=8.5)
    save_figure(fig, "fig_state_dependence_decision")
    plt.close(fig)


def write_figures() -> None:
    configure_plotting()
    data = load_data()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    main_effects(data)
    candidate_to_set_gap(data)
    state_dependence_decision(data)

    files = []
    for stem in FIGURE_STEMS:
        for suffix in (".pdf", ".png"):
            path = FIGURE_DIR / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"figure was not created: {path}")
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "schema_version": "public-figure-manifest-v1",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WRITTEN figures={len(files)}")


def check_figures() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "public-figure-manifest-v1":
        raise ValueError("unexpected figure manifest schema")
    if payload.get("source_sha256") != sha256(SOURCE):
        raise RuntimeError("figure source hash does not match canonical results")
    for item in payload.get("files", []):
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"figure hash mismatch: {path}")
    print(f"PASS figures={len(payload['files'])}")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_figures()
    else:
        check_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
