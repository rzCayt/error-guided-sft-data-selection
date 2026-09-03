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
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.14,
            "grid.linestyle": "-",
        }
    )


def validate_text_layout(fig, stem: str) -> None:
    """Fail figure generation when visible text overlaps or leaves the canvas."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    items = []
    for artist in fig.findobj(match=Text):
        label = artist.get_text().strip()
        if not artist.get_visible() or not label:
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        if (
            bbox.x0 < canvas.x0 - 1
            or bbox.y0 < canvas.y0 - 1
            or bbox.x1 > canvas.x1 + 1
            or bbox.y1 > canvas.y1 + 1
        ):
            raise RuntimeError(f"{stem}: text leaves canvas: {label!r}")
        items.append((label, bbox))

    for index, (left_label, left_box) in enumerate(items):
        for right_label, right_box in items[index + 1 :]:
            overlap_x = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            overlap_y = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if overlap_x > 2 and overlap_y > 2:
                raise RuntimeError(
                    f"{stem}: text overlap: {left_label!r} vs {right_label!r}"
                )


def save_figure(fig, stem: str) -> None:
    validate_text_layout(fig, stem)
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

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.84, bottom=0.24)
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
    ax.set_xlim(-3.4, 2.4)
    ax.set_xticks(np.arange(-3, 3, 1))
    ax.set_xlabel("RDS − matched Random (accuracy percentage points)")
    ax.set_title("Estimated downstream effects remain uncertain", pad=10)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.12)
    fig.text(
        0.59,
        0.065,
        "Point = estimated difference; horizontal bar = 95% interval",
        ha="center",
        va="center",
        fontsize=7.7,
        color="#5B616B",
    )
    save_figure(fig, "fig_main_effects")
    plt.close(fig)


def candidate_to_set_gap(data: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    candidate = data["candidate_utility"]
    downstream = data["downstream_results"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.76, wspace=0.34)

    names = ["Tulu\n(n=96)", "GSM8K domain\n(n=48)"]
    rhos = [candidate["tulu96"]["partial_spearman"], candidate["gsm8k_domain48"]["partial_spearman"]]
    passed = [candidate["tulu96"]["original_gate_passed"], candidate["gsm8k_domain48"]["original_gate_passed"]]
    colors = ["#009E73" if value else "#B0BEC5" for value in passed]
    axes[0].bar(np.arange(2), rhos, color=colors, width=0.58)
    axes[0].axhline(0.15, color="#D55E00", linestyle="--", linewidth=1.2)
    axes[0].set_xticks(np.arange(2), names)
    axes[0].set_ylabel("Partial Spearman ρ")
    axes[0].set_title("(a) Candidate-level screening", pad=9)
    axes[0].set_ylim(0, 0.27)
    axes[0].grid(axis="x", visible=False)
    axes[0].text(
        1.46,
        0.153,
        "gate: ρ = 0.15",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#A44900",
    )
    for idx, (rho, label) in enumerate(zip(rhos, ["gate met", "gate not met"])):
        axes[0].text(
            idx,
            rho + 0.009,
            f"{rho:.3f}  ({label})",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#333940",
        )

    values = [
        downstream["gsm8k"]["difference_percentage_points"],
        downstream["ood_macro"]["difference_percentage_points"],
    ]
    labels = ["GSM8K", "OOD macro"]
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in values]
    axes[1].bar(np.arange(2), values, color=colors, width=0.58)
    axes[1].axhline(0, color="#2E3440", linewidth=1)
    axes[1].set_xticks(np.arange(2), labels)
    axes[1].set_ylabel("Accuracy difference (pp)")
    axes[1].set_title("(b) 500-example downstream training", pad=9)
    axes[1].set_ylim(-0.18, 0.62)
    axes[1].grid(axis="x", visible=False)
    for idx, value in enumerate(values):
        if value >= 0:
            y = value + 0.025
            va = "bottom"
        else:
            y = value - 0.025
            va = "top"
        axes[1].text(idx, y, f"{value:+.3f} pp", ha="center", va=va, fontsize=7.8)

    fig.suptitle(
        "Candidate-level signal did not yield a reliable set-level gain",
        x=0.54,
        y=0.96,
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, "fig_candidate_to_set_gap")
    plt.close(fig)


def state_dependence_decision(data: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    state = data["state_dependence_v3"]
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    def box(x, y, width, height, text, color, fontsize=8.2):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color + "18",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
        return patch

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color="#6B7280"))

    box(
        3.0,
        5.25,
        4.0,
        0.9,
        f"Predefined unseen panel\n{state['frozen_panel_count']} candidates · training overlap = {state['training_overlap_count']}",
        "#0072B2",
    )
    box(
        3.0,
        3.65,
        4.0,
        0.9,
        (
            f"{state['frozen_panel_count']} candidates × 3 repeated\n"
            "fixed-state measurements"
        ),
        "#E69F00",
    )
    box(
        0.15,
        0.95,
        2.95,
        1.45,
        "Unreliable\nQuantify measurement noise;\nstop cross-state claims",
        "#D55E00",
        7.7,
    )
    box(
        3.525,
        0.95,
        2.95,
        1.45,
        "Reliable; rankings change\nStudy when candidates\nneed revaluation",
        "#CC79A7",
        7.7,
    )
    box(
        6.90,
        0.95,
        2.95,
        1.45,
        "Reliable; rankings stable\nStudy redundancy, conflict,\nand complementarity",
        "#009E73",
        7.7,
    )
    arrow((5.0, 5.25), (5.0, 4.57))
    arrow((4.15, 3.65), (1.63, 2.43))
    arrow((5.0, 3.65), (5.0, 2.43))
    arrow((5.85, 3.65), (8.37, 2.43))
    fig.suptitle(
        "Fixed-state reliability follow-up: measure before cross-state comparison",
        x=0.50,
        y=0.965,
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        5.0,
        0.30,
        "Current status · CPU preflight complete · no GPU result",
        ha="center",
        color="#5B616B",
        fontsize=8.2,
    )
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
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
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
