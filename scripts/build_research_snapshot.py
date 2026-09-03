#!/usr/bin/env python3
"""Build and verify the two-page public research snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "public_summary" / "main_results.json"
MAIN_FIGURE = ROOT / "figures" / "fig_main_effects.png"
GAP_FIGURE = ROOT / "figures" / "fig_candidate_to_set_gap.png"
STATE_FIGURE = ROOT / "figures" / "fig_state_dependence_decision.png"
OUTPUT_MD = ROOT / "docs" / "current" / "research_snapshot_2p.md"
OUTPUT_PDF = ROOT / "docs" / "current" / "research_snapshot_2p.pdf"
MANIFEST = ROOT / "docs" / "current" / "research_snapshot_manifest.json"


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
    if data["state_dependence_v3"]["gpu_result_available"]:
        raise ValueError("snapshot template assumes no State Dependence v3 GPU result")
    return data


def signed(value: float) -> str:
    return f"{value:+.3f}"


def build_markdown(data: dict) -> str:
    completed = data["completed_study"]
    results = data["downstream_results"]
    candidate = data["candidate_utility"]
    state = data["state_dependence_v3"]
    gsm = results["gsm8k"]
    ood = results["ood_macro"]
    return f"""# LLM Post-training Data Selection: Two-page Research Snapshot

Status date: {data["study_date"]}

## Research question

{data["research_question"]}

## Controlled study

- Model: {completed["model"]}
- Training: {completed["training_method"]}
- Design: 2 methods x {completed["selection_list_count_per_method"]} frozen list
  realizations per method x {len(completed["training_seeds"])} training seeds =
  {completed["cell_count"]} cells
- Budget: {completed["examples_per_list"]} examples per list, with matched
  response-supervision exposure and source x answer-length composition
- Primary comparison: {completed["primary_estimand"]}

The RDS lists were generated under distinct query-bootstrap seeds and overlap
substantially. Their effective independence is lower than the nominal list
count; training seeds do not create new selection policies.

## Main results

| Evaluation | RDS minus Random | 95% interval | Judgment |
|---|---:|---:|---|
| GSM8K exact numeric accuracy | {signed(gsm["difference_percentage_points"])} pp | [{signed(gsm["ci95_percentage_points"][0])}, {signed(gsm["ci95_percentage_points"][1])}] pp | Insufficient evidence |
| Three-task OOD macro accuracy | {signed(ood["difference_percentage_points"])} pp | [{signed(ood["ci95_percentage_points"][0])}, {signed(ood["ci95_percentage_points"][1])}] pp | Insufficient evidence |

The controlled block found no reliable downstream accuracy advantage for the
frozen error-conditioned selection policy. The intervals still include practically relevant positive
and negative effects, so the study does not establish ineffectiveness or
equivalence.

## Candidate-to-set gap

- Tulu96 partial Spearman: {candidate["tulu96"]["partial_spearman"]:.3f};
  one-sided permutation p = {candidate["tulu96"]["one_sided_permutation_p"]:.3f};
  met the preregistered screen: partial Spearman >= 0.15, one-sided p <= 0.10,
  and positive top-minus-bottom utility.
- GSM8K-domain48 partial Spearman:
  {candidate["gsm8k_domain48"]["partial_spearman"]:.3f}; one-sided permutation
  p = {candidate["gsm8k_domain48"]["one_sided_permutation_p"]:.3f}; did not meet
  the same screen because p exceeded 0.10.
- The limited candidate-level signal did not become a stable downstream gain
  when 500 selected examples were trained together.

## Fixed-state reliability follow-up

The next study first measures fixed-state candidate-utility reliability:

- frozen panel: {state["frozen_panel_count"]} candidates;
- direct training overlap: {state["training_overlap_count"]};
- {state["frozen_panel_count"]} candidates × 3 repeated fixed-state measurements;
- status: CPU preflight complete; GPU qualification and formal measurement
  have not started.

Decision rule:

1. unreliable fixed-state measurement -> study measurement uncertainty;
2. reliable measurement with changing rankings -> study candidate revaluation;
3. reliable and stable rankings -> study redundancy, conflict, and
   complementarity within training sets.

## Four-week research module

1. qualify the fresh-process utility runner and seed semantics;
2. complete and audit the 48 × 3 fixed-state reliability panel;
3. apply the predefined stop/go decision before any cross-state evaluation;
4. deliver code, evidence manifests, a bounded result memo, and the next
   falsifiable experiment.

## Claim boundaries

Supported: no reliable RDS advantage was observed in this setting; Tulu96
met the preregistered candidate screen while GSM8K-domain48 did not.

Not supported: RDS is generally ineffective; RDS and Random are equivalent;
state dependence has been observed; a local final-adapter probe reconstructs
the historical optimizer trajectory.
"""


def wrap_lines(text: str, font: str, size: float, max_width: float) -> list[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        elif current:
            lines.append(current)
            current = word
        else:
            lines.append(word)
            current = ""
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    pdf,
    text: str,
    *,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 9,
    leading: float = 12,
    color=None,
) -> float:
    if color is not None:
        pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in wrap_lines(text, font, size, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_bullet(pdf, text: str, *, x: float, y: float, width: float) -> float:
    from reportlab.lib.colors import HexColor

    pdf.setFillColor(HexColor("#0072B2"))
    pdf.circle(x + 2.5, y + 3, 2, fill=1, stroke=0)
    return draw_wrapped(
        pdf,
        text,
        x=x + 12,
        y=y,
        width=width - 12,
        size=8.4,
        leading=10.8,
        color=HexColor("#354052"),
    ) - 3


def draw_card(pdf, x: float, y: float, width: float, height: float, color: str) -> None:
    from reportlab.lib.colors import HexColor

    pdf.setFillColor(HexColor(color))
    pdf.roundRect(x, y, width, height, 8, fill=1, stroke=0)


def draw_scaled_image(pdf, path: Path, x: float, y: float, width: float, height: float) -> None:
    from reportlab.lib.utils import ImageReader

    image = ImageReader(str(path))
    source_width, source_height = image.getSize()
    scale = min(width / source_width, height / source_height)
    draw_width = source_width * scale
    draw_height = source_height * scale
    draw_x = x + (width - draw_width) / 2
    draw_y = y + (height - draw_height) / 2
    pdf.drawImage(
        image,
        draw_x,
        draw_y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )


def draw_header(pdf, title: str, subtitle: str) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4

    page_width, page_height = A4
    pdf.setFillColor(HexColor("#0072B2"))
    pdf.rect(0, page_height - 8, page_width, 8, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, page_height - 48, title)
    pdf.setFillColor(HexColor("#5D6678"))
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(40, page_height - 66, subtitle)


def draw_footer(pdf, page_number: int) -> None:
    from reportlab.lib.colors import HexColor

    pdf.setStrokeColor(HexColor("#DCE2EA"))
    pdf.line(40, 31, 555, 31)
    pdf.setFillColor(HexColor("#697386"))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(40, 19, "Canonical source: results/public_summary/main_results.json")
    pdf.drawRightString(555, 19, f"Page {page_number} of 2")


def build_pdf(data: dict) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    completed = data["completed_study"]
    results = data["downstream_results"]
    candidate = data["candidate_utility"]
    state = data["state_dependence_v3"]
    gsm = results["gsm8k"]
    ood = results["ood_macro"]
    page_width, page_height = A4
    pdf = canvas.Canvas(
        str(OUTPUT_PDF),
        pagesize=A4,
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle("LLM Post-training Data Selection: Research Snapshot")
    pdf.setAuthor("Ruizhe Cao")
    pdf.setSubject("Controlled data selection and candidate-utility reliability")

    draw_header(
        pdf,
        "LLM Post-training Data Selection",
        "A controlled study of budget-equivalent selection and candidate-utility reliability",
    )

    draw_card(pdf, 40, 690, 515, 66, "#EEF6FB")
    pdf.setFillColor(HexColor("#0072B2"))
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(54, 737, "RESEARCH QUESTION")
    draw_wrapped(
        pdf,
        data["research_question"],
        x=54,
        y=717,
        width=485,
        font="Helvetica-Bold",
        size=10.5,
        leading=14,
        color=HexColor("#172033"),
    )

    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, 666, "Controlled design")
    card_width = 119
    card_x = [40, 172, 304, 436]
    metrics = [
        ("2", "selection methods"),
        ("4", "list realizations per method"),
        ("3", "training seeds"),
        (str(completed["cell_count"]), "audited cells"),
    ]
    for x, (value, label) in zip(card_x, metrics):
        draw_card(pdf, x, 600, card_width, 52, "#FFFFFF")
        pdf.setStrokeColor(HexColor("#DCE2EA"))
        pdf.roundRect(x, 600, card_width, 52, 7, fill=0, stroke=1)
        pdf.setFillColor(HexColor("#0072B2"))
        pdf.setFont("Helvetica-Bold", 17)
        pdf.drawString(x + 11, 625, value)
        pdf.setFillColor(HexColor("#5D6678"))
        pdf.setFont("Helvetica", 7.5)
        pdf.drawString(x + 11, 610, label)

    draw_card(pdf, 40, 524, 515, 62, "#F8FAFC")
    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(52, 568, "MATCHED FACTORS")
    y = 551
    y = draw_bullet(
        pdf,
        "500 examples, response-supervision exposure, source x answer-length composition",
        x=52,
        y=y,
        width=240,
    )
    draw_bullet(
        pdf,
        "same base model, LoRA recipe, optimizer protocol, parser, and evaluation",
        x=306,
        y=551,
        width=238,
    )

    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, 501, "Main result")
    draw_scaled_image(pdf, MAIN_FIGURE, 44, 286, 507, 205)

    draw_card(pdf, 40, 88, 515, 176, "#FFF8F2")
    pdf.setFillColor(HexColor("#A44900"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(54, 245, "BOUNDED INTERPRETATION")
    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(
        54,
        222,
        f"GSM8K: {signed(gsm['difference_percentage_points'])} pp",
    )
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        54,
        205,
        "95% interval "
        f"[{signed(gsm['ci95_percentage_points'][0])}, "
        f"{signed(gsm['ci95_percentage_points'][1])}] pp",
    )
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(
        302,
        222,
        f"OOD macro: {signed(ood['difference_percentage_points'])} pp",
    )
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        302,
        205,
        "95% interval "
        f"[{signed(ood['ci95_percentage_points'][0])}, "
        f"{signed(ood['ci95_percentage_points'][1])}] pp",
    )
    y = 179
    y = draw_bullet(
        pdf,
        (
            "No reliable downstream advantage was observed for the frozen "
            "error-conditioned selection policy."
        ),
        x=54,
        y=y,
        width=480,
    )
    y = draw_bullet(
        pdf,
        "The intervals still include relevant positive and negative effects.",
        x=54,
        y=y,
        width=480,
    )
    draw_bullet(
        pdf,
        "The study therefore establishes neither ineffectiveness nor equivalence.",
        x=54,
        y=y,
        width=480,
    )
    draw_footer(pdf, 1)
    pdf.showPage()

    draw_header(
        pdf,
        "From candidate signal to a falsifiable next study",
        "Why a local screening result did not become a stable set-level benefit",
    )
    draw_scaled_image(pdf, GAP_FIGURE, 40, 520, 515, 245)
    draw_card(pdf, 40, 446, 515, 60, "#EEF6FB")
    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(53, 487, "OBSERVED GAP")
    gap_text = (
        "Tulu96 met the preregistered screen "
        f"(partial Spearman {candidate['tulu96']['partial_spearman']:.3f} >= 0.15; "
        f"one-sided p = {candidate['tulu96']['one_sided_permutation_p']:.3f} <= 0.10; "
        "positive top-minus-bottom utility), but the 500-example downstream "
        "study did not show a stable gain."
    )
    draw_wrapped(
        pdf,
        gap_text,
        x=53,
        y=469,
        width=485,
        size=8.5,
        leading=11,
        color=HexColor("#354052"),
    )

    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, 421, "Next validation step")
    draw_scaled_image(pdf, STATE_FIGURE, 40, 188, 355, 218)
    draw_card(pdf, 410, 188, 145, 218, "#F8FAFC")
    pdf.setFillColor(HexColor("#0072B2"))
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(423, 386, "FIXED-STATE RELIABILITY")
    y = 365
    y = draw_bullet(
        pdf,
        f"{state['frozen_panel_count']} predefined candidates",
        x=423,
        y=y,
        width=120,
    )
    y = draw_bullet(
        pdf,
        f"training overlap = {state['training_overlap_count']}",
        x=423,
        y=y,
        width=120,
    )
    y = draw_bullet(
        pdf,
        f"{state['frozen_panel_count']} × 3 repeated measures",
        x=423,
        y=y,
        width=120,
    )
    y = draw_bullet(
        pdf,
        "CPU preflight complete",
        x=423,
        y=y,
        width=120,
    )
    draw_bullet(
        pdf,
        "no GPU result",
        x=423,
        y=y,
        width=120,
    )
    pdf.setFillColor(HexColor("#A44900"))
    pdf.setFont("Helvetica-Bold", 7.8)
    pdf.drawString(423, 215, "CROSS-STATE TESTING IS GATED")

    pdf.setFillColor(HexColor("#172033"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, 164, "Four-week research module")
    week_titles = ["Week 1", "Week 2", "Week 3", "Week 4"]
    week_text = [
        "Qualify fresh-process probes and seed semantics.",
        "Run and audit 48 × 3 fixed-state reliability.",
        "Apply the predefined stop/go decision before any cross-state evaluation.",
        "Deliver evidence, bounded claims, and next protocol.",
    ]
    for x, title, text in zip(card_x, week_titles, week_text):
        draw_card(pdf, x, 55, card_width, 92, "#FFFFFF")
        pdf.setStrokeColor(HexColor("#DCE2EA"))
        pdf.roundRect(x, 55, card_width, 92, 7, fill=0, stroke=1)
        pdf.setFillColor(HexColor("#0072B2"))
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(x + 10, 128, title.upper())
        draw_wrapped(
            pdf,
            text,
            x=x + 10,
            y=109,
            width=card_width - 20,
            size=7.6,
            leading=9.8,
            color=HexColor("#354052"),
        )
    draw_footer(pdf, 2)
    pdf.save()

    if not OUTPUT_PDF.is_file() or OUTPUT_PDF.stat().st_size == 0:
        raise RuntimeError("research snapshot PDF was not created")
    if page_width <= 0 or page_height <= 0:
        raise RuntimeError("invalid A4 page dimensions")


def build_manifest() -> dict:
    inputs = [SOURCE, MAIN_FIGURE, GAP_FIGURE, STATE_FIGURE]
    outputs = [OUTPUT_MD, OUTPUT_PDF]
    return {
        "schema_version": "research-snapshot-manifest-v1",
        "inputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
        "outputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }


def write_snapshot() -> None:
    data = load_data()
    OUTPUT_MD.write_text(build_markdown(data), encoding="utf-8", newline="\n")
    build_pdf(data)
    MANIFEST.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("WRITTEN research_snapshot pages=2")


def check_snapshot() -> None:
    from pypdf import PdfReader

    data = load_data()
    if OUTPUT_MD.read_text(encoding="utf-8") != build_markdown(data):
        raise RuntimeError("research snapshot Markdown is stale")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "research-snapshot-manifest-v1":
        raise ValueError("unexpected research snapshot manifest schema")
    for item in manifest["inputs"] + manifest["outputs"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"research snapshot hash mismatch: {path}")
    reader = PdfReader(str(OUTPUT_PDF))
    if len(reader.pages) != 2:
        raise RuntimeError(f"research snapshot must have two pages, found {len(reader.pages)}")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = (
        "LLM Post-training Data Selection",
        "+0.480 pp",
        "-0.094 pp",
        "48 × 3 repeated measures",
        "no GPU result",
        "Four-week research module",
    )
    missing = [text for text in required if text not in extracted]
    if missing:
        raise RuntimeError(f"research snapshot PDF is missing text: {missing}")
    print("PASS research_snapshot pages=2")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_snapshot()
    else:
        check_snapshot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
