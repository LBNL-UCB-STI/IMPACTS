"""Render the CES population-weighted concentration table in manuscript style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GROUP_LABELS = {
    "0.00-0.25": "0–25%",
    "0.25-0.50": "25–50%",
    "0.50-0.75": "50–75%",
    "0.75-1.00": "75–100%",
    "All Bay Area valid-CES": "All Bay Area",
}


def percent_change(reference: float, enhanced: float) -> float:
    return (reference - enhanced) / enhanced * 100.0


def prepare(summary: pd.DataFrame) -> pd.DataFrame:
    lookup = summary.set_index(["CES_Group", "Pollutant", "Method"])
    rows = []
    for source_group, display_group in GROUP_LABELS.items():
        pm_isrm = lookup.loc[(source_group, "PM2.5", "InMAP-only")]
        pm_final = lookup.loc[(source_group, "PM2.5", "Hybrid/AERMOD-enhanced")]
        bc_isrm = lookup.loc[(source_group, "BC", "InMAP-only")]
        bc_final = lookup.loc[(source_group, "BC", "Indicator-corrected final")]
        population = int(round(float(pm_final["ModelPopulation"])))
        pm_i = float(pm_isrm["PopulationWeightedConcentration_ug_m3"])
        pm_f = float(pm_final["PopulationWeightedConcentration_ug_m3"])
        bc_i = float(bc_isrm["PopulationWeightedConcentration_ug_m3"])
        bc_f = float(bc_final["PopulationWeightedConcentration_ug_m3"])
        rows.append(
            {
                "CES group": display_group,
                "Population": population,
                "ISRM PM2.5 PWM": pm_i,
                "ISRM BC PWM": bc_i,
                "Enhanced PM2.5 PWM": pm_f,
                "Enhanced BC PWM": bc_f,
                "PM2.5 change (%)": percent_change(pm_i, pm_f),
                "BC change (%)": percent_change(bc_i, bc_f),
            }
        )
    return pd.DataFrame(rows)


def render(data: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    fig, ax = plt.subplots(figsize=(12.8, 6.7), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        r"TABLE 1. Cal–ISRM and AERMOD-enhanced PWC of total PM$_{2.5}$/BC (μg/m³) by CES group",
        ha="center",
        va="center",
        fontsize=16,
    )
    ax.plot([0.02, 0.98], [0.945, 0.945], color="black", lw=3.0)
    ax.text(0.5, 0.895, r"PWM total PM$_{2.5}$/BC by CES", ha="center", va="center", fontsize=16)

    x = [0.03, 0.245, 0.425, 0.68, 0.91]
    headers = ["CES group", "Population", "ISRM PWM", "AERMOD-enhanced PWM", "Conc. change"]
    alignments = ["left", "center", "center", "center", "center"]
    for xpos, header, alignment in zip(x, headers, alignments):
        ax.text(xpos, 0.80, header, ha=alignment, va="center", fontsize=15, fontweight="bold")

    row_y = [0.66, 0.52, 0.38, 0.24, 0.10]
    for ypos, (_, row) in zip(row_y, data.iterrows()):
        ax.text(x[0], ypos, str(row["CES group"]), ha="left", va="center", fontsize=14.5)
        ax.text(x[1], ypos, f'{int(row["Population"]):,}', ha="center", va="center", fontsize=14.5)
        ax.text(
            x[2], ypos,
            f'{row["ISRM PM2.5 PWM"]:.4f} / {row["ISRM BC PWM"]:.4f}',
            ha="center", va="center", fontsize=14.5,
        )
        ax.text(
            x[3], ypos,
            f'{row["Enhanced PM2.5 PWM"]:.4f} / {row["Enhanced BC PWM"]:.4f}',
            ha="center", va="center", fontsize=14.5,
        )
        ax.text(
            x[4], ypos,
            f'{row["PM2.5 change (%)"]:.2f}% / {row["BC change (%)"]:.2f}%',
            ha="center", va="center", fontsize=14.5,
        )

    ax.plot([0.02, 0.98], [0.055, 0.055], color="black", lw=4.0)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    data = prepare(pd.read_csv(args.summary))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "TABLE_1_CES_PWC_PM25_BC_data.csv", index=False)
    render(
        data,
        args.output_dir / "TABLE_1_CES_PWC_PM25_BC.png",
        args.output_dir / "TABLE_1_CES_PWC_PM25_BC.pdf",
    )
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
