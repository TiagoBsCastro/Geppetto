#!/usr/bin/env python3
"""Create a read-only PDF report from portable galaxy-diagnostic products.

Requires reportlab (available in /usr/bin/python3 in the test environment).
The scientific data and plots are made by diagnose_pinocchio_galaxies.py.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


def paragraph(pdf, text, x, y, width, size=11):
    style = ParagraphStyle("body", fontName="Helvetica", fontSize=size, leading=size*1.35, textColor=colors.HexColor("#252525"))
    block = Paragraph(text, style)
    _, height = block.wrap(width, 2000)
    block.drawOn(pdf, x, y-height)
    return y-height-10


def table(pdf, data, x, y, widths, font_size=9):
    style = ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size, leading=font_size*1.3)
    cells = [[Paragraph(escape(str(value)), style) for value in row] for row in data]
    block = Table(cells, colWidths=widths, repeatRows=1)
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eceb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f6f7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#607d78")),
    ]))
    _, height = block.wrap(sum(widths), 2000)
    block.drawOn(pdf, x, y-height)
    return y-height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="examples/pinocchio_galaxy_diagnostics")
    args = parser.parse_args()
    directory = Path(args.input_dir)
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"audit_summary.json").read_text())
    output = directory/"diagnostic_report.pdf"
    page_width, page_height = 13.2*72, 9.6*72
    pdf = canvas.Canvas(str(output), pagesize=(page_width, page_height), invariant=1)
    pdf.setTitle("Frozen PINOCCHIO-hodpy lightcone diagnostic audit")
    pdf.setAuthor("GEPPETTO scientific validation")
    y = page_height-35
    y = paragraph(pdf, "<b>PINOCCHIO-hodpy: frozen-model diagnostic audit</b>", 38, y, page_width-76, 21)
    y = paragraph(pdf, "Real L3870N4096/000 lightcone | 660,873 galaxies | 5,435,239 native halo occurrences<br/>"
                  "No HOD, colour, satellite, or cosmological parameter was changed. No fitting was performed.", 38, y, page_width-76, 12)
    y = paragraph(pdf, "<b>Reference classes:</b> only the SDSS/GAMA luminosity function is an independently specified observational target. "
                  "HOD occupations, colours, NFW radii and Gaussian velocities are predictions of the model under test. "
                  "The native HMF is measured simulation input. Clustering is an unfitted diagnostic, not a validation against observations.", 38, y, page_width-76)
    y = paragraph(pdf, "<b>Conventions:</b> M_PIN is native fragmentation-group mass [Msun/h], not a spherical-overdensity mass. "
                  "Native floor: 10 particles; painting cut: 32 particles = 2.93126e12 Msun/h. "
                  "Positions: comoving Mpc/h; velocities: proper peculiar km/s. AB magnitudes: ^0.1 M_r - 5log10(h); rest-frame colour: ^0.1(g-r). "
                  "Inner cone: 65 deg. LF claims: M_r &lt;= -22 and 0.05 &lt;= z_cos &lt; 0.32; r&lt;20 where labelled. "
                  "The underlying M_r&lt;-21.5 catalogue is not a complete faint flux-limited survey.", 38, y, page_width-76)
    y = paragraph(pdf, "<b>Cosmology:</b> Omega_m=0.3913, Omega_b=0.0419, Omega_DE=0.6087, h=0.7276, n_s=0.9312, "
                  "sigma8=0.60850325, w0=-1.168, wa=-0.6605. Simulation seed=1386; galaxy seed=20260917.", 38, y, page_width-76, 10)
    with (directory/"luminosity_function.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if float(r["magnitude_bright"]) >= -22.8-1.e-8
                and float(r["magnitude_faint"]) <= -22+1.e-8]
    contents = [["z range", "M_r interval", "N", "HOD residual", "Mock residual", "Allowed", "Status"]]
    for row in rows:
        contents.append([f"{float(row['z_min']):.2f}-{float(row['z_max']):.2f}",
                         f"[{float(row['magnitude_bright']):.1f},{float(row['magnitude_faint']):.1f})", row["count"],
                         f"{100*float(row['deterministic_residual']):+.4f}%", f"{100*float(row['mock_residual']):+.3f}%",
                         f"{100*float(row['allowed_relative_error']):.2f}%", "PASS" if row["passed"] == "True" else "REVIEW"])
    y = table(pdf, contents, 38, y-4, [110, 130, 85, 115, 115, 100, 100])
    y = paragraph(pdf, "<b>LF uncertainty:</b> max(conditional HOD sigma, 32-region spatial-jackknife sigma); tolerance max(5%,3 sigma). "
                  "Full densities, volumes, errors and completeness estimates are saved in CSV files. "
                  "Every following plot states its own selection, sample size, units and uncertainty model.", 38, y-12, page_width-76, 10)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(38, 16, "Reproducible code, binned data, provenance and full methods: report.md, manifest.json, code/ and *.csv")
    pdf.showPage()
    for number, figure in enumerate(manifest["figures"], start=1):
        image = ImageReader(str(directory/figure["file"]))
        width, height = image.getSize()
        rendered_height = page_width*height/width
        pdf.setPageSize((page_width, rendered_height+32))
        pdf.drawImage(image, 0, 5, width=page_width, height=rendered_height)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(38, rendered_height+19, f"Frozen-model audit | Figure {number}/{len(manifest['figures'])} | {figure['file']}")
        pdf.showPage()
    pdf.setPageSize((page_width, page_height))
    y = paragraph(pdf, "<b>Checks, visible discrepancies and likely causes</b>", 38, page_height-35, page_width-76, 18)
    contents = [["Check", "Result", "Evidence / discrepancy", "Interpretation / likely cause"]]
    for row in summary["checks"]:
        contents.append([row["check"], row["status"], row["evidence"], row["cause"]])
    y = table(pdf, contents, 38, y-4, [130, 85, 320, 340], font_size=9)
    if y < 45:
        raise ValueError("PDF conclusion table overflows its page")
    paragraph(pdf, "No parameter was adjusted to improve any diagnostic. References: "
              "<link href='https://github.com/amjsmith/hodpy' color='#2166ac'>hodpy</link>, pinned revision d303bef8; "
              "<link href='https://arxiv.org/abs/1701.06581' color='#2166ac'>Smith et al. (2017)</link>.", 38, y-14, page_width-76, 10)
    pdf.save()
    code = directory/"code"
    code.mkdir(exist_ok=True)
    shutil.copy2(__file__, code/Path(__file__).name)
    print(output)


if __name__ == "__main__":
    main()
