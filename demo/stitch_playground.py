#!/usr/bin/env python3
"""
Combine all HTML outputs from demo/playground.ipynb into a single self-contained
HTML page suitable for screenshotting or converting to PDF.

Reads the SAVED notebook file (so you need to have run all cells in Jupyter
and saved before running this — the script reads cell outputs from disk,
not from a live kernel).

Usage:
    # 1. Open demo/playground.ipynb in classic Jupyter, run all cells, save.
    # 2. Run this:
    python3 demo/stitch_playground.py demo/playground.ipynb -o demo/demo_figure.html
    # 3. Open demo_figure.html in a browser, screenshot, or print to PDF.

Selecting which sections to include:
    --sections case,turns,brief,replay   (default: all)
        case   - the patient/scoreboard card from cell 4
        turns  - the turn-by-turn cards from cell 7 (or wherever turns ran)
        brief  - the content brief inspector from cell 9
        replay - the naive-prompting replay from cell 11
"""
import argparse
import json
import sys
from pathlib import Path


# Map cell-index ranges to logical sections. Adjust if your notebook structure
# changes — these are based on the playground.ipynb structure inspected.
SECTION_CELLS = {
    "case":   [4],            # case card + initial scoreboard
    "turns":  [7],            # turn cards (multiple if you ran more turns)
    "brief":  [9],            # content brief inspector
    "replay": [11],           # naive prompting replay + comparison
}

SECTION_TITLES = {
    "case":   "Case &amp; Scoreboard",
    "turns":  "Conversation Turns (Isolated Architecture)",
    "brief":  "Content Brief — What the Generator Sees",
    "replay": "Same Questions, Naive Prompting Baseline",
}


def extract_html_outputs(cell):
    """Pull all text/html outputs from a single cell, preserving order."""
    htmls = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "text/html" in data:
                html = "".join(data["text/html"])
                htmls.append(html)
    return htmls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("-o", "--output", default="demo_figure.html")
    ap.add_argument(
        "--sections",
        default="case,turns,brief,replay",
        help="Comma-separated list of sections to include. "
             "Options: case, turns, brief, replay. Default: all.",
    )
    ap.add_argument(
        "--title",
        default="You Can't Leak What You Don't Know — Demo Figure",
        help="Page title at the top of the figure.",
    )
    ap.add_argument(
        "--no-titles",
        action="store_true",
        help="Skip the section title bars between sections.",
    )
    args = ap.parse_args()

    sections_wanted = [s.strip() for s in args.sections.split(",") if s.strip()]
    for s in sections_wanted:
        if s not in SECTION_CELLS:
            print(f"Unknown section: {s}. Valid: {list(SECTION_CELLS)}",
                  file=sys.stderr)
            sys.exit(1)

    nb_path = Path(args.notebook)
    with open(nb_path) as f:
        nb = json.load(f)

    cells = nb["cells"]
    n_cells = len(cells)

    # Collect HTML in section order
    section_htmls = []  # list of (section_title, [html, html, ...])
    for section in sections_wanted:
        cell_indices = SECTION_CELLS[section]
        all_html = []
        for idx in cell_indices:
            if idx >= n_cells:
                print(f"  warning: cell index {idx} out of range "
                      f"(notebook has {n_cells} cells); section '{section}' "
                      f"will be partial or empty.", file=sys.stderr)
                continue
            html_chunks = extract_html_outputs(cells[idx])
            if not html_chunks:
                print(f"  warning: cell {idx} (section '{section}') has no "
                      f"HTML output. Did you run all cells before saving?",
                      file=sys.stderr)
            all_html.extend(html_chunks)
        section_htmls.append((SECTION_TITLES[section], all_html))

    # Compose the page
    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>{args.title}</title>")
    parts.append("""
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #fafafa;
    margin: 0;
    padding: 32px;
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
}
h1.demo-title {
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 24px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid #333;
    color: #222;
}
.section-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin: 32px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #ddd;
}
.section-title:first-of-type {
    margin-top: 16px;
}
</style>
""")
    parts.append("</head><body>")
    parts.append(f"<h1 class='demo-title'>{args.title}</h1>")

    for section_title, htmls in section_htmls:
        if not args.no_titles:
            parts.append(f"<div class='section-title'>{section_title}</div>")
        if not htmls:
            parts.append("<div style='color:#999;font-style:italic;padding:8px'>"
                         "(no output found — run the cell in Jupyter and save)</div>")
        else:
            for h in htmls:
                parts.append(h)

    parts.append("</body></html>")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        f.write("\n".join(parts))

    print(f"Wrote {out_path}")
    print(f"Sections included: {sections_wanted}")
    print()
    print("To convert to PDF:")
    print(f"    open {out_path}                    # open in browser")
    print(f"    # then File > Print > Save as PDF")
    print(f"    # or: wkhtmltopdf {out_path} demo_figure.pdf")


if __name__ == "__main__":
    main()
