#!/usr/bin/env python3
"""Build the single offline page: demo/argus_demo.html (template, scripts, colour maps and field data inlined).

    python demo/export_demo.py && python demo/build_html.py [observation name, default f294]
"""

import json
import sys
from pathlib import Path

import matplotlib

DEMO = Path(__file__).resolve().parent
FIELD = sys.argv[1] if len(sys.argv) > 1 else "f294"  # the observation exported (demo/build/<name>.json)


def lut(name):
    cmap = matplotlib.colormaps[name].resampled(256)
    return [[round(255 * c) for c in cmap(i)[:3]] for i in range(256)]


def main():
    page = (DEMO / "src" / "page.html").read_text()
    parts = {
        "/*COLORMAPS*/": json.dumps({"magma": lut("magma"), "RdBu_r": lut("RdBu_r")}),
        "/*DATA*/": (DEMO / "build" / f"{FIELD}.json").read_text(),
        "/*SIM*/": (DEMO / "src" / "sim.js").read_text(),
        "/*SEP*/": (DEMO / "src" / "sep.js").read_text(),
        "/*MEASURE*/": (DEMO / "src" / "measure.js").read_text(),
        "/*PLOT*/": (DEMO / "src" / "plot.js").read_text(),
        "/*APP*/": (DEMO / "src" / "app.js").read_text(),
    }
    for key, text in parts.items():
        page = page.replace(key, text.replace("</script", "<\\/script"))
    out = DEMO / "argus_demo.html"
    out.write_text(page)
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
