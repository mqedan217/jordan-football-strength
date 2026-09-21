# -*- coding: utf-8 -*-
"""Rebuild every notebook from source.

    python docs/build/build_all.py

Each notebook is generated from its own script in this directory, so the prose
and the code that produces a figure live together in one reviewable file. The
notebooks in `notebooks/` are BUILD OUTPUT -- edit the scripts here, never the
.ipynb, or the next rebuild discards the change.

Order matters only for reading: notebook 02 consumes the model outputs in
`outputs/`, and notebooks 03-05 consume the squad layer in `outputs/squad/`.
Both are produced by the scripts in `JordanData/`, not here.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRIPTS = [
    "build_nb1_data.py",
    "build_nb2_strength.py",
    "build_nb3_squad.py",
    "build_nb4_goals.py",
    "build_nb5_improve.py",
]

for s in SCRIPTS:
    runpy.run_path(str(HERE / s), run_name="__main__")
