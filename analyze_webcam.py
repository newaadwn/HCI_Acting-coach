from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "backend" / "acting_coach" / "analyze_webcam.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
