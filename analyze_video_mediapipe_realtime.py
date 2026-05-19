from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "backend" / "acting_coach" / "analyze_actor_video.py"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python3 analyze_video_mediapipe_realtime.py /path/to/actor_video.mp4")
        raise SystemExit(1)

    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
