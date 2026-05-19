This folder contains the reusable HCI-acting-coach MediaPipe pipeline adapted from the final updated repository.

Files:
- `common.py`: shared MediaPipe setup and the latest emotion scoring formulas
- `analyze_actor_video.py`: analyzes a reference actor video and writes `actor_expression.csv`
- `analyze_user_video.py`: analyzes a recorded user take from a video file and can optionally use a personalized neutral baseline profile before writing `user_expression.csv`
- `analyze_webcam.py`: runs the live desktop webcam flow, calibrates a 3-second neutral baseline, and writes `user_expression.csv`

Example commands:

```bash
python3 backend/acting_coach/analyze_actor_video.py /path/to/actor_video.mp4
python3 backend/acting_coach/analyze_user_video.py /path/to/user_take.webm --headless
python3 backend/acting_coach/analyze_webcam.py
```

Optional personalized user video analysis:

```bash
python3 backend/acting_coach/analyze_user_video.py /path/to/user_take.webm --headless --neutral-profile /path/to/user_neutral_profile.json
```

Notes:
- `analyze_actor_video.py` is the project-structured replacement for the upstream `analyze_video_mediapipe_realtime.py`.
- The Node server uses `--headless` video analysis for uploads and forwards the browser-calibrated neutral profile when it is available.
- The desktop webcam route uses `analyze_webcam.py` directly and performs its own neutral calibration before scoring.
- All scripts use `backend/face_landmarker.task` by default.
