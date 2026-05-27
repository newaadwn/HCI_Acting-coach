This folder contains the reusable HCI-acting-coach MediaPipe pipeline adapted from the final updated repository.

Files:
- `common.py`: shared MediaPipe setup and the latest emotion scoring formulas
- `analyze_actor_video.py`: analyzes a reference actor video and writes `actor_expression.csv`
- `analyze_user_video.py`: analyzes a recorded user take with `anger_distribution.csv` and `neutral_distribution.csv`, then writes `video_expression_mediapipe.csv`
- `analyze_webcam.py`: runs the live desktop webcam flow with the same AI-Hub anger/neutral CSV distributions, then writes `webcam_expression_mediapipe_with_aihub_csv.csv`

Example commands:

```bash
python3 backend/acting_coach/analyze_actor_video.py /path/to/actor_video.mp4
python3 backend/acting_coach/analyze_user_video.py /path/to/user_take.webm --headless
python3 backend/acting_coach/analyze_webcam.py
```

Notes:
- `analyze_actor_video.py` is the project-structured replacement for the upstream `analyze_video_mediapipe_realtime.py`.
- The Node server uses `--headless` video analysis for uploads and reads the generated `dominant_emotion` / `dominant_percent` columns through JSON responses.
- The desktop webcam route uses `analyze_webcam.py` directly and reports `final_source` as `mediapipe`, `csv_label_aux`, or `mediapipe+csv_score`.
- All scripts use `backend/face_landmarker.task` by default.
