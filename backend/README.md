This backend folder now combines two sources:

- `actor_expression.csv`: actor baseline data used by the current Node app
- `user_expression.csv`: sample user output from `HCI-acting-coach`
- `face_landmarker.task`: MediaPipe model asset from `HCI-acting-coach`
- `acting_coach/`: reusable Python CLI pipeline rebuilt from `HCI-acting-coach`
- `main.py` / `converter.py`: preserved FastAPI files from the earlier MediaPipe_Backend merge
- `generated/`: runtime CSV outputs created by the local Node + Python integration

Current app runtime:

- `server.mjs` is the main server used by the browser app
- it exposes HCI-based analysis endpoints for uploaded reference videos, recorded user takes, and the desktop webcam flow
- recorded user takes are written as `video_expression_mediapipe.csv`
- desktop webcam runs are written as `webcam_expression_mediapipe_with_aihub_csv.csv`
- both user-take paths use MediaPipe plus `anger_distribution.csv` / `neutral_distribution.csv`, and the frontend reads `dominant_emotion` / `dominant_percent` through the server JSON output

Python pipeline:

```bash
python3 backend/acting_coach/analyze_actor_video.py /path/to/actor_video.mp4
python3 backend/acting_coach/analyze_user_video.py /path/to/user_take.webm --headless
python3 backend/acting_coach/analyze_webcam.py
```

Install dependencies first if you want to run the Python pipeline:

```bash
python3 -m pip install -r backend/requirements.txt
```
