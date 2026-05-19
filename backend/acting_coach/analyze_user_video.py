from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_USER_CSV_PATH,
    add_aihub_reference_scores,
    blendshapes_to_dict,
    build_face_landmarker,
    calculate_user_emotions,
    draw_emotion_box,
    get_dominant_emotion,
    get_face_box,
    load_user_neutral_profile,
    score_personalized_user_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a recorded user performance video with optional personalized neutral calibration."
    )
    parser.add_argument("video_path", type=Path, help="Path to the recorded user performance video file.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_USER_CSV_PATH, help="Path to save user_expression.csv.")
    parser.add_argument("--sample-every", type=int, default=3, help="Analyze every Nth frame.")
    parser.add_argument("--headless", action="store_true", help="Disable the OpenCV preview window.")
    parser.add_argument(
        "--neutral-profile",
        type=Path,
        default=None,
        help="Optional JSON file containing the browser-calibrated neutral baseline profile.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    detector = build_face_landmarker(args.model_path)
    cap = cv2.VideoCapture(str(args.video_path))

    if not cap.isOpened():
        print(f"Could not open video: {args.video_path}")
        return 1

    neutral_distribution = None
    if args.neutral_profile is not None:
        neutral_distribution = load_user_neutral_profile(args.neutral_profile)
        print(f"Loaded personalized neutral profile from {args.neutral_profile}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    delay = max(1, int(1000 / fps))

    if neutral_distribution is None:
        print("User performance video opened. Starting analysis with the default actor-style pipeline.")
    else:
        print("User performance video opened. Starting analysis with the personalized neutral baseline pipeline.")

    if not args.headless:
        print("Press q to stop early and save the CSV.")

    results: list[dict[str, float]] = []
    frame_idx = 0
    last_box = None
    last_emotion = "Neutral"
    last_percent = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video analysis complete.")
            break

        frame_idx += 1
        height, width, _ = frame.shape

        if frame_idx % max(1, args.sample_every) == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks and detection_result.face_blendshapes:
                face_landmarks = detection_result.face_landmarks[0]
                raw_blendshapes = blendshapes_to_dict(detection_result.face_blendshapes[0])
                data = {"frame": frame_idx}

                if neutral_distribution is None:
                    data.update(raw_blendshapes)
                    data = calculate_user_emotions(data)
                    data = add_aihub_reference_scores(data)
                    emotion, percent = get_dominant_emotion(data)
                    data["dominant_emotion"] = emotion
                    data["dominant_percent"] = percent
                else:
                    data.update(score_personalized_user_frame(raw_blendshapes, neutral_distribution))
                    emotion = str(data.get("dominant_emotion", "Neutral"))
                    percent = float(data.get("dominant_percent", 0.0) or 0.0)

                last_box = get_face_box(face_landmarks, width, height)
                last_emotion = emotion
                last_percent = percent
                results.append(data)

        if args.headless:
            continue

        if last_box is not None:
            draw_emotion_box(frame, last_box, last_emotion, last_percent)

        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0.0
        cv2.putText(
            frame,
            f"Progress: {progress:.1f}%",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("HCI Acting Coach User Video Analysis", frame)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("Stopped early by user input.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not results:
        print("No face data was saved. Check that the face is visible in the video.")
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved user CSV to {args.output_csv}")
    print(f"Stored frames: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
