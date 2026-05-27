from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_WEBCAM_CSV_PATH,
    add_aihub_reference_scores,
    blendshapes_to_dict,
    build_face_landmarker,
    calculate_user_emotions,
    draw_emotion_box,
    get_dominant_emotion,
    get_face_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a live webcam performance with MediaPipe and AI-Hub anger/neutral CSV distributions."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Specific camera index to try first. Falls back to auto-scan when unavailable.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Highest camera index to scan when auto-detecting a webcam.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to face_landmarker.task.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_WEBCAM_CSV_PATH,
        help="Path to save webcam_expression_mediapipe_with_aihub_csv.csv.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=3,
        help="Analyze every Nth frame.",
    )
    return parser.parse_args()


def create_video_capture(camera_index: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def open_camera(preferred_index: int | None, max_camera_index: int) -> tuple[cv2.VideoCapture | None, int | None]:
    candidate_indices: list[int] = []

    if preferred_index is not None:
        candidate_indices.append(preferred_index)

    candidate_indices.extend(index for index in range(max_camera_index + 1) if index not in candidate_indices)

    for camera_index in candidate_indices:
        cap = create_video_capture(camera_index)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, camera_index

        cap.release()

    return None, None


def draw_status_text(frame, text: str, y: int, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(
        frame,
        text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def detect_face(detector, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks or not detection_result.face_blendshapes:
        return None, None

    return detection_result.face_landmarks[0], detection_result.face_blendshapes[0]


def main() -> int:
    args = parse_args()
    detector = build_face_landmarker(args.model_path)
    cap, camera_index = open_camera(args.camera_index, args.max_camera_index)

    if cap is None:
        print("Could not open a webcam. Check the camera connection and permissions.")
        return 1

    print("Webcam analysis started with the AI-Hub anger/neutral CSV pipeline.")
    print(f"Using camera index: {camera_index}")
    print("Press q to finish and save the CSV.")

    results: list[dict[str, float]] = []
    frame_idx = 0
    start_time = time.time()
    last_box = None
    last_emotion = "Neutral"
    last_percent = 0.0
    last_source = "mediapipe"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Could not read a webcam frame.")
                break

            frame_idx += 1
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            if frame_idx % max(1, args.sample_every) == 0:
                face_landmarks, blendshapes = detect_face(detector, frame)

                if face_landmarks is not None and blendshapes is not None:
                    raw_blendshapes = blendshapes_to_dict(blendshapes)
                    data = {
                        "frame": frame_idx,
                        "time": time.time() - start_time,
                    }
                    data.update(raw_blendshapes)
                    data = calculate_user_emotions(data)
                    data = add_aihub_reference_scores(data)
                    emotion, percent = get_dominant_emotion(data)
                    data["dominant_emotion"] = emotion
                    data["dominant_percent"] = percent

                    last_box = get_face_box(face_landmarks, width, height)
                    last_emotion = emotion
                    last_percent = percent
                    last_source = str(data.get("final_source", "mediapipe"))
                    results.append(data)

            if last_box is not None:
                draw_emotion_box(frame, last_box, last_emotion, last_percent)

            draw_status_text(frame, f"q: finish | source: {last_source}", height - 60)
            draw_status_text(frame, f"emotion: {last_emotion} {last_percent:.0f}%", height - 30)
            cv2.imshow("HCI Acting Coach Webcam Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user input.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Saving the collected webcam data.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not results:
        print("No face data was saved. Check that your face is visible to the webcam.")
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved webcam CSV to {args.output_csv}")
    print(f"Stored frames: {len(df)}")

    summary_columns = [
        "joy",
        "sadness",
        "anger",
        "surprise",
        "aihub_neutral_score",
        "aihub_anger_score",
        "aihub_neutral_distance",
        "aihub_anger_distance",
        "aihub_reference_emotion",
        "mediapipe_label",
        "mediapipe_percent",
        "final_source",
        "dominant_emotion",
        "dominant_percent",
    ]
    existing_columns = [column for column in summary_columns if column in df.columns]

    print("\nResult summary:")
    print(df[existing_columns].tail())
    print("\nFinal emotion counts:")
    print(df["dominant_emotion"].value_counts())

    if "final_source" in df.columns:
        print("\nFinal source counts:")
        print(df["final_source"].value_counts())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
