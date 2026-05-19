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
    DEFAULT_USER_CSV_PATH,
    blendshapes_to_dict,
    build_face_landmarker,
    build_user_neutral_distribution,
    draw_emotion_box,
    get_face_box,
    save_distribution_csv,
    save_user_neutral_profile,
    score_personalized_user_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a live webcam performance with user neutral baseline calibration."
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
        default=DEFAULT_USER_CSV_PATH,
        help="Path to save the webcam analysis CSV.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=3,
        help="Analyze every Nth frame.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=3.0,
        help="Seconds to collect a neutral baseline before analysis.",
    )
    parser.add_argument(
        "--baseline-min-samples",
        type=int,
        default=10,
        help="Minimum neutral samples required before the run can continue.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=None,
        help="Optional path to save the calibrated neutral profile as JSON or CSV.",
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


def collect_user_neutral_baseline(
    cap: cv2.VideoCapture,
    detector,
    sample_every: int,
    calibration_seconds: float,
    baseline_min_samples: int,
) -> dict[str, dict[str, float]] | None:
    print("\n[Baseline] Look at the camera with a neutral face for calibration.")
    print("[Baseline] This neutral profile becomes the personal reference for emotion sensitivity.")

    samples: list[dict[str, float]] = []
    calibration_start_time = time.time()
    calibration_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Baseline] Could not read a webcam frame.")
            return None

        calibration_frame_idx += 1
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        elapsed = time.time() - calibration_start_time
        remaining = max(0.0, calibration_seconds - elapsed)

        if calibration_frame_idx % max(1, sample_every) == 0:
            face_landmarks, blendshapes = detect_face(detector, frame)
            if face_landmarks is not None and blendshapes is not None:
                raw_blendshapes = blendshapes_to_dict(blendshapes)
                samples.append(raw_blendshapes)

                box = get_face_box(face_landmarks, width, height)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)

        draw_status_text(frame, "Neutral calibration: keep a neutral face", 35)
        draw_status_text(frame, f"Remaining: {remaining:.1f}s | samples: {len(samples)}", 70)
        cv2.imshow("HCI Acting Coach Webcam Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Baseline] Stopped by user input.")
            return None

        if elapsed >= calibration_seconds:
            break

    if len(samples) < baseline_min_samples:
        print(
            f"[Baseline] Not enough neutral samples. "
            f"Collected: {len(samples)}, required: {baseline_min_samples}"
        )
        return None

    print(f"[Baseline] Neutral calibration complete with {len(samples)} samples.")
    return build_user_neutral_distribution(samples)


def maybe_save_neutral_profile(
    neutral_distribution: dict[str, dict[str, float]],
    output_path: Path | None,
) -> None:
    if output_path is None:
        return

    if output_path.suffix.lower() == ".csv":
        save_distribution_csv(neutral_distribution, output_path)
        print(f"[Baseline] Saved neutral profile CSV to {output_path}")
        return

    save_user_neutral_profile(neutral_distribution, output_path)
    print(f"[Baseline] Saved neutral profile JSON to {output_path}")


def main() -> int:
    args = parse_args()
    detector = build_face_landmarker(args.model_path)
    cap, camera_index = open_camera(args.camera_index, args.max_camera_index)

    if cap is None:
        print("Could not open a webcam. Check the camera connection and permissions.")
        return 1

    print("Webcam analysis started with personalized neutral baseline calibration.")
    print(f"Using camera index: {camera_index}")

    neutral_distribution = collect_user_neutral_baseline(
        cap,
        detector,
        args.sample_every,
        args.calibration_seconds,
        args.baseline_min_samples,
    )

    if neutral_distribution is None:
        cap.release()
        cv2.destroyAllWindows()
        return 1

    maybe_save_neutral_profile(neutral_distribution, args.baseline_output)
    print("Personalized emotion analysis is now active. Press q to finish and save the CSV.")

    results: list[dict[str, float]] = []
    frame_idx = 0
    start_time = time.time()
    last_box = None
    last_emotion = "Neutral"
    last_percent = 0.0
    last_source = "user_neutral_baseline"

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
                    data.update(score_personalized_user_frame(raw_blendshapes, neutral_distribution))

                    last_box = get_face_box(face_landmarks, width, height)
                    last_emotion = str(data.get("dominant_emotion", "Neutral"))
                    last_percent = float(data.get("dominant_percent", 0.0) or 0.0)
                    last_source = str(data.get("final_source", "relative_blendshape"))
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
    print(f"Saved user CSV to {args.output_csv}")
    print(f"Stored frames: {len(df)}")

    summary_columns = [
        "joy",
        "sadness",
        "anger",
        "surprise",
        "user_neutral_score",
        "user_neutral_distance",
        "user_neutral_delta_energy",
        "aihub_anger_score",
        "aihub_anger_distance",
        "anger_core_delta",
        "mediapipe_label",
        "mediapipe_percent",
        "is_user_neutral",
        "anger_csv_confirms",
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
