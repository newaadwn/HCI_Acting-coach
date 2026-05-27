from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "face_landmarker.task"
DEFAULT_ACTOR_CSV_PATH = ROOT / "actor_expression.csv"
DEFAULT_USER_CSV_PATH = ROOT / "user_expression.csv"
DEFAULT_RECORDED_VIDEO_CSV_PATH = ROOT / "video_expression_mediapipe.csv"
DEFAULT_WEBCAM_CSV_PATH = ROOT / "webcam_expression_mediapipe_with_aihub_csv.csv"
DEFAULT_NEUTRAL_DISTRIBUTION_PATH = ASSET_DIR / "neutral_distribution.csv"
DEFAULT_ANGER_DISTRIBUTION_PATH = ASSET_DIR / "anger_distribution.csv"

SENSITIVITY = {
    "joy": 4.0,
    "happiness": 4.0,
    "sadness": 6.0,
    "anger": 4.0,
    "surprise": 4.5,
}

DEFAULT_EMOTION_KEYS = ["joy", "sadness", "anger", "surprise"]
MEDIAPIPE_NEUTRAL_THRESHOLD = 15
AIHUB_STD_FLOOR = 0.02
AIHUB_ANGER_SCORE_THRESHOLD = 58
AIHUB_DISTANCE_MARGIN = 0.95
AIHUB_MIN_ANGER_CORE = 0.03
LOW_INTENSITY_NEUTRAL_THRESHOLD = 40
LOW_INTENSITY_NEUTRAL_AVERAGE = 28
LOW_INTENSITY_NEUTRAL_ANGER_CORE = 0.22
LOW_INTENSITY_WEAK_ANGER_SCORE = 62

AIHUB_COMPARE_KEYS = [
    "browDownLeft",
    "browDownRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "mouthPressLeft",
    "mouthPressRight",
    "noseSneerLeft",
    "noseSneerRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "browInnerUp",
    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen",
]


def clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(value, max_value))


def raw_to_percent(raw_value: float, emotion_name: str) -> float:
    sensitivity = SENSITIVITY.get(emotion_name, 3.0)
    adjusted = raw_value * sensitivity
    if adjusted <= 0:
        return 0.0
    return clamp((adjusted / (adjusted + 1)) * 100)


def build_face_landmarker(model_path: Path):
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def blendshapes_to_dict(blendshapes) -> dict[str, float]:
    data: dict[str, float] = {}
    for category in blendshapes:
        data[category.category_name] = float(category.score)
    return data


def load_distribution_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    distribution_df = pd.read_csv(csv_path)
    required_columns = {"blendshape", "mean", "std"}
    missing_columns = required_columns - set(distribution_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    distribution: dict[str, dict[str, float]] = {}
    for _, row in distribution_df.iterrows():
        distribution[str(row["blendshape"])] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }

    return distribution


def save_distribution_csv(distribution: dict[str, dict[str, float]], csv_path: Path) -> None:
    rows = [
        {
            "blendshape": blendshape_name,
            "mean": stats["mean"],
            "std": stats["std"],
        }
        for blendshape_name, stats in distribution.items()
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")


NEUTRAL_DISTRIBUTION = load_distribution_csv(DEFAULT_NEUTRAL_DISTRIBUTION_PATH)
ANGER_DISTRIBUTION = load_distribution_csv(DEFAULT_ANGER_DISTRIBUTION_PATH)


def calculate_distribution_distance(
    data: dict[str, float],
    distribution: dict[str, dict[str, float]],
    keys: list[str] | None = None,
    std_floor: float = AIHUB_STD_FLOOR,
) -> float | None:
    squared_z_values: list[float] = []

    for key in keys or AIHUB_COMPARE_KEYS:
        if key not in distribution:
            continue

        value = float(data.get(key, 0.0))
        mean_value = distribution[key]["mean"]
        std_value = max(distribution[key]["std"], std_floor)
        z_value = (value - mean_value) / std_value
        squared_z_values.append(z_value**2)

    if not squared_z_values:
        return None

    return float(np.sqrt(np.mean(squared_z_values)))


def add_aihub_reference_scores(data: dict[str, float]) -> dict[str, float]:
    neutral_distance = calculate_distribution_distance(data, NEUTRAL_DISTRIBUTION, AIHUB_COMPARE_KEYS, AIHUB_STD_FLOOR)
    anger_distance = calculate_distribution_distance(data, ANGER_DISTRIBUTION, AIHUB_COMPARE_KEYS, AIHUB_STD_FLOOR)

    if neutral_distance is None or anger_distance is None:
        data["aihub_neutral_distance"] = None
        data["aihub_anger_distance"] = None
        data["aihub_neutral_score"] = 0.0
        data["aihub_anger_score"] = 0.0
        data["aihub_reference_emotion"] = "Unknown"
        return data

    total_distance = neutral_distance + anger_distance
    if total_distance == 0:
        neutral_score = 50.0
        anger_score = 50.0
    else:
        neutral_score = anger_distance / total_distance * 100
        anger_score = neutral_distance / total_distance * 100

    data["aihub_neutral_distance"] = neutral_distance
    data["aihub_anger_distance"] = anger_distance
    data["aihub_neutral_score"] = neutral_score
    data["aihub_anger_score"] = anger_score
    data["aihub_reference_emotion"] = "Anger" if anger_distance < neutral_distance else "Neutral"
    return data


def calculate_anger_core(data: dict[str, float]) -> float:
    return (
        data.get("browDownLeft", 0.0)
        + data.get("browDownRight", 0.0)
        + data.get("eyeSquintLeft", 0.0)
        + data.get("eyeSquintRight", 0.0)
        + data.get("mouthPressLeft", 0.0)
        + data.get("mouthPressRight", 0.0)
    ) / 6


def _calculate_emotion_raws(data: dict[str, float]) -> tuple[float, float, float, float]:
    mouth_smile_left = data.get("mouthSmileLeft", 0.0)
    mouth_smile_right = data.get("mouthSmileRight", 0.0)

    mouth_frown_left = data.get("mouthFrownLeft", 0.0)
    mouth_frown_right = data.get("mouthFrownRight", 0.0)

    mouth_press_left = data.get("mouthPressLeft", 0.0)
    mouth_press_right = data.get("mouthPressRight", 0.0)

    brow_inner_up = data.get("browInnerUp", 0.0)
    brow_down_left = data.get("browDownLeft", 0.0)
    brow_down_right = data.get("browDownRight", 0.0)
    brow_outer_up_left = data.get("browOuterUpLeft", 0.0)
    brow_outer_up_right = data.get("browOuterUpRight", 0.0)

    eye_squint_left = data.get("eyeSquintLeft", 0.0)
    eye_squint_right = data.get("eyeSquintRight", 0.0)
    eye_wide_left = data.get("eyeWideLeft", 0.0)
    eye_wide_right = data.get("eyeWideRight", 0.0)

    nose_sneer_left = data.get("noseSneerLeft", 0.0)
    nose_sneer_right = data.get("noseSneerRight", 0.0)

    jaw_open = data.get("jawOpen", 0.0)

    smile_avg = (mouth_smile_left + mouth_smile_right) / 2
    eye_wide_avg = (eye_wide_left + eye_wide_right) / 2
    mouth_press_avg = (mouth_press_left + mouth_press_right) / 2
    brow_outer_up_avg = (brow_outer_up_left + brow_outer_up_right) / 2
    brow_up_avg = (brow_inner_up + brow_outer_up_avg) / 2

    joy_raw = (
        mouth_smile_left * 0.45
        + mouth_smile_right * 0.45
        + eye_squint_left * 0.05
        + eye_squint_right * 0.05
    )

    sadness_raw = (
        mouth_frown_left * 0.30
        + mouth_frown_right * 0.30
        + brow_inner_up * 0.35
        + mouth_press_left * 0.025
        + mouth_press_right * 0.025
    )

    anger_raw = (
        brow_down_left * 0.23
        + brow_down_right * 0.23
        + eye_squint_left * 0.13
        + eye_squint_right * 0.13
        + nose_sneer_left * 0.10
        + nose_sneer_right * 0.10
        + mouth_press_left * 0.04
        + mouth_press_right * 0.04
    )

    if smile_avg < 0.15 and eye_wide_avg > 0.18:
        anger_raw += eye_wide_avg * 0.35

    if mouth_press_avg > 0.15:
        anger_raw += mouth_press_avg * 0.25

    if not (smile_avg < 0.15 and eye_wide_avg > 0.18):
        anger_raw *= 1 - jaw_open * 0.4

    surprise_raw = (
        jaw_open * 0.30
        + brow_up_avg * 0.30
        + eye_wide_left * 0.20
        + eye_wide_right * 0.20
    )

    anger_signal = (
        brow_down_left
        + brow_down_right
        + eye_squint_left
        + eye_squint_right
        + nose_sneer_left
        + nose_sneer_right
        + mouth_press_left
        + mouth_press_right
    ) / 8

    surprise_raw *= 1 - anger_signal * 0.5

    if jaw_open > 0.4 and brow_inner_up < 0.15 and eye_wide_avg < 0.15:
        surprise_raw *= 0.4

    if smile_avg < 0.15 and eye_wide_avg > 0.18 and jaw_open < 0.25:
        surprise_raw *= 0.5

    return joy_raw, sadness_raw, anger_raw, surprise_raw


def _append_emotion_scores(
    target: dict[str, float],
    joy_raw: float,
    sadness_raw: float,
    anger_raw: float,
    surprise_raw: float,
    *,
    add_neutral: bool,
) -> dict[str, float]:
    target["joy_raw"] = joy_raw
    target["sadness_raw"] = sadness_raw
    target["anger_raw"] = anger_raw
    target["surprise_raw"] = surprise_raw

    target["joy"] = raw_to_percent(joy_raw, "joy")
    target["sadness"] = raw_to_percent(sadness_raw, "sadness")
    target["anger"] = raw_to_percent(anger_raw, "anger")
    target["surprise"] = raw_to_percent(surprise_raw, "surprise")
    target["happiness_raw"] = target["joy_raw"]
    target["happiness"] = target["joy"]

    if add_neutral:
        max_emotion_percent = max(target["joy"], target["sadness"], target["anger"], target["surprise"])
        target["neutral"] = clamp(100 - max_emotion_percent)
        target["neutral_detected"] = max_emotion_percent < MEDIAPIPE_NEUTRAL_THRESHOLD

    return target


def calculate_performance_emotions(data: dict[str, float]) -> dict[str, float]:
    joy_raw, sadness_raw, anger_raw, surprise_raw = _calculate_emotion_raws(data)
    return _append_emotion_scores(
        data,
        joy_raw,
        sadness_raw,
        anger_raw,
        surprise_raw,
        add_neutral=True,
    )


def calculate_actor_emotions(data: dict[str, float]) -> dict[str, float]:
    return calculate_performance_emotions(data)


def calculate_user_emotions(data: dict[str, float]) -> dict[str, float]:
    return calculate_performance_emotions(data)


def get_dominant_emotion(
    scores: dict[str, float],
    keys: list[str] | None = None,
    include_data: bool = False,
) -> tuple[str, float] | tuple[str, float, dict[str, float]]:
    label_map = {
        "joy": "Joy",
        "happiness": "Happiness",
        "sadness": "Sadness",
        "anger": "Anger",
        "surprise": "Surprise",
    }
    score_keys = keys or DEFAULT_EMOTION_KEYS
    emotion_scores = {label_map[key]: scores.get(key, 0.0) for key in score_keys}
    mediapipe_emotion = max(emotion_scores, key=emotion_scores.get)
    mediapipe_percent = emotion_scores[mediapipe_emotion]

    scores["mediapipe_label"] = mediapipe_emotion
    scores["mediapipe_percent"] = mediapipe_percent

    aihub_neutral_distance = scores.get("aihub_neutral_distance")
    aihub_anger_distance = scores.get("aihub_anger_distance")
    aihub_anger_score = float(scores.get("aihub_anger_score", 0.0) or 0.0)
    aihub_neutral_score = float(scores.get("aihub_neutral_score", 0.0) or 0.0)

    anger_core = calculate_anger_core(scores)
    scores["anger_core"] = anger_core
    expressive_average = float(sum(emotion_scores.values()) / max(1, len(emotion_scores)))
    scores["expressive_average"] = expressive_average
    neutral_percent = max(
        float(scores.get("neutral", clamp(100 - mediapipe_percent))),
        aihub_neutral_score,
    )

    if aihub_neutral_distance is not None and aihub_anger_distance is not None:
        anger_is_closer = aihub_anger_distance < aihub_neutral_distance * AIHUB_DISTANCE_MARGIN
    else:
        anger_is_closer = False

    low_intensity_neutral = (
        mediapipe_percent < LOW_INTENSITY_NEUTRAL_THRESHOLD
        and expressive_average < LOW_INTENSITY_NEUTRAL_AVERAGE
        and anger_core < LOW_INTENSITY_NEUTRAL_ANGER_CORE
        and aihub_anger_score < LOW_INTENSITY_WEAK_ANGER_SCORE
    )

    if low_intensity_neutral:
        scores["final_source"] = "csv_label_aux"
        result = ("Neutral", neutral_percent)
    elif mediapipe_percent < MEDIAPIPE_NEUTRAL_THRESHOLD:
        if (
            anger_is_closer
            and aihub_anger_score >= AIHUB_ANGER_SCORE_THRESHOLD
            and anger_core >= AIHUB_MIN_ANGER_CORE
        ):
            scores["final_source"] = "csv_label_aux"
            result = ("Anger", max(30.0, aihub_anger_score))
        else:
            scores["final_source"] = "csv_label_aux"
            result = ("Neutral", neutral_percent)
    elif mediapipe_emotion == "Anger":
        scores["final_source"] = "mediapipe+csv_score"
        result = ("Anger", max(mediapipe_percent, aihub_anger_score))
    elif anger_is_closer and aihub_anger_score >= 65 and anger_core >= AIHUB_MIN_ANGER_CORE:
        scores["final_source"] = "csv_label_aux"
        result = ("Anger", max(mediapipe_percent, aihub_anger_score))
    else:
        scores["final_source"] = "mediapipe"
        result = (mediapipe_emotion, mediapipe_percent)

    if include_data:
        return result[0], result[1], scores
    return result


def get_emotion_color(emotion: str) -> tuple[int, int, int]:
    colors = {
        "Joy": (0, 255, 255),
        "Happiness": (0, 255, 255),
        "Sadness": (255, 0, 0),
        "Anger": (0, 0, 255),
        "Surprise": (255, 0, 255),
        "Neutral": (255, 255, 255),
    }
    return colors.get(emotion, (255, 255, 255))


def get_face_box(face_landmarks, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x_values = [int(landmark.x * frame_width) for landmark in face_landmarks]
    y_values = [int(landmark.y * frame_height) for landmark in face_landmarks]
    x_min = max(min(x_values) - 25, 0)
    y_min = max(min(y_values) - 35, 0)
    x_max = min(max(x_values) + 25, frame_width)
    y_max = min(max(y_values) + 25, frame_height)
    return x_min, y_min, x_max, y_max


def draw_emotion_box(frame, box: tuple[int, int, int, int], emotion: str, percent: float) -> None:
    if box is None or emotion == "Neutral":
        return

    x_min, y_min, x_max, y_max = box
    box_color = get_emotion_color(emotion)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)

    text = f"{emotion} {percent:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    text_x = x_min
    text_y = max(y_min - 12, text_height + 12)

    cv2.rectangle(
        frame,
        (text_x, text_y - text_height - 10),
        (text_x + text_width + 12, text_y + 6),
        box_color,
        -1,
    )
    cv2.putText(frame, text, (text_x + 6, text_y), font, font_scale, (0, 0, 0), thickness)
