import pandas as pd

EMOTION_COLS = ["happiness", "surprise", "anger", "sadness"]


def convert_row_to_frontend_payload(row, fps=30):
    raw_scores = {
        emotion: float(row[emotion])
        for emotion in EMOTION_COLS
    }

    total = sum(raw_scores.values())

    if total == 0:
        scores = {
            emotion: 0
            for emotion in EMOTION_COLS
        }
    else:
        scores = {
            emotion: round(value / total * 100, 1)
            for emotion, value in raw_scores.items()
        }

    dominant_emotion = max(scores, key=scores.get)

    return {
        "frame": int(row["frame"]),
        "timestamp_ms": int(row["frame"] / fps * 1000),
        "dominantEmotion": dominant_emotion,
        "dominantScore": scores[dominant_emotion],
        "scores": scores,
        "rawScores": raw_scores,
    }


def convert_csv_to_frontend_json(csv_path, fps=30):
    df = pd.read_csv(csv_path)

    result = []

    for _, row in df.iterrows():
        payload = convert_row_to_frontend_payload(row, fps)
        result.append(payload)

    return result
