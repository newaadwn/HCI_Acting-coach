import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from converter import convert_csv_to_frontend_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "actor_expression.csv"
FPS = 30


@app.get("/api/actor-emotions")
def get_actor_emotions():
    data = convert_csv_to_frontend_json(CSV_PATH, fps=FPS)
    return data


@app.websocket("/ws/actor-emotions")
async def websocket_actor_emotions(websocket: WebSocket):
    await websocket.accept()

    data = convert_csv_to_frontend_json(CSV_PATH, fps=FPS)

    try:
        for payload in data:
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)

        await websocket.close()

    except Exception:
        await websocket.close()
