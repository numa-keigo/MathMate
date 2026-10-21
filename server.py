from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# server.pyが置かれているフォルダ
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()


# JavaScriptから受け取るデータの形
class ChatRequest(BaseModel):
    message: str


# チャット用のAPI
@app.post("/api/chat")
def chat(request: ChatRequest):
    return {
        "reply": f"「{request.message}」をPythonで受け取りました！"
    }


# トップページ
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# CSSとJavaScriptを配信する
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )