import os
import getpass
from pathlib import Path
from threading import Lock
from typing import Literal

# ライブラリを読み込む前に設定する
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "GPU-490452e1-93a9-2524-849a-2ec1f1098bdf"
)
os.environ["HF_HOME"] = (
    f"/shared_workspace/{getpass.getuser()}/MathMate/huggingface"
)

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_ID = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """
あなたは、中学生の数学の学習を支援する先生です。
・親しみやすく、わかりやすい日本語を使う。
・1回の返答は2〜4文程度にする。
・最終的な答えをすぐに教えず、次の一歩のヒントを出す。
・これまでの会話と、生徒の考え方や途中式に合わせて説明する。
・誤りがあれば、どこを見直すか具体的に伝える。
・一度に質問するのは1つだけにする。
・返答の最後には、次の一歩を促す質問を1つ入れる。
・方程式では、両辺に同じ操作をすることを明確にする。
・生徒が正しい答えに到達した場合は、正解と伝えてよい。
""".strip()


# 会話履歴の1メッセージ
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


# JavaScriptから受け取るデータ
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=12,
    )


def create_app(tokenizer, model):
    app = FastAPI()

    # GPUでの生成処理が重ならないようにする
    generation_lock = Lock()

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        # 最初にAIへの指示を入れる
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # これまでの会話を追加する
        messages.extend(
            {"role": item.role, "content": item.content}
            for item in request.history
        )

        # 今回の質問を追加する
        messages.append(
            {"role": "user", "content": request.message}
        )

        with generation_lock:
            # 会話全体をモデル用の形式に整える
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # 文章をトークンIDに変換し、GPUへ送る
            inputs = tokenizer(
                text,
                return_tensors="pt",
            ).to("cuda:0")

            # 回答を生成する
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 入力部分を除き、生成された部分だけ取り出す
            generated_tokens = outputs[
                0, inputs["input_ids"].shape[1]:
            ]

            answer = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

        return {"reply": answer}

    # トップページを返す
    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    # CSSとJavaScriptを配信する
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

    return app


def main():
    # モデルは起動時に一度だけ読み込む
    print("モデルを読み込み中...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    print("モデルの読み込み完了！", flush=True)

    app = create_app(tokenizer, model)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()