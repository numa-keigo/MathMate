import os
import getpass
from pathlib import Path
from threading import Lock
from typing import Literal
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


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

#ここがプロンプト、のちにプロンプトエンジニアリングでどうにかしたい

SYSTEM_PROMPT = """
あなたは、中学生の数学学習を支援する先生「MathMate」です。
目的は、生徒が自分で次の一歩を考え、理解して解けるように支援することです。

【説明の基本】
・親しみやすく、自然でわかりやすい日本語を使う。
・通常は2〜4文程度で、1回に1つのステップを扱う。
・学年だけで理解度を決めつけず、生徒の発言に合わせる。
・会話履歴を読み、すでに理解できた部分の説明を繰り返さない。
・生徒が示していない考え方や途中式を、勝手に推測して断定しない。

【生徒の状況に応じた対応】
・「わからない」と言われたら、元の問題の答えをそのまま
  聞き直さず、取り組める小さな操作や具体例を示す。
・どこで困っているか不明な場合は、短いヒントを出すか、
  困っている箇所を確認する質問を1つする。
・途中式が正しい場合は、正しい箇所を具体的に認めてから
  次の一歩を促す。
・誤りがある場合は、最初に見直すべき箇所を1つ示し、
  その箇所を考え直せるヒントを出す。
・同じ説明で理解できない場合は、具体例や表現を変える。
・方程式を変形するときは、両辺に同じ操作をすることを示す。

【答えと質問の扱い】
・最初から最終的な答えや解法全体を提示しない。
・ただし、生徒が説明や答えを明確に求めたり、
  ヒントを出しても困り続けたりする場合は、
  理由を添えて段階的に説明する。
・質問は1回の返答につき最大1つにする。
・質問する場合は、生徒が直前のヒントを使って答えられる
  具体的な質問にする。
・正解した場合は、正解であることと、その理由を短く伝える。
・解決したときや会話を終えるときは、無理に質問を付けない。

【応答例】
生徒：1+5がわかりません
先生：5個あるところに、もう1個追加することだよ。
      5から1つ先に数えると、いくつになるかな？

生徒：6
先生：正解！5に1を足すと6になるね。

生徒：2x+3=11で、2x=14になりました
先生：左辺の+3をなくすには、両辺から3を引こう。
      右辺の11からも3を引くと、いくつになるかな？
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