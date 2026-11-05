from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import load_dotenv

# torchやtransformersを読み込む前に環境変数を設定する
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


STATIC_DIR = BASE_DIR / "static"
MODEL_ID = "Qwen/Qwen3-14B"

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
        print("チャットに送信", flush=True)
        # システムプロンプトを追加する
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # ブラウザから受け取った会話履歴を追加する
        messages.extend(
            {"role": item.role, "content": item.content}
            for item in request.history
        )

        # 今回の質問を追加する
        messages.append(
            {"role": "user", "content": request.message}
        )

        with generation_lock:
            # thinkingモードで、会話をモデル用の形式に整える
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            # トークンIDに変換してGPUへ送る
            inputs = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda:0")

            # 回答を生成する
            # max_new_tokensには思考部分も含まれる

            print("回答の生成を開始", flush=True)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                )
            print("回答の生成が完了", flush=True)

            # 入力を除き、生成された部分だけを取り出す
            generated_tokens = outputs[
                0, inputs["input_ids"].shape[1]:
            ].tolist()

            # 思考の終了位置を探す
            think_end_id = tokenizer.convert_tokens_to_ids(
                "</think>"
            )

            end_positions = [
                i
                for i, token_id in enumerate(generated_tokens)
                if token_id == think_end_id
            ]

            if end_positions:
                # 最後の </think> より後だけを取り出す
                answer_tokens = generated_tokens[
                    end_positions[-1] + 1:
                ]

                answer = tokenizer.decode(
                    answer_tokens,
                    skip_special_tokens=True,
                ).strip()

                if not answer:
                    answer = (
                        "回答を完成できませんでした。"
                        "もう一度送信してください。"
                    )
            else:
                # 思考途中で生成が終了した場合は表示しない
                answer = (
                    "回答を完成できませんでした。"
                    "もう一度送信してください。"
                )

        # 最終回答だけをブラウザへ返す
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
    # モデルは起動時に1回だけ読み込む
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