import os
import getpass

# 使用するGPUとモデルの保存先を指定
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "GPU-490452e1-93a9-2524-849a-2ec1f1098bdf"
)
os.environ["HF_HOME"] = (
    f"/shared_workspace/{getpass.getuser()}/MathMate/huggingface"
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """
あなたは、中学生の数学の学習を支援する先生です。
以下のルールに従ってください。

・親しみやすく、わかりやすい日本語を使う。
・1回の返答は2〜4文程度にする。
・最終的な答えをすぐに教えず、次の一歩のヒントを出す。
・生徒の考え方や途中式を確認し、理解に合わせて説明する。
・誤りがあれば、責めずに、どこを見直すか具体的に伝える。
・一度に質問するのは1つだけにする。
・方程式では、等式を保つために両辺に同じ操作をすることを明確にする。
・生徒が正しい答えに到達した場合は、正解であることを伝えてよい。
""".strip()


def main():
    print("モデルを読み込み中...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("\nMathMate：数学で困っていることを入力してください。")
    print("「リセット」で新しい会話、「終了」で終了します。\n")

    while True:
        try:
            user_input = input("あなた > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMathMateを終了します。")
            break

        if user_input == "終了":
            print("MathMateを終了します。")
            break

        if user_input == "リセット":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            print("会話をリセットしました。\n")
            continue

        if not user_input:
            continue

        messages.append(
            {"role": "user", "content": user_input}
        )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to("cuda:0")

        print("MathMate：考え中...", flush=True)

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

        generated_tokens = outputs[
            0, inputs["input_ids"].shape[1]:
        ]
        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        print(f"先生AI > {answer}\n")

        messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()