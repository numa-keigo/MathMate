const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const sendButton = form.querySelector("button");

// このブラウザの会話履歴
let history = [];

function addMessage(text, role) {
  const message = document.createElement("div");

  message.classList.add("message", role);
  message.textContent = text;

  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  const text = input.value.trim();

  if (text === "" || sendButton.disabled) {
    return;
  }

  // Python側の入力上限に合わせる
  if (text.length > 2000) {
    alert("質問は2000文字以内で入力してください。");
    return;
  }

  addMessage(text, "user");
  input.value = "";

  sendButton.disabled = true;
  sendButton.textContent = "考え中…";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
        history: history,
      }),
    });

    if (!response.ok) {
      throw new Error(`通信エラー: ${response.status}`);
    }

    const data = await response.json();

    addMessage(data.reply, "assistant");

    // 成功したやり取りを履歴に保存する
    history.push(
      { role: "user", content: text },
      { role: "assistant", content: data.reply }
    );

    // 直近6往復だけを残す
    history = history.slice(-12);
  } catch (error) {
    console.error(error);

    addMessage(
      "送信できませんでした。この質問は会話履歴に保存されていません。もう一度入力してください。",
      "assistant"
    );

    // 再送しやすいように、入力欄へ戻す
    if (input.value === "") {
      input.value = text;
    }
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "送信";
    input.focus();
  }
});