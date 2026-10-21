const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const sendButton = form.querySelector("button");

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

  addMessage(text, "user");
  input.value = "";

  sendButton.disabled = true;
  sendButton.textContent = "送信中…";

  try {
    // PythonのAPIへ質問を送る
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`通信エラー: ${response.status}`);
    }

    // Pythonから返ってきたJSONを取り出す
    const data = await response.json();

    addMessage(data.reply, "assistant");
  } catch (error) {
    console.error(error);
    addMessage(
      "送信できませんでした。サーバーの状態を確認して、もう一度入力してください。",
      "assistant"
    );
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "送信";
    input.focus();
  }
});