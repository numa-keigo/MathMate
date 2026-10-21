// HTMLから、操作する要素を取得する
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");

// 会話欄にメッセージを追加する関数
function addMessage(text, role) {
  const message = document.createElement("div");

  message.classList.add("message", role);
  message.textContent = text;

  chatMessages.appendChild(message);

  // 最新のメッセージが見える位置までスクロールする
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 送信ボタンが押されたときの処理
form.addEventListener("submit", function (event) {
  // フォーム送信によるページの再読み込みを止める
  event.preventDefault();

  const text = input.value.trim();

  // 空白だけの場合は送信しない
  if (text === "") {
    return;
  }

  addMessage(text, "user");

  input.value = "";

  addMessage(
    "入力を受け取りました。次に、この画面を数学の先生AIにつなぎます！",
    "assistant"
  );

  input.focus();
});