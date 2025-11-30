// Hard-coded project ID for now — must exist in database
const PROJECT_ID = 1;

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("status");

function setStatus(text) {
    statusEl.textContent = text || "";
}

function appendMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.classList.add("message", role);

    const roleLabel = document.createElement("div");
    roleLabel.classList.add("role");
    roleLabel.textContent = role === "user" ? "You" : "Orb";

    const content = document.createElement("div");
    content.textContent = text;

    bubble.appendChild(roleLabel);
    bubble.appendChild(content);

    chatContainer.appendChild(bubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage("user", text);
    userInput.value = "";
    userInput.focus();

    sendBtn.disabled = true;
    setStatus("Thinking…");

    try {
        const resp = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                project_id: PROJECT_ID,
                message: text
            }),
        });

        if (!resp.ok) {
            const errorData = await resp.json().catch(() => ({}));
            throw new Error(`HTTP ${resp.status} – ${JSON.stringify(errorData)}`);
        }

        const data = await resp.json();
        appendMessage("assistant", data.reply || "[No reply text]");
        setStatus("");
    } catch (err) {
        console.error(err);
        appendMessage("assistant", `Backend error: ${err.message}`);
        setStatus("Error");
    } finally {
        sendBtn.disabled = false;
    }
}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

window.addEventListener("load", () => {
    userInput.focus();
});