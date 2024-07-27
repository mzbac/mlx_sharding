document.addEventListener("DOMContentLoaded", (event) => {
    const chatHistory = document.querySelector(".chat-history");
    const input = document.querySelector(".input-container textarea");
    const sendButton = document.querySelector(".input-container button");
    const clearButton = document.querySelector(".clear-button");
    const historyButton = document.querySelector(".history-button");
    const endpointInput = document.getElementById("endpoint");
    const stopWordInput = document.getElementById("stopWord");
    const maxTokensInput = document.getElementById("maxTokens");
    const modelInput = document.getElementById("model");
    const temperatureInput = document.getElementById("temperature");
    const topPInput = document.getElementById("topP");
    const apiKeyInput = document.getElementById("apiKey");
    const saveSettingsButton = document.getElementById("saveSettings");
    const settingsContainer = document.querySelector(".settings-container");
    const settingsToggle = document.getElementById("settings-toggle");
    const chatHistoryToggle = document.getElementById("chat-history-toggle");
    const sessionsContainer = document.querySelector(".sessions-container");
    const sessionListContainer = document.querySelector(".session-list");
  
    settingsToggle.addEventListener("change", () => {
      settingsContainer.classList.toggle("hidden");
    });
  
    chatHistoryToggle.addEventListener("click", () => {
      sessionsContainer.classList.toggle("hidden");
    });
  
    let messages = JSON.parse(localStorage.getItem("chatHistory")) || [];
    const settings = JSON.parse(localStorage.getItem("settings")) || {
      endpoint: "http://localhost:8080/v1/chat/completions",
      stopWord: "",
      maxTokens: 500,
      model: "default_model",
      temperature: 0.1,
      topP: 0.95,
      apiKey: "",
    };
  
    const throttle = (func, limit) => {
      let lastFunc;
      let lastRan;
      return function () {
        const context = this;
        const args = arguments;
        if (!lastRan) {
          func.apply(context, args);
          lastRan = Date.now();
        } else {
          clearTimeout(lastFunc);
          lastFunc = setTimeout(function () {
            if (Date.now() - lastRan >= limit) {
              func.apply(context, args);
              lastRan = Date.now();
            }
          }, limit - (Date.now() - lastRan));
        }
      };
    };
  
    function clearErrorMessages() {
      messages = messages.filter((msg) => msg.role !== "error");
      localStorage.setItem("chatHistory", JSON.stringify(messages));
      throttledRenderMessages();
    }
    const renderMessages = () => {
      if (!marked) {
        console.error("Marked library is not loaded correctly.");
        return;
      }
  
      chatHistory.innerHTML = "";
      messages.forEach((msg, index) => {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${msg.role}`;
        messageDiv.dataset.index = index;
  
        if (msg.role === "assistant") {
          messageDiv.innerHTML = marked.parse(msg.content);
          const regenerateButton = document.createElement("button");
          regenerateButton.className = "regenerate-button";
          regenerateButton.dataset.index = index;
          regenerateButton.textContent = "Regenerate";
          messageDiv.appendChild(regenerateButton);
        } else {
          const editableDiv = document.createElement("div");
          editableDiv.contentEditable = true;
          editableDiv.innerHTML = marked.parse(msg.content);
          editableDiv.className = "editable-user-input";
          editableDiv.dataset.originalContent = msg.content;
          messageDiv.appendChild(editableDiv);
        }
  
        chatHistory.appendChild(messageDiv);
      });
  
      chatHistory.scrollTop = chatHistory.scrollHeight;
    };
  
    const throttledRenderMessages = throttle(renderMessages, 100);
  
    chatHistory.addEventListener("click", async (event) => {
      if (event.target.classList.contains("regenerate-button")) {
        const index = parseInt(event.target.dataset.index);
        if (messages[index].role === "assistant") {
          await regenerateAssistantMessage(index);
        }
      }
    });
  
    chatHistory.addEventListener(
      "blur",
      async (event) => {
        if (event.target.classList.contains("editable-user-input")) {
          const index = parseInt(event.target.parentNode.dataset.index);
          const originalContent = event.target.dataset.originalContent;
          const currentContent = event.target.innerText;
  
          if (currentContent !== originalContent) {
            messages[index].content = currentContent;
            localStorage.setItem("chatHistory", JSON.stringify(messages));
            await regenerateFromUserEdit(index);
          }
        }
      },
      true
    );
  
    async function regenerateAssistantMessage(index) {
      clearErrorMessages();
      messages.splice(index, 1);
      const updatedMessages = messages.slice(0, index);
      await regenerateResponse(updatedMessages, index);
    }
  
    async function regenerateFromUserEdit(index) {
      clearErrorMessages();
      messages = messages.slice(0, index + 1);
      await regenerateResponse(messages, index + 1);
    }
  
    async function regenerateResponse(messageList, startIndex) {
      sendButton.disabled = true;
      sendButton.textContent = "Generating...";
  
      try {
        const response = await fetch(settings.endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${settings.apiKey}`,
          },
          body: JSON.stringify({
            model: settings.model,
            messages: messageList,
            stream: true,
            stop: settings.stopWord ? settings.stopWord : undefined,
            max_tokens: settings.maxTokens,
            temperature: settings.temperature,
            top_p: settings.topP,
          }),
        });
  
        if (!response.ok) {
          throw new Error(`Failed to generate response: ${response.statusText}`);
        }
        const reader = response.body.getReader();
        let currentResponse = "";
        let initial = true;
  
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = new TextDecoder().decode(value);
          if (chunk.trim()) {
            const lines = chunk.split("\n");
            for (let line of lines) {
              if (line.startsWith("data:")) {
                if (line === "data: [DONE]") break;
  
                const data = JSON.parse(line.substring(5));
                if (data.choices) {
                  data.choices.forEach((choice) => {
                    currentResponse += choice.delta.content ?? "";
                  });
                }
              }
            }
            if (initial) {
              messages.splice(startIndex, 0, {
                role: "assistant",
                content: currentResponse,
              });
              initial = false;
            } else {
              messages[startIndex].content = currentResponse;
            }
            throttledRenderMessages();
          }
        }
      } catch (error) {
        console.error("Error processing message:", error);
        messages.push({
          role: "error",
          content: "An error occurred while processing your message.",
        });
        throttledRenderMessages();
      } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Send";
        localStorage.setItem("chatHistory", JSON.stringify(messages));
      }
    }
  
    clearButton.addEventListener("click", () => {
      localStorage.removeItem("chatHistory");
      sessionListContainer.innerHTML = "";
      messages = [];
      throttledRenderMessages();
    });
  
    historyButton.addEventListener("click", () => {
      const now = new Date();
      const defaultSessionName = `Session ${now.getFullYear()}-${
        now.getMonth() + 1
      }-${now.getDate()} ${now.getHours()}:${now.getMinutes()}`;
      const sessionName = prompt(
        "Please enter a name for this session:",
        defaultSessionName
      );
      if (sessionName) {
        localStorage.setItem(sessionName, JSON.stringify(messages));
        addSessionToList(sessionName);
      }
      messages = [];
      throttledRenderMessages();
    });
  
    const addSessionToList = (sessionName) => {
      const sessionDiv = document.createElement("div");
      sessionDiv.className = "session";
      sessionDiv.textContent = sessionName;
      sessionDiv.onclick = function () {
        messages = JSON.parse(localStorage.getItem(sessionName));
        throttledRenderMessages();
      };
      sessionListContainer.appendChild(sessionDiv);
    };
  
    Object.keys(localStorage).forEach((key) => {
      if (key !== "settings" && key !== "chatHistory") {
        addSessionToList(key);
      }
    });
  
    saveSettingsButton.addEventListener("click", () => {
      settings.endpoint = endpointInput.value.trim();
      settings.stopWord = stopWordInput.value.trim();
      settings.maxTokens = parseInt(maxTokensInput.value);
      settings.model = modelInput.value.trim();
      settings.temperature = parseFloat(temperatureInput.value);
      settings.topP = parseFloat(topPInput.value);
      settings.apiKey = document.getElementById("apiKey").value.trim(); // Ensure API key is saved
      localStorage.setItem("settings", JSON.stringify(settings));
    });
  
    sendButton.addEventListener("click", async () => {
      clearErrorMessages();
      const userInput = input.value.trim();
      if (userInput === "") return;
  
      const userMessage = { role: "user", content: userInput };
      messages.push(userMessage);
      throttledRenderMessages();
      input.value = "";
  
      await regenerateResponse(messages, messages.length);
    });
  
    throttledRenderMessages();
  
    endpointInput.value = settings.endpoint;
    stopWordInput.value = settings.stopWord;
    maxTokensInput.value = settings.maxTokens;
    modelInput.value = settings.model;
    temperatureInput.value = settings.temperature;
    topPInput.value = settings.topP;
    apiKeyInput.value = settings.apiKey ?? "";
  });