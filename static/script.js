const analyzeBtn = document.getElementById("analyzeBtn");
const detectBtn = document.getElementById("detectBtn");
const inputText = document.getElementById("inputText");
const resultDiv = document.getElementById("result");
const loadingDiv = document.getElementById("loading");

async function sendRequest(endpoint, payload) {
  resultDiv.classList.add("hidden");
  loadingDiv.classList.remove("hidden");
  loadingDiv.textContent = "⏳ Analyzing... Please wait.";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    loadingDiv.classList.add("hidden");
    resultDiv.classList.remove("hidden");

    resultDiv.innerHTML = `
      <strong>🔍 Result:</strong><br>
      <pre>${JSON.stringify(data, null, 2)}</pre>
    `;
  } catch (err) {
    loadingDiv.classList.add("hidden");
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `<span style="color:#f87171;">❌ Error:</span> ${err.message}`;
  }
}

analyzeBtn.addEventListener("click", () => {
  const url = inputText.value.trim();
  if (!url) return alert("Please enter a URL.");
  sendRequest("/api/analyze", { url });
});

detectBtn.addEventListener("click", () => {
  const text = inputText.value.trim();
  if (!text) return alert("Please enter email text.");
  sendRequest("/api/detect", { text });
});
