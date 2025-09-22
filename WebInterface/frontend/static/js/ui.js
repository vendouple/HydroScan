const form = document.getElementById("analyze-form");
const statusEl = document.getElementById("status");

form?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  const files = document.getElementById("files");
  const description = document.getElementById("description");
  const debug = document.getElementById("debug");

  for (const f of files.files) fd.append("files", f);
  if (description.value) fd.append("description", description.value);
  fd.append("debug", debug.checked ? "true" : "false");

  statusEl.textContent = "Uploading and analyzing...";
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const json = await res.json();
    statusEl.textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    statusEl.textContent = "Error: " + err;
  }
});
