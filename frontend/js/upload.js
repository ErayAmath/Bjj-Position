// Video upload for sparring analysis.
//
// Privacy contract: the file is only read locally (object URL for preview) until the user
// clicks "Start analysis", and even then it is only sent to the backend on this machine
// (same origin, /api). No third-party request ever carries the video.
//
// Backend contract (not implemented yet — see README "Backend plan"):
//   GET  /api/health           -> 200 {"status": "ok"}
//   POST /api/analyses         multipart form field "video" -> 202 {"id": "..."}

const $ = (id) => document.getElementById(id);

const input = $("video-input");
const dropzone = $("dropzone");
const preview = $("preview");
const video = $("preview-video");
const meta = $("filemeta");
const startBtn = $("start-analysis");
const status = $("analysis-status");

let current = null; // { file, url }

function setStatus(text, tone = "") {
  status.textContent = text;
  status.dataset.tone = tone;
}

function formatBytes(n) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function renderMeta(rows) {
  meta.replaceChildren(...rows.map(([label, value]) => {
    const div = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    div.append(dt, dd);
    return div;
  }));
}

function clear() {
  if (current) URL.revokeObjectURL(current.url); // free the memory held by the preview
  current = null;
  video.removeAttribute("src");
  video.load();
  preview.hidden = true;
  dropzone.hidden = false;
  input.value = "";
  startBtn.disabled = true;
  setStatus("No video selected.");
}

function load(file) {
  if (!file) return;
  if (!file.type.startsWith("video/")) {
    setStatus(`"${file.name}" is not a video file.`, "warn");
    return;
  }
  if (current) URL.revokeObjectURL(current.url);
  current = { file, url: URL.createObjectURL(file) };

  video.src = current.url;
  dropzone.hidden = true;
  preview.hidden = false;
  renderMeta([["File", file.name], ["Size", formatBytes(file.size)], ["Length", "…"]]);
  setStatus("Reading video…");

  video.onloadedmetadata = () => {
    renderMeta([
      ["File", file.name],
      ["Size", formatBytes(file.size)],
      ["Length", formatDuration(video.duration)],
      ["Resolution", `${video.videoWidth} × ${video.videoHeight}`],
    ]);
    startBtn.disabled = false;
    setStatus("Ready. The video has not left this device.", "ok");
  };
  video.onerror = () => {
    startBtn.disabled = true;
    setStatus("This browser cannot decode the video. Try MP4 (H.264).", "warn");
  };
}

async function backendAvailable() {
  try {
    const res = await fetch("/api/health", { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function start() {
  if (!current) return;
  startBtn.disabled = true;
  setStatus("Looking for the local analysis backend…");

  if (!(await backendAvailable())) {
    setStatus("No analysis backend running on this machine yet (planned for milestone 3). Your video stayed local.", "warn");
    startBtn.disabled = false;
    return;
  }

  const form = new FormData();
  form.append("video", current.file);
  setStatus("Uploading to the local backend…");
  try {
    const res = await fetch("/api/analyses", { method: "POST", body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { id } = await res.json();
    setStatus(`Analysis queued (${id}).`, "ok");
  } catch (err) {
    setStatus(`Upload failed: ${err.message}`, "warn");
    startBtn.disabled = false;
  }
}

input.addEventListener("change", () => load(input.files[0]));
$("clear-video").addEventListener("click", clear);
startBtn.addEventListener("click", start);

// Drag & drop onto the dropzone.
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("is-over"); });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, () => dropzone.classList.remove("is-over"));
}
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  load(e.dataTransfer.files[0]);
});
// A file dropped next to the zone would otherwise make the browser navigate to it.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());
