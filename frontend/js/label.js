// Labelling tool: mark position changes in a video, download them as ground truth.
//
// A label applies from its timestamp until the next mark, which matches how BJJ actually looks:
// positions last seconds. Marking every frame would take an hour per two minutes of video.
//
// The video is opened with URL.createObjectURL and never uploaded anywhere.

import { POSITIONS, describe } from "./positions.js";

const $ = (id) => document.getElementById(id);

// Digit -> base position. Shift means "athlete 2" for the positions that distinguish sides.
const KEYS = [
  ["1", "standing"], ["2", "takedown"], ["3", "open_guard"], ["4", "closed_guard"],
  ["5", "half_guard"], ["6", "5050_guard"], ["7", "side_control"], ["8", "mount"],
  ["9", "back"], ["0", "turtle"],
];
const SPEEDS = [0.25, 0.5, 1, 1.5, 2];

const video = $("video");
const marks = [];          // [{ t, label }], kept sorted by time
let file = null;
let speedIndex = 2;

const symmetric = (base) => POSITIONS.find((p) => p.base === base)?.who === null;
const labelFor = (base, athlete) => (symmetric(base) ? base : `${base}${athlete}`);
const format = (t) => `${t.toFixed(1)} s`;

function title(label) {
  const d = describe(label);
  return d.athlete ? `${d.title} · ${d.athlete}` : d.title;
}

function colourFor(label) {
  const family = {
    standing: "#ece8e1", takedown: "#ece8e1",
    open_guard: "#4a9fe0", closed_guard: "#4a9fe0", half_guard: "#4a9fe0", "5050_guard": "#4a9fe0",
    side_control: "#d16ba5", mount: "#d16ba5",
    back: "#d9a441", turtle: "#5fbf8f",
  };
  return family[label.replace(/[12]$/, "")] ?? "#8a8f98";
}

function renderKeys() {
  $("keylist").replaceChildren(...KEYS.map(([key, base]) => {
    const li = document.createElement("li");
    const kbd = document.createElement("kbd");
    kbd.textContent = key;
    const name = document.createElement("span");
    name.textContent = POSITIONS.find((p) => p.base === base)?.name ?? base;
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = colourFor(base);
    li.append(kbd, name, swatch);
    return li;
  }));
}

function activeLabelAt(time) {
  let active = null;
  for (const mark of marks) {
    if (mark.t <= time + 1e-6) active = mark;
    else break;
  }
  return active;
}

function render() {
  $("count").textContent = String(marks.length);
  $("save").disabled = marks.length === 0;

  $("marks").replaceChildren(...marks.map((mark, index) => {
    const li = document.createElement("li");
    const time = document.createElement("button");
    time.type = "button";
    time.className = "marks__time mono";
    time.textContent = format(mark.t);
    time.addEventListener("click", () => { video.currentTime = mark.t; });
    const name = document.createElement("span");
    name.className = "marks__name";
    name.textContent = title(mark.label);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "marks__remove";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${title(mark.label)} at ${format(mark.t)}`);
    remove.addEventListener("click", () => { marks.splice(index, 1); render(); });
    li.append(time, name, remove);
    return li;
  }));

  // Timeline strip: one block per labelled stretch.
  const duration = video.duration || 1;
  $("strip").replaceChildren(...marks.map((mark, index) => {
    const end = index + 1 < marks.length ? marks[index + 1].t : duration;
    const block = document.createElement("button");
    block.type = "button";
    block.className = "tl__block";
    block.style.flexGrow = String(Math.max((end - mark.t) / duration, 0.002));
    block.style.background = colourFor(mark.label);
    block.title = `${title(mark.label)} — ${format(mark.t)}`;
    block.addEventListener("click", () => { video.currentTime = mark.t; });
    return block;
  }));

  const active = activeLabelAt(video.currentTime);
  $("current").textContent = active ? title(active.label) : "— no position yet —";
}

function addMark(base, athlete) {
  const label = labelFor(base, athlete);
  const t = Math.round(video.currentTime * 10) / 10;
  const existing = marks.findIndex((mark) => Math.abs(mark.t - t) < 0.05);
  if (existing >= 0) marks[existing].label = label;      // correcting a mark just set
  else marks.push({ t, label });
  marks.sort((a, b) => a.t - b.t);
  render();
}

function setSpeed(index) {
  speedIndex = Math.max(0, Math.min(SPEEDS.length - 1, index));
  video.playbackRate = SPEEDS[speedIndex];
  $("speed-label").textContent = `${SPEEDS[speedIndex].toFixed(2)}×`;
}

function download() {
  const payload = {
    video: file ? file.name : "unknown",
    duration_s: Math.round((video.duration || 0) * 10) / 10,
    created: new Date().toISOString(),
    marks: marks.map((mark) => ({ t: mark.t, label: mark.label })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${(file ? file.name : "labels").replace(/\.[^.]+$/, "")}.labels.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function load(chosen) {
  if (!chosen || !chosen.type.startsWith("video/")) return;
  file = chosen;
  video.src = URL.createObjectURL(chosen);
  $("dropzone").hidden = true;
  $("labeller").hidden = false;
  marks.length = 0;
  video.addEventListener("loadedmetadata", render, { once: true });
}

$("video-input").addEventListener("change", (event) => load(event.target.files[0]));
video.addEventListener("timeupdate", () => {
  $("clock").textContent = format(video.currentTime);
  const active = activeLabelAt(video.currentTime);
  $("current").textContent = active ? title(active.label) : "— no position yet —";
});
$("play").addEventListener("click", () => (video.paused ? video.play() : video.pause()));
video.addEventListener("play", () => { $("play").textContent = "Pause"; });
video.addEventListener("pause", () => { $("play").textContent = "Play"; });
$("back").addEventListener("click", () => { video.currentTime = Math.max(0, video.currentTime - 1); });
$("forward").addEventListener("click", () => { video.currentTime += 1; });
$("slower").addEventListener("click", () => setSpeed(speedIndex - 1));
$("faster").addEventListener("click", () => setSpeed(speedIndex + 1));
$("undo").addEventListener("click", () => { marks.pop(); render(); });
$("save").addEventListener("click", download);

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || $("labeller").hidden) return;
  const entry = KEYS.find(([key]) => key === event.key || key === event.code.replace("Digit", ""));
  if (entry) {
    event.preventDefault();
    addMark(entry[1], event.shiftKey ? 2 : 1);
    return;
  }
  switch (event.key) {
    case " ": event.preventDefault(); video.paused ? video.play() : video.pause(); break;
    case "Backspace": event.preventDefault(); marks.pop(); render(); break;
    case "ArrowLeft": video.currentTime = Math.max(0, video.currentTime - (event.shiftKey ? 0.1 : 1)); break;
    case "ArrowRight": video.currentTime += event.shiftKey ? 0.1 : 1; break;
    case "-": setSpeed(speedIndex - 1); break;
    case "+": setSpeed(speedIndex + 1); break;
  }
});

renderKeys();
setSpeed(speedIndex);
