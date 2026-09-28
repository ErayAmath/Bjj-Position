// Round analysis view: timeline of positions, time per position, and the tracked skeletons.
// Data comes from scripts/analyze_round.py -> scripts/export_round_to_frontend.py.

import { SkeletonStage } from "./skeleton.js";
import { describe } from "./positions.js";
import { attachTooltip } from "./charts.js";

// Position families. Colours validated for colour-vision deficiency (worst pair dE 8.6);
// every segment is also labelled in text, so colour never carries the meaning alone.
const FAMILY = {
  standing: "neutral", takedown: "neutral",
  open_guard: "guard", closed_guard: "guard", half_guard: "guard", "5050_guard": "guard",
  side_control: "control", mount: "control",
  back: "back",
  turtle: "turtle",
};
const FAMILY_COLOR = {
  neutral: "#ece8e1",
  guard: "#4a9fe0",
  control: "#d16ba5",
  back: "#d9a441",
  turtle: "#5fbf8f",
};
const FAMILY_LABEL = {
  neutral: "Standing / takedown",
  guard: "Guard",
  control: "Side control / mount",
  back: "Back",
  turtle: "Turtle",
};

const $ = (id) => document.getElementById(id);
const fmtSeconds = (s) => (s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")} min` : `${s.toFixed(1)} s`);

function colorOf(className) {
  const base = className.replace(/[12]$/, "");
  return FAMILY_COLOR[FAMILY[base]] ?? "#8a8f98";
}

function renderTimeline(container, data, onSeek, bindTooltip) {
  container.replaceChildren();
  const total = data.duration_s || 1;
  for (const segment of data.timeline) {
    const name = data.classes[segment.label];
    const share = (segment.end_time - segment.start_time) / total;
    const block = document.createElement("button");
    block.type = "button";
    block.className = "tl__block";
    block.style.flexGrow = String(Math.max(share, 0.001));
    block.style.background = colorOf(name);
    block.setAttribute("aria-label", `${describe(name).title}, ${fmtSeconds(segment.seconds)}`);
    // Direct label when the block is wide enough; the tooltip always has the full text.
    if (share > 0.07) {
      const label = document.createElement("span");
      label.textContent = describe(name).title;
      block.append(label);
    }
    bindTooltip(block, () =>
      `<b>${describe(name).title}${describe(name).athlete ? ` · ${describe(name).athlete}` : ""}</b><br>` +
      `${fmtSeconds(segment.seconds)} <span class="muted">(${(share * 100).toFixed(1)}%)</span><br>` +
      `<span class="muted">${segment.start_time.toFixed(1)}s – ${segment.end_time.toFixed(1)}s</span>`);
    block.addEventListener("click", () => onSeek(segment.start_time));
    container.append(block);
  }
}

function renderTotals(container, data, bindTooltip) {
  container.replaceChildren();
  const entries = Object.entries(data.time_per_position);
  const max = Math.max(...entries.map(([, s]) => s), 1);
  const total = data.duration_s || 1;
  for (const [name, seconds] of entries) {
    const row = document.createElement("div");
    row.className = "totals__row";
    const label = document.createElement("span");
    label.className = "totals__name";
    const d = describe(name);
    label.textContent = d.athlete ? `${d.title} · ${d.athlete}` : d.title;
    const track = document.createElement("div");
    track.className = "totals__track";
    const bar = document.createElement("div");
    bar.className = "totals__bar";
    bar.style.width = `${(seconds / max) * 100}%`;
    bar.style.background = colorOf(name);
    track.append(bar);
    const value = document.createElement("span");
    value.className = "totals__value mono";
    value.textContent = fmtSeconds(seconds);
    bindTooltip(track, () => `<b>${d.title}</b><br>${fmtSeconds(seconds)} <span class="muted">(${((seconds / total) * 100).toFixed(1)}%)</span>`);
    row.append(label, track, value);
    container.append(row);
  }
}

function renderLegend(container) {
  container.replaceChildren(...Object.entries(FAMILY_LABEL).map(([key, text]) => {
    const li = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = FAMILY_COLOR[key];
    li.append(swatch, document.createTextNode(text));
    return li;
  }));
}

export async function setupRound(bindTooltip) {
  // A locally exported analysis wins over the committed demo. On the public site the private
  // file does not exist (it is git-ignored), so visitors always see the demo round.
  let data;
  for (const path of ["data/private/round.json", "data/round.json"]) {
    try {
      const res = await fetch(path);
      if (res.ok) { data = await res.json(); break; }
    } catch { /* keep trying */ }
  }
  if (!data) return;                           // nothing exported yet: section stays hidden

  const section = $("round");
  section.hidden = false;

  $("round-source").textContent = data.source;
  if (data.private) {
    $("round-privacy").hidden = false;
  }
  $("round-duration").textContent = fmtSeconds(data.duration_s);
  $("round-detection").textContent = `${Math.round(data.both_after_gap_fill * 100)}%`;
  $("round-changes").textContent = `${data.changes_raw} → ${data.changes_smoothed}`;
  if (data.accuracy != null) {
    $("round-accuracy").textContent = `${Math.round(data.accuracy * 100)}%`;
    $("round-accuracy-row").hidden = false;
  }

  renderLegend($("round-legend"));
  renderTotals($("round-totals"), data, bindTooltip);

  // Skeleton playback, driven by the frame index.
  const frames = (data.poses || []).map((frame) => frame.map((person) => person));
  const stage = new SkeletonStage($("round-canvas"), data.skeleton, {
    fps: data.fps,
    captionSpace: 0.12,
    onFrame: (i) => {
      const name = data.classes[data.labels[i]];
      const d = describe(name);
      $("round-label").textContent = d.athlete ? `${d.title} · ${d.athlete}` : d.title;
      $("round-time").textContent = `${(data.times[i] - data.times[0]).toFixed(1)}s`;
      $("round-scrub").value = String(i);
      marker.style.left = `${(i / Math.max(frames.length - 1, 1)) * 100}%`;
    },
  });
  stage.onLoopEnd = () => stage.setFrame(0);

  const marker = $("round-marker");
  const scrub = $("round-scrub");
  scrub.max = String(Math.max(frames.length - 1, 0));

  renderTimeline($("round-timeline"), data, (time) => {
    const index = data.times.findIndex((t) => t - data.times[0] >= time - data.times[0]);
    stage.setFrame(Math.max(index, 0));
  }, bindTooltip);

  if (frames.length) {
    stage.setSequence({ frames });
    const toggle = $("round-play");
    const setPlaying = (on) => {
      on ? stage.play() : stage.pause();
      toggle.textContent = on ? "Pause" : "Play";
      toggle.setAttribute("aria-pressed", String(on));
    };
    toggle.addEventListener("click", () => setPlaying(!stage.playing));
    scrub.addEventListener("input", () => { setPlaying(false); stage.setFrame(Number(scrub.value)); });
    setPlaying(!window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  } else {
    $("round-player").hidden = true;
  }
}
