import { SkeletonStage } from "./skeleton.js";
import { POSITIONS, describe } from "./positions.js";
import { attachTooltip, renderBars, renderHeatmap } from "./charts.js";
import { setupRound } from "./round.js";

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

function metaText(seq) {
  return `SEG ${String(seq.segment).padStart(2, "0")} · F ${seq.frame_start}–${seq.frame_end}`;
}

function labelText(name) {
  const d = describe(name);
  return d.athlete ? `${d.title} · ${d.athlete}` : d.title;
}

function renderFacts(overview) {
  $("fact-frames").textContent = fmt.format(overview.num_frames);
  $("fact-classes").textContent = overview.num_classes;
  $("fact-sequences").textContent = overview.num_sequences;
  $("fact-sequences-label").textContent = `sparring sequences · ${overview.num_segments} camera views`;
  $("fact-missing").textContent = `${Math.round(overview.missing_athlete_share * 100)}%`;
}

function setupHero(data) {
  // Cycles through a fixed tour of positions, one full clip each.
  const tour = ["standing", "takedown1", "closed_guard2", "half_guard1", "side_control1", "mount1", "back2", "turtle2"]
    .map((n) => data.sequences.find((s) => s.name === n))
    .filter(Boolean);
  let i = 0;
  const stage = new SkeletonStage($("hero-canvas"), data.skeleton, { fps: 12 });
  const show = () => {
    const seq = tour[i % tour.length];
    stage.setSequence(seq);
    $("hero-label").textContent = labelText(seq.name);
    $("hero-meta").textContent = metaText(seq);
  };
  stage.onLoopEnd = () => { i += 1; show(); };
  show();
  if (!reducedMotion) stage.play();
}

function setupExplorer(data) {
  const list = $("position-list");
  const scrub = $("scrub");
  const toggle = $("play-toggle");
  const total = (n) => String(n).padStart(2, "0");

  const stage = new SkeletonStage($("explorer-canvas"), data.skeleton, {
    fps: 12,
    onFrame: (f, n) => {
      scrub.max = n - 1;
      scrub.value = f;
      $("frame-counter").textContent = `${total(f + 1)} / ${n}`;
    },
  });
  stage.onLoopEnd = () => stage.setFrame(0);

  const select = (name) => {
    const seq = data.sequences.find((s) => s.name === name);
    if (!seq) return;
    stage.setSequence(seq);
    $("explorer-label").textContent = labelText(name);
    $("explorer-meta").textContent = metaText(seq);
    $("explorer-desc").textContent = describe(name).text;
    for (const chip of list.querySelectorAll(".chip")) {
      const active = chip.dataset.name === name;
      chip.setAttribute("aria-pressed", active);
      chip.closest(".position-row").classList.toggle("is-active", active);
    }
  };

  for (const pos of POSITIONS) {
    const li = document.createElement("li");
    li.className = "position-row";
    const label = document.createElement("span");
    label.className = "position-row__name";
    label.textContent = pos.name;
    li.append(label);

    const variants = pos.who === null ? [[pos.base, "·"]] : [[`${pos.base}1`, "1"], [`${pos.base}2`, "2"]];
    for (const [name, text] of variants) {
      if (!data.sequences.some((s) => s.name === name)) continue;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.dataset.name = name;
      if (text !== "·") chip.dataset.athlete = text;
      chip.textContent = text === "·" ? "Both" : `A${text}`;
      chip.setAttribute("aria-label", `${pos.name}${text === "·" ? "" : `, athlete ${text}`}`);
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", () => select(name));
      li.append(chip);
    }
    list.append(li);
  }

  const setPlaying = (on) => {
    on ? stage.play() : stage.pause();
    toggle.textContent = on ? "Pause" : "Play";
    toggle.setAttribute("aria-pressed", on);
  };
  toggle.addEventListener("click", () => setPlaying(!stage.playing));
  scrub.addEventListener("input", () => { setPlaying(false); stage.setFrame(Number(scrub.value)); });

  select("mount1");
  setPlaying(!reducedMotion);
}

async function main() {
  try {
    const [overview, sequences] = await Promise.all([
      loadJSON("data/overview.json"),
      loadJSON("data/sequences.json"),
    ]);
    renderFacts(overview);
    setupHero(sequences);
    setupExplorer(sequences);
    const bind = attachTooltip($("tooltip"));
    await setupRound(bind);
    renderBars($("bars"), overview, bind);
    renderHeatmap($("heatmap"), overview, bind);
  } catch (err) {
    console.error(err);
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<p class="callout" style="margin-top:24px">Could not load data (${err.message}).
       Run <code>python scripts/export_frontend_data.py</code> and serve this folder over HTTP.</p>`);
  }
}

main();
