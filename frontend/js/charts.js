// Plain-DOM charts: grouped horizontal bars and a segment x position heatmap.
// Both share one tooltip element.

import { POSITIONS } from "./positions.js";

const fmt = new Intl.NumberFormat("en-US");
const pct = (x) => `${(x * 100).toFixed(1)}%`;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

export function attachTooltip(tooltip) {
  const show = (html, evt) => {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    move(evt);
  };
  const move = (evt) => {
    const pad = 14;
    const { width, height } = tooltip.getBoundingClientRect();
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + width > window.innerWidth - 8) x = evt.clientX - width - pad;
    if (y + height > window.innerHeight - 8) y = evt.clientY - height - pad;
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
  };
  const hide = () => { tooltip.hidden = true; };
  return (node, html) => {
    node.addEventListener("pointerenter", (e) => show(html(), e));
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerleave", hide);
  };
}

function byBase(classes) {
  // [{pos, a1, a2, sym}] in POSITIONS order, counts from the overview.
  return POSITIONS.map((pos) => {
    const find = (name) => classes.find((c) => c.name === name)?.count ?? 0;
    const sym = classes.find((c) => c.name === pos.base);
    return sym
      ? { pos, sym: sym.count, total: sym.count }
      : { pos, a1: find(`${pos.base}1`), a2: find(`${pos.base}2`), get total() { return this.a1 + this.a2; } };
  });
}

export function renderBars(container, overview, bindTooltip) {
  const rows = byBase(overview.classes);
  const max = Math.max(...overview.classes.map((c) => c.count));
  const N = overview.num_frames;

  for (const row of rows) {
    const wrap = el("div", "bars__row");
    wrap.setAttribute("role", "row");
    const name = el("div", "bars__name", row.pos.name);
    name.setAttribute("role", "rowheader");
    const track = el("div", "bars__track");
    track.setAttribute("role", "cell");

    const series = row.sym != null
      ? [["sym", row.sym, "Symmetric"]]
      : [["a1", row.a1, "Athlete 1"], ["a2", row.a2, "Athlete 2"]];
    for (const [key, value, label] of series) {
      const bar = el("div", `bar bar--${key}`);
      bar.style.width = `${(value / max) * 100}%`;
      bar.setAttribute("aria-label", `${label}: ${fmt.format(value)} frames`);
      track.append(bar);
    }
    bindTooltip(track, () =>
      `<b>${row.pos.name}</b><br>` +
      series.map(([, v, l]) => `${l}: ${fmt.format(v)} <span class="muted">(${pct(v / N)})</span>`).join("<br>"));

    const total = el("div", "bars__total mono", fmt.format(row.total));
    total.setAttribute("role", "cell");
    wrap.append(name, track, total);
    container.append(wrap);
  }
}

export function renderHeatmap(container, overview, bindTooltip) {
  const rows = byBase(overview.classes);
  const classIndex = Object.fromEntries(overview.classes.map((c, i) => [c.name, i]));
  container.style.setProperty("--cols", rows.length);

  container.append(el("div", "heatmap__col", "Seg"));
  for (const r of rows) container.append(el("div", "heatmap__col", r.pos.name));

  // Share of each segment's frames per base position (1 and 2 summed).
  const shares = overview.segments.map((seg) =>
    rows.map((r) => {
      const names = r.sym != null ? [r.pos.base] : [`${r.pos.base}1`, `${r.pos.base}2`];
      return names.reduce((s, n) => s + seg.counts[classIndex[n]], 0) / seg.frames;
    }));
  const maxShare = Math.max(...shares.flat());

  overview.segments.forEach((seg, si) => {
    const newSequence = si === 0 || seg.sequence !== overview.segments[si - 1].sequence;
    if (newSequence) container.append(el("div", "heatmap__group", `Sequence ${seg.sequence + 1}`));
    container.append(el("div", "heatmap__row", String(seg.id).padStart(2, "0")));
    rows.forEach((r, ri) => {
      const share = shares[si][ri];
      const cell = el("div", "heatmap__cell");
      cell.setAttribute("role", "cell");
      cell.setAttribute("aria-label", `Segment ${seg.id}, ${r.pos.name}: ${pct(share)}`);
      if (share > 0) {
        // Sequential single hue: surface -> belt red, sqrt to lift small shares.
        const t = Math.sqrt(share / maxShare);
        cell.style.background = `color-mix(in oklab, var(--red) ${Math.round(8 + t * 92)}%, var(--surface-2))`;
      }
      bindTooltip(cell, () =>
        `<b>Segment ${String(seg.id).padStart(2, "0")} · ${r.pos.name}</b>` +
        ` <span class="muted">sequence ${seg.sequence + 1}</span><br>` +
        `${pct(share)} of ${fmt.format(seg.frames)} frames`);
      container.append(cell);
    });
  });
}
