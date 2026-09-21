// Draws the two athletes of one sequence onto a canvas.
// Coordinates are raw image pixels; each sequence is fitted into the canvas once
// (fixed bounding box over all its frames) so the camera does not "breathe".

const COLORS = () => {
  const css = getComputedStyle(document.documentElement);
  return {
    a1: css.getPropertyValue("--a1").trim(),
    a2: css.getPropertyValue("--a2").trim(),
    grid: css.getPropertyValue("--line").trim(),
  };
};

function boundsOf(frames) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const athletes of frames) {
    for (const kps of athletes) {
      if (!kps) continue;
      for (const p of kps) {
        if (!p) continue;
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
    }
  }
  return { minX, minY, maxX, maxY };
}

export class SkeletonStage {
  constructor(canvas, skeleton, { fps = 12, padding = 0.14, captionSpace = 0.1, onFrame } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.skeleton = skeleton;
    this.fps = fps;
    this.padding = padding;
    this.captionSpace = captionSpace; // bottom share of the canvas kept free for the caption
    this.onFrame = onFrame;
    this.sequence = null;
    this.frame = 0;
    this.playing = false;
    this.colors = COLORS();
    this._last = 0;
    this._raf = null;

    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
  }

  setSequence(sequence) {
    this.sequence = sequence;
    this.frame = 0;
    this._fit();
    this.draw();
  }

  setFrame(i) {
    if (!this.sequence) return;
    this.frame = (i + this.sequence.frames.length) % this.sequence.frames.length;
    this.draw();
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    const tick = (t) => {
      if (!this.playing) return;
      if (t - this._last >= 1000 / this.fps) {
        this._last = t;
        const next = this.frame + 1;
        if (next >= this.sequence.frames.length && this.onLoopEnd) this.onLoopEnd();
        else this.setFrame(next);
      }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  pause() {
    this.playing = false;
    cancelAnimationFrame(this._raf);
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = width;
    this.h = height;
    this._fit();
    this.draw();
  }

  _fit() {
    if (!this.sequence || !this.w) return;
    const b = boundsOf(this.sequence.frames);
    const availW = this.w * (1 - 2 * this.padding);
    const availH = this.h * (1 - 2 * this.padding - this.captionSpace);
    const scale = Math.min(availW / (b.maxX - b.minX || 1), availH / (b.maxY - b.minY || 1));
    this.scale = scale;
    this.offX = (this.w - (b.maxX - b.minX) * scale) / 2 - b.minX * scale;
    this.offY = (this.h * (1 - this.captionSpace) - (b.maxY - b.minY) * scale) / 2 - b.minY * scale;
  }

  draw() {
    const { ctx, w, h } = this;
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    this._drawGrid();
    if (!this.sequence) return;

    const frames = this.sequence.frames;
    // Motion trail: two faint previous poses, then the current one.
    for (const [lag, alpha] of [[4, 0.08], [2, 0.16], [0, 1]]) {
      const i = this.frame - lag;
      if (i < 0) continue;
      frames[i].forEach((kps, a) => {
        if (kps) this._drawAthlete(kps, a === 0 ? this.colors.a1 : this.colors.a2, alpha, lag === 0);
      });
    }
    if (this.onFrame) this.onFrame(this.frame, frames.length);
  }

  _drawGrid() {
    // Sparse mat grid: reads as "floor", stays out of the way.
    const { ctx, w, h } = this;
    const step = Math.max(40, Math.round(w / 12));
    ctx.save();
    ctx.strokeStyle = this.colors.grid;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = step; x < w; x += step) { ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); }
    for (let y = step; y < h; y += step) { ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); }
    ctx.stroke();
    ctx.restore();
  }

  _drawAthlete(kps, color, alpha, withJoints) {
    const { ctx } = this;
    const P = (p) => [p[0] * this.scale + this.offX, p[1] * this.scale + this.offY];
    const lw = Math.max(2, Math.min(4, this.w / 180));

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.beginPath();
    for (const [i, j] of this.skeleton) {
      if (i <= 4 && j <= 4) continue; // face edges are replaced by a head ring
      if (!kps[i] || !kps[j]) continue;
      const [x1, y1] = P(kps[i]);
      const [x2, y2] = P(kps[j]);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // Head: ring around the face keypoints, radius from shoulder width.
    const face = kps.slice(0, 5).filter(Boolean);
    if (face.length && kps[5] && kps[6]) {
      const cx = face.reduce((s, p) => s + p[0], 0) / face.length;
      const cy = face.reduce((s, p) => s + p[1], 0) / face.length;
      const shoulder = Math.hypot(kps[5][0] - kps[6][0], kps[5][1] - kps[6][1]);
      const r = Math.max(4, Math.min(shoulder * 0.32, 60) * this.scale);
      const [hx, hy] = P([cx, cy]);
      ctx.beginPath();
      ctx.arc(hx, hy, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (withJoints) {
      for (let k = 5; k < kps.length; k++) {
        if (!kps[k]) continue;
        const [x, y] = P(kps[k]);
        ctx.beginPath();
        ctx.arc(x, y, lw * 0.9, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }
}
