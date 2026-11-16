/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * ASCII shell for single-screen interactive guides (overrides/tuning-lab.html).
 *
 * Two parts:
 *
 *   1. createAsciiField -- draws the Gemma mark (docs/assets/icon.png) as live ASCII
 *      art on a canvas behind the page: the circle, its inscribed-square grid, the
 *      axes and diagonals as tangent-aware line glyphs, and the four-arc star as a
 *      shimmering density band. A sparse "token fog" of stage-specific vocabulary
 *      drifts behind it. Everything is a distance field evaluated per character
 *      cell, so the mark can move and rescale between the overview and the stages.
 *
 *   2. createShell -- the stage rail, panels, hash routing, keyboard shortcuts and
 *      the docked reading terminal, which shows one chapter "file" at a time out of
 *      the rendered page body.
 */
(() => {
  "use strict";

  const SHELL = "[data-llm-shell]";
  const STORAGE_KEY = "gemma.llm-shell.terminal";
  const FPS = 24;
  const TAU = Math.PI * 2;
  const STAR_RAMP = " .:+*%#@";
  const FONT_FALLBACK = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

  /* Vocabulary for the token fog, per stage. Kept to glyphs monospace fonts carry. */
  const FOG_WORDS = {
    overview: [
      "Gemma 4", "262,144 tokens", "E2B", "E4B", "12B", "26B A4B", "31B", "256K context",
      "softmax(QK^T/sqrt(d))", "p-RoPE", "K = V", "PLE", "MoE 8/128", "GeGLU", "RMSNorm",
      "top_k=64", "top_p=0.95", "T=1.0", "next token ->", "d_model=1536", "140+ languages", "<|turn>",
    ],
    tokenization: [
      "▁the #506", "▁cat #5866", "▁Gemma #147224", "▁is #563", "<|turn> #105", "<turn|> #106",
      "<|think|> #98", "<0xEB> #473", "<0x98> #390", "<0xA0> #398", "▁▁▁▁ #140", "\\n #107",
      "def #2063", "return #2060", "<unused0> #6", "<unk> #3", "<bos> #2", "<eos> #1", "4 #236812",
      "0 #236771", "BPE", "SentencePiece", "262,144", "▁multimodal #151359", "얌 #249170",
    ],
    embeddings: [
      "[-0.0049, -0.0491, ...]", "d_model = 1,536", "cos(king, queen) = 0.58", "cos(kitten, puppy) = 0.54",
      "cos(cat, kitten) = 0.43", "king - man + woman ~ queen", "262,144 x 1,536", "PLE 262,144 x 35 x 256",
      "2.35B params", "per_layer_input_gate", "GeLU gate", "256 -> 1,536", "<|image|> #258880",
      "<|audio|> #258881", "48x48 patch", "40 ms frame", "flash storage", "a.b / |a||b|", "#9615",
    ],
    attention: [
      "Q K V", "QK^T / sqrt(d_k) + M", "softmax", "causal mask", "5 : 1 local : global", "4 : 1 on E2B",
      "sliding window 512", "sliding window 1024", "K = V global", "p-RoPE", "RoPE", "RMSNorm", "x + Layer(x)",
      "GeGLU FFN", "128 experts", "8 routed + 1 shared", "3.8B active / 25.2B", "L1 L2 L3 L4 G5",
      "it -> trophy", "it -> suitcase", "KV cache", "256K",
    ],
    sampling: [
      "lm_head", "262,144 logits", "30 * tanh(z/30)", "P(t_i) = exp(z_i/T) / sum exp(z_j/T)", "T = 1.0",
      "top_k = 64", "top_p = 0.95", "nucleus 95%", "greedy T -> 0", "▁Paris #9079 0.61", "▁located #5628",
      "▁Lyon #49533", "weighted die", "MTP draft x4", "verify in 1 pass", "~3x faster", "<turn|> #106",
      "<eos> #1", "re-normalize", "argmax",
    ],
    "putting-it-together": [
      "<|turn>system", "<|turn>user", "<|turn>model", "<turn|>", "<|think|>", "<|channel>thought",
      "<channel|>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", '<|"|>',
      "apply_chat_template(...)", "enable_thinking=True", "add_generation_prompt=True", "parse_response()",
      "do_sample=True", "max_new_tokens=1024", "H2O", "SFT + RL", "post-training", "TextStreamer",
    ],
    "tuning-overview": [
      "Gemma 4", "E2B", "E4B", "12B", "26B A4B", "31B", "SFTTrainer", "LoraConfig",
      "BitsAndBytesConfig", "PromptTuningConfig", "DPOTrainer", "r = 16", "lora_alpha = 16",
      "4-bit NF4", "bfloat16", "gradient_checkpointing", "adamw_torch_fused",
      "modules_to_save", "merge_and_unload()", "safetensors", "PEFT", "TRL",
    ],
    full: [
      "SFTTrainer", "SFTConfig", "lr = 2e-5", "bfloat16", "8 B / param", "weights + grads + moments",
      "gradient_checkpointing=True", "per_device_train_batch_size=2", "gradient_accumulation_steps=4",
      "cosine", "warmup_steps=0.03", "adamw_torch_fused", "trainer.save_model()", "safetensors",
      "100% trainable", "catastrophic forgetting", "eval_strategy='epoch'", "Gemma 4 E2B",
    ],
    lora: [
      "LoraConfig", "W + (alpha/r) B A", "r = 16", "lora_alpha = 16", "lora_dropout = 0.05",
      "lr = 2e-4", "B = 0 at init", "task_type='CAUSAL_LM'", "q_proj k_proj v_proj o_proj",
      "gate_proj up_proj down_proj", "adapter_model.safetensors", "merge_and_unload()",
      "PeftModel", "frozen bf16 base", "~1% trainable", "ensure_weight_tying=True", "enable_lora(rank=4)",
    ],
    qlora: [
      "BitsAndBytesConfig", "load_in_4bit=True", "bnb_4bit_quant_type='nf4'",
      "bnb_4bit_use_double_quant=True", "bnb_4bit_compute_dtype=bfloat16", "NormalFloat4",
      "~0.5 B / param", "dequantize -> bf16 matmul", "paged_adamw_8bit", "modules_to_save",
      "embed_tokens", "lm_head", "ensure_weight_tying=True", "Colab T4 16 GB", "LoraConfig(r=16)",
    ],
    prompt: [
      "PromptTuningConfig", "PromptTuningInit.TEXT", "num_virtual_tokens = 20", "20 x d_model",
      "lr = 3e-3", "num_train_epochs = 10", "soft prompt prefix", "[P_1, P_2, ..., P_k]",
      "frozen transformer", "PrefixTuningConfig", "prefix_projection=True",
      "adapter_model.safetensors", "< 1 MB artifact", "task_type='CAUSAL_LM'",
    ],
    dpo: [
      "DPOTrainer", "DPOConfig", "beta = 0.1", "lr = 5e-6", "chosen vs. rejected",
      "log pi(y_w|x) - log pi_ref(y_w|x)", "ref_model=None", "adapters disabled = ref",
      "ultrafeedback_binarized", "rewards/margins", "rewards/accuracies", "loss_type='sigmoid'",
      "precompute_ref_log_probs", "LoraConfig(r=16)", "SFT -> DPO",
    ],
  };

  /* ------------------------------------------------------------------ utils */
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (e0, e1, x) => {
    const t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  };
  const prefersReducedMotion = () =>
    typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isDark = () => document.body.getAttribute("data-md-color-scheme") === "slate";

  const parseHex = (value, fallback) => {
    const m = /^#?([0-9a-f]{6})$/i.exec(String(value || "").trim());
    if (!m) return fallback;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const mix = (a, b, t) => [Math.round(lerp(a[0], b[0], t)), Math.round(lerp(a[1], b[1], t)), Math.round(lerp(a[2], b[2], t))];
  const rgb = (c) => `rgb(${c[0]} ${c[1]} ${c[2]})`;

  /* The brand ramp, derived the way gemma.css derives it: the seed mixed with white
     (or black), so the canvas follows `--gemma-blue-seed` like everything else. */
  const readPalette = (el) => {
    const cs = getComputedStyle(el);
    const seed = parseHex(cs.getPropertyValue("--gemma-blue-seed"), [11, 87, 208]);
    const n500 = parseHex(cs.getPropertyValue("--gemma-n500"), [107, 116, 134]);
    const n300 = parseHex(cs.getPropertyValue("--gemma-n300"), [170, 179, 194]);
    const white = [255, 255, 255];
    const black = [0, 0, 0];
    return isDark()
      ? { star: rgb(mix(white, seed, 0.52)), line: rgb(mix(white, seed, 0.7)), fog: rgb(n300), hot: rgb(mix(white, seed, 0.24)) }
      : { star: rgb(seed), line: rgb(mix(white, seed, 0.78)), fog: rgb(n500), hot: rgb(mix(black, seed, 0.78)) };
  };

  const readFont = (el) => {
    const family = getComputedStyle(el).getPropertyValue("--md-code-font-family").trim();
    return family ? `${family}, ${FONT_FALLBACK}` : FONT_FALLBACK;
  };

  const segmentDistance = (px, py, x1, y1, x2, y2) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len2 = dx * dx + dy * dy || 1e-6;
    const t = clamp(((px - x1) * dx + (py - y1) * dy) / len2, 0, 1);
    return Math.hypot(px - (x1 + dx * t), py - (y1 + dy * t));
  };

  /* Glyph for a line running along (tx, ty) in screen space (y down). */
  const lineGlyph = (tx, ty) => {
    const ax = Math.abs(tx);
    const ay = Math.abs(ty);
    if (ax > 2.414 * ay) return 45; // -
    if (ay > 2.414 * ax) return 124; // |
    return tx * ty < 0 ? 47 : 92; // '/' rises to the right, otherwise '\'
  };

  /* ================================================================== canvas */
  const createAsciiField = (shell, canvas) => {
    const ctx = canvas.getContext("2d");
    const reduced = prefersReducedMotion();
    const KIND_LINE = 1;
    const KIND_STAR = 2;

    let W = 1;
    let H = 1;
    let cols = 1;
    let rows = 1;
    let cellW = 7;
    let cellH = 13;
    let fontPx = 12;
    let font = readFont(shell);
    let palette = readPalette(shell);

    /* Per-cell buffers for the mark: intensity, kind, glyph code, angle around the centre. */
    let I = new Float32Array(0);
    let K = new Uint8Array(0);
    let C = new Uint16Array(0);
    let A = new Float32Array(0);
    let fieldDirty = true;

    /* Where the mark is and where it is going; eased every frame. */
    const POSE_KEYS = ["cx", "cy", "R", "gain", "mask", "maskX", "maskY", "maskRX", "maskRY"];
    const pose = { cx: 0, cy: 0, R: 120, gain: 1, mask: 0, maskX: 0, maskY: 0, maskRX: 1, maskRY: 1 };
    const target = { ...pose };
    let snapNext = true;

    const fogList = (s) =>
      (s === "overview" && shell.dataset.llmFog && FOG_WORDS[`${shell.dataset.llmFog}-overview`]) ||
      FOG_WORDS[s] ||
      FOG_WORDS.overview;
    const fog = { words: [], list: fogList("overview") };
    const pointer = { x: -1e9, y: -1e9, active: false, energy: 0 };
    const ripples = [];
    let step = "overview";
    let running = false;
    let rafId = 0;
    let lastFrame = 0;
    const t0 = performance.now();

    /* ---------------------------------------------------------- geometry */
    const measure = () => {
      fontPx = clamp(Math.round(W / 150), 10, 14);
      ctx.font = `${fontPx}px ${font}`;
      cellW = Math.max(5, ctx.measureText("M").width);
      cellH = Math.round(fontPx * 1.18);
      cols = Math.ceil(W / cellW);
      rows = Math.ceil(H / cellH);
    };

    const snapX = (x) => (Math.round(x / cellW - 0.5) + 0.5) * cellW;
    const snapY = (y) => (Math.round(y / cellH - 0.5) + 0.5) * cellH;

    const buildField = () => {
      const n = cols * rows;
      if (I.length !== n) {
        I = new Float32Array(n);
        K = new Uint8Array(n);
        C = new Uint16Array(n);
        A = new Float32Array(n);
      } else {
        I.fill(0);
        K.fill(0);
      }

      const R = pose.R;
      const cx = snapX(pose.cx);
      const cy = snapY(pose.cy);
      const s = R * Math.SQRT1_2; // half-side of the inscribed square
      const ext = 1.06 * R; // grid lines overshoot the circle slightly
      const tipEnd = 0.88 * R; // star spikes end here; the axes resume
      const far = Math.max(W, H);
      const gridX = [snapX(cx - s) - cx, snapX(cx + s) - cx];
      const gridY = [snapY(cy - s) - cy, snapY(cy + s) - cy];
      const diagIn = 0.2929 * s; // where a diagonal meets the star's concave edge
      const corners = [[1, -1], [1, 1], [-1, 1], [-1, -1]];
      const diagAlong = Math.hypot(Math.SQRT1_2 * cellW, Math.SQRT1_2 * cellH);
      const { mask, maskX, maskY } = pose;
      const maskRX = Math.max(1, pose.maskRX);
      const maskRY = Math.max(1, pose.maskRY);

      for (let j = 0; j < rows; j++) {
        const y = (j + 0.5) * cellH;
        const py = y - cy;
        const ay = Math.abs(py);
        for (let i = 0; i < cols; i++) {
          const x = (i + 0.5) * cellW;
          const px = x - cx;
          const ax = Math.abs(px);
          let best = 0;
          let kind = 0;
          let glyph = 0;

          // Circle.
          const r = Math.hypot(px, py) || 1e-6;
          if (Math.abs(r - R) < 2 * cellH) {
            const nx = px / r;
            const ny = py / r;
            const along = Math.hypot(nx * cellW, ny * cellH);
            const v = (1 - Math.abs(r - R) / (0.72 * along)) * 1.35;
            if (v > best) {
              best = v;
              kind = KIND_LINE;
              glyph = lineGlyph(-ny, nx);
            }
          }
          // Inscribed-square grid.
          if (ay <= ext) {
            const d = Math.min(Math.abs(px - gridX[0]), Math.abs(px - gridX[1]));
            const v = (1 - d / (0.55 * cellW)) * (1 - 0.5 * smoothstep(ext - 0.25 * R, ext, ay));
            if (v > best) {
              best = v;
              kind = KIND_LINE;
              glyph = 124;
            }
          }
          if (ax <= ext) {
            const d = Math.min(Math.abs(py - gridY[0]), Math.abs(py - gridY[1]));
            const v = (1 - d / (0.55 * cellH)) * (1 - 0.5 * smoothstep(ext - 0.25 * R, ext, ax));
            if (v > best) {
              best = v;
              kind = KIND_LINE;
              glyph = 45;
            }
          }
          // Axes, from the star tips outwards, fading with distance.
          if (ay >= tipEnd) {
            const v = (1 - ax / (0.55 * cellW)) * (1 - 0.6 * smoothstep(tipEnd, far, ay));
            if (v > best) {
              best = v;
              kind = KIND_LINE;
              glyph = 124;
            }
          }
          if (ax >= tipEnd) {
            const v = (1 - ay / (0.55 * cellH)) * (1 - 0.6 * smoothstep(tipEnd, far, ax));
            if (v > best) {
              best = v;
              kind = KIND_LINE;
              glyph = 45;
            }
          }
          // The two diagonals: square corner to the star's concave edge.
          if (ax <= s + cellW && ay <= s + cellH) {
            const d = Math.min(
              segmentDistance(px, py, -s, -s, -diagIn, -diagIn),
              segmentDistance(px, py, diagIn, diagIn, s, s),
            );
            const v = (1 - d / (0.72 * diagAlong)) * 1.35;
            if (v > best) {
              best = v;
              kind = KIND_LINE;
              glyph = 92;
            }
          }
          // The star: four concave arcs, one per corner of the square.
          if (ax <= s + 0.04 * R && ay <= s + 0.04 * R) {
            for (let c = 0; c < 4; c++) {
              const sx = corners[c][0];
              const sy = corners[c][1];
              if (sx * px < -0.04 * R || sy * py < -0.04 * R) continue;
              const ex = px - sx * s;
              const ey = py - sy * s;
              const dc = Math.hypot(ex, ey) || 1e-6;
              const along = Math.hypot((ex / dc) * cellW, (ey / dc) * cellH);
              const w = Math.max(0.05 * R, 1.1 * along);
              const v = 1 - Math.abs(dc - s) / w;
              if (v > best) {
                best = v;
                kind = KIND_STAR;
              }
            }
          }
          // Star tips: short spikes along the axes, tapering in width and density.
          // One pixel width for both axes, or the horizontal pair reads as heavy bars.
          const spikeW = Math.max(0.05 * R, 1.1 * cellW);
          if (ay >= s - cellH && ay <= tipEnd && ax < 0.12 * R) {
            const u = clamp((ay - s) / (tipEnd - s), 0, 1);
            const w = spikeW * Math.pow(1 - u, 1.35) + 0.45 * cellW;
            const v = (1 - ax / w) * (1 - 0.5 * u);
            if (v > best) {
              best = v;
              kind = KIND_STAR;
            }
          }
          if (ax >= s - cellW && ax <= tipEnd && ay < 0.12 * R) {
            const u = clamp((ax - s) / (tipEnd - s), 0, 1);
            // Rows are twice as tall as columns are wide, so a row of glyphs reads heavier
            // than a column: keep this band thinner than the vertical one.
            const w = spikeW * 0.7 * Math.pow(1 - u, 1.35) + 0.3 * cellH;
            const v = (1 - ay / w) * (1 - 0.5 * u);
            if (v > best) {
              best = v;
              kind = KIND_STAR;
            }
          }
          if (best <= 0.03) continue;

          // Keep the hero copy legible. `mask` carries the depth: 1 thins the field to
          // ~30% under the copy (centered hero), 1.4 hollows it out (split hero).
          if (mask > 0.001) {
            const d = Math.hypot((x - maskX) / maskRX, (y - maskY) / maskRY);
            best *= 1 - Math.min(1, 0.7 * mask) * (1 - smoothstep(0.85, 1.3, d));
          }
          const idx = j * cols + i;
          I[idx] = Math.min(1, best);
          K[idx] = kind;
          C[idx] = glyph;
          A[idx] = Math.atan2(py, px);
        }
      }
      fieldDirty = false;
    };

    /* ------------------------------------------------------------- fog */
    /* True when a word at (col,row,len) would sit on the mark or over the hero copy. */
    const wordBlocked = (col, row, len) => {
      const y = (row + 0.5) * cellH;
      const x0 = (col + 0.5) * cellW;
      const x1 = (col + len - 0.5) * cellW;
      // Nearest point of the span to the mark's centre, against a disc a little larger than it.
      const nx = clamp(pose.cx, x0, x1);
      if (Math.hypot(nx - pose.cx, y - pose.cy) < 1.12 * pose.R) return true;
      if (pose.mask > 0.5) {
        const mx = clamp(pose.maskX, x0, x1);
        if (Math.hypot((mx - pose.maskX) / pose.maskRX, (y - pose.maskY) / pose.maskRY) < 1.15) return true;
      }
      const base = row * cols;
      for (let k = 0; k < len; k++) {
        if (col + k >= cols) break;
        if (I[base + col + k] > 0.1) return true;
      }
      return false;
    };

    /* True when a word at (col,row,len) would collide with a live word (one row of air). */
    const wordCollides = (self, col, row, len) => {
      for (const w of fog.words) {
        if (w === self || !w.text) continue;
        if (Math.abs(w.row - row) > 1) continue;
        if (col <= w.col + w.text.length + 1 && w.col <= col + len + 1) return true;
      }
      return false;
    };

    const spawnWord = (word, now, stagger) => {
      const list = fog.list;
      const text = list[Math.floor(Math.random() * list.length)];
      const life = 7000 + Math.random() * 7000;
      let col = 0;
      let row = 0;
      for (let attempt = 0; attempt < 10; attempt++) {
        col = Math.floor(Math.random() * Math.max(1, cols - text.length));
        row = Math.floor(Math.random() * rows);
        if (!wordBlocked(col, row, text.length) && !wordCollides(word, col, row, text.length)) break;
      }
      word.text = text;
      word.col = col;
      word.row = row;
      word.life = life;
      word.born = now - (stagger ? Math.random() * life : 0);
      word.peak = 0.1 + Math.random() * 0.16;
      word.dieAt = 0;
      return word;
    };

    const seedFog = (now) => {
      fog.words.length = 0;
      const count = clamp(Math.round((cols * rows) / 120), 24, 120);
      for (let i = 0; i < count; i++) fog.words.push(spawnWord({}, now, true));
    };

    /* ---------------------------------------------------------- drawing */
    const drawFog = (now) => {
      ctx.fillStyle = palette.fog;
      for (const w of fog.words) {
        const age = (now - w.born) / w.life;
        if (age >= 1 || (w.dieAt && now >= w.dieAt) || wordBlocked(w.col, w.row, w.text.length)) {
          spawnWord(w, now, false);
          continue;
        }
        let env = reduced ? 0.85 : smoothstep(0, 0.22, age) * (1 - smoothstep(0.72, 1, age));
        if (w.dieAt) env *= 1 - smoothstep(w.dieAt - 700, w.dieAt, now);
        const alpha = env * w.peak;
        if (alpha < 0.01) continue;
        const y = (w.row + 0.5) * cellH;
        ctx.globalAlpha = alpha;
        for (let k = 0; k < w.text.length; k++) {
          const col = w.col + k;
          if (col >= cols) break;
          const ch = w.text[k];
          if (ch === " ") continue;
          ctx.fillText(ch, (col + 0.5) * cellW, y);
        }
      }
    };

    const drawMark = (t) => {
      const sweep = (t * 0.55) % TAU;
      const pr = 120;
      const pr2 = pr * pr;
      const lineGain = 0.35 + 0.65 * pose.gain;
      for (let idx = 0; idx < I.length; idx++) {
        const v0 = I[idx];
        if (v0 < 0.03) continue;
        const kind = K[idx];
        const i = idx % cols;
        const j = (idx - i) / cols;
        const x = (i + 0.5) * cellW;
        const y = (j + 0.5) * cellH;

        let dAng = Math.abs(A[idx] - sweep);
        if (dAng > Math.PI) dAng = TAU - dAng;
        const highlight = reduced ? 0 : Math.exp(-dAng * dAng * 9) * 0.6;

        let alpha;
        let glyph;
        let style;
        if (kind === KIND_STAR) {
          const shimmer = reduced ? 1 : 0.82 + 0.18 * Math.sin(t * 1.9 + A[idx] * 3);
          const v = Math.min(1, v0 * pose.gain * shimmer + highlight * v0);
          glyph = STAR_RAMP[Math.min(STAR_RAMP.length - 1, 1 + Math.floor(v * (STAR_RAMP.length - 1)))];
          alpha = (0.3 + 0.7 * v) * lineGain;
          style = palette.star;
        } else {
          glyph = String.fromCharCode(C[idx]);
          alpha = (0.2 + 0.5 * Math.min(1, v0) + highlight * 0.35 * v0) * lineGain;
          style = palette.line;
        }

        let boost = 0;
        if (pointer.energy > 0.02) {
          const dx = x - pointer.x;
          const dy = y - pointer.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < pr2) {
            const f = 1 - Math.sqrt(d2) / pr;
            boost = f * f * pointer.energy;
          }
        }
        for (let k = 0; k < ripples.length; k++) {
          const rp = ripples[k];
          const d = Math.abs(Math.hypot(x - rp.x, y - rp.y) - rp.r);
          if (d < 30) boost = Math.max(boost, (1 - d / 30) * rp.a);
        }
        if (boost > 0.22) style = palette.hot;
        ctx.globalAlpha = Math.min(1, alpha + boost * 0.7);
        ctx.fillStyle = style;
        ctx.fillText(glyph, x, y);
      }
    };

    const render = (now) => {
      rafId = 0;
      if (!shell.isConnected) {
        stop();
        return;
      }
      if (running && !reduced) rafId = requestAnimationFrame(render);
      if (document.hidden) return;
      if (now - lastFrame < 1000 / FPS - 1) return;
      lastFrame = now;
      const t = (now - t0) / 1000;

      let moved = false;
      for (const key of POSE_KEYS) {
        const d = target[key] - pose[key];
        if (snapNext || reduced || Math.abs(d) < 0.02) {
          if (pose[key] !== target[key]) moved = true;
          pose[key] = target[key];
        } else {
          pose[key] += d * 0.14;
          moved = true;
        }
      }
      snapNext = false;
      if (moved || fieldDirty) buildField();

      pointer.energy += ((pointer.active ? 1 : 0) - pointer.energy) * 0.12;
      for (let k = ripples.length - 1; k >= 0; k--) {
        const rp = ripples[k];
        rp.r += 8;
        rp.a = 1 - rp.r / rp.max;
        if (rp.a <= 0) ripples.splice(k, 1);
      }

      ctx.clearRect(0, 0, W, H);
      ctx.font = `${fontPx}px ${font}`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      drawFog(now);
      drawMark(t);
      ctx.globalAlpha = 1;

      // Reduced motion: settle, then stop. Any later change requests a single frame.
      if (reduced && running && (moved || ripples.length || pointer.energy > 0.02)) {
        rafId = requestAnimationFrame(render);
      }
    };

    const requestRender = () => {
      if (!rafId && running) rafId = requestAnimationFrame(render);
    };

    /* ----------------------------------------------------------- layout */
    const relayout = () => {
      const rect = shell.getBoundingClientRect();
      if (step === "overview") {
        const hero = shell.querySelector("[data-llm-hero-text]");
        // The stylesheet switches to a split hero on wide screens; follow it.
        const split = hero ? getComputedStyle(hero).textAlign === "left" : false;
        const narrow = W < 640;
        if (split) {
          target.cx = W * 0.72;
          target.cy = H * 0.52;
          target.R = clamp(Math.min(W * 0.19, H * 0.36), 120, 300);
          target.gain = 1;
        } else {
          target.cx = W * 0.5;
          target.cy = H * (narrow ? 0.42 : 0.47);
          target.R = narrow
            ? clamp(Math.min(W * 0.46, H * 0.3), 100, 220)
            : clamp(Math.min(W * 0.24, H * 0.36), 110, 340);
          target.gain = narrow ? 0.7 : 1;
        }
        if (hero) {
          const hr = hero.getBoundingClientRect();
          target.maskX = hr.left - rect.left + hr.width / 2;
          target.maskY = hr.top - rect.top + hr.height / 2;
          target.maskRX = Math.max(1, hr.width * 0.52);
          target.maskRY = Math.max(1, hr.height * 0.58);
          target.mask = split ? 1.4 : 1;
        }
      } else {
        const card = shell.querySelector(`[data-llm-panel="${step}"] .llm-shell__card`);
        const right = card ? card.getBoundingClientRect().right - rect.left : W;
        const room = W - right;
        target.mask = 0;
        if (room >= 280) {
          target.cx = right + room * 0.5;
          target.cy = H * 0.46;
          target.R = clamp(Math.min(room * 0.42, H * 0.3), 90, 260);
          target.gain = 0.95;
        } else {
          target.cx = W * 0.5;
          target.cy = H * 0.45;
          target.R = clamp(Math.min(W * 0.3, H * 0.34), 110, 320);
          target.gain = 0.38;
        }
      }
      requestRender();
    };

    const resize = () => {
      const rect = shell.getBoundingClientRect();
      W = Math.max(1, Math.round(rect.width));
      H = Math.max(1, Math.round(rect.height));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const prevCols = cols;
      const prevRows = rows;
      measure();
      fieldDirty = true;
      snapNext = true;
      if (!fog.words.length || cols !== prevCols || rows !== prevRows) seedFog(performance.now());
      relayout();
    };

    /* -------------------------------------------------------------- api */
    const setStep = (next) => {
      if (next === step) return;
      step = next;
      fog.list = fogList(next);
      const now = performance.now();
      for (const w of fog.words) {
        if (!w.dieAt) w.dieAt = now + 400 + Math.random() * 1400;
      }
      relayout();
    };

    const setTheme = () => {
      palette = readPalette(shell);
      requestRender();
    };

    const onPointerMove = (e) => {
      const rect = shell.getBoundingClientRect();
      pointer.x = e.clientX - rect.left;
      pointer.y = e.clientY - rect.top;
      pointer.active = true;
      requestRender();
    };
    const onPointerLeave = () => {
      pointer.active = false;
    };
    const onPointerDown = (e) => {
      if (e.target.closest("button, a, input, textarea, select, .llm-shell__card, .llm-terminal, .llm-shell__tabs")) return;
      const rect = shell.getBoundingClientRect();
      ripples.push({ x: e.clientX - rect.left, y: e.clientY - rect.top, r: 0, a: 1, max: Math.max(W, H) * 0.6 });
      requestRender();
    };

    const start = () => {
      if (running) return;
      running = true;
      resize();
      requestRender();
    };

    const stop = () => {
      running = false;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
    };

    const refreshFont = () => {
      font = readFont(shell);
      resize();
    };

    return { start, stop, resize, relayout, setStep, setTheme, refreshFont, onPointerMove, onPointerLeave, onPointerDown };
  };

  /* =================================================================== shell */
  const createShell = (shell) => {
    const controller = new AbortController();
    const { signal } = controller;
    const reduced = prefersReducedMotion();

    const canvas = shell.querySelector("[data-llm-canvas]");
    const tabList = shell.querySelector(".llm-shell__tabs");
    const terminal = shell.querySelector("[data-llm-terminal]");
    const tBar = shell.querySelector("[data-llm-terminal-bar]");
    const tBody = shell.querySelector("[data-llm-terminal-body]");
    const tContent = shell.querySelector("[data-llm-terminal-content]");
    const tToggle = shell.querySelector("[data-llm-terminal-toggle]");
    const tToggleLabel = shell.querySelector("[data-llm-terminal-toggle-label]");
    const tMax = shell.querySelector("[data-llm-terminal-max]");
    const tHint = shell.querySelector("[data-llm-terminal-hint]");
    const tHandle = shell.querySelector("[data-llm-terminal-handle]");
    const tFiles = shell.querySelectorAll("[data-llm-terminal-file]");

    const formatWords = (n) => (n >= 950 ? `${(n / 1000).toFixed(1)}k words` : `${n} words`);

    const steps = [...shell.querySelectorAll(".llm-shell__tab[data-llm-step]")].map((tab) => {
      const id = tab.dataset.llmStep;
      const doc = tContent ? tContent.querySelector(`.llm-doc[data-llm-doc="${id}"]`) : null;
      const words = doc ? doc.textContent.trim().split(/\s+/).length : 0;
      return {
        id,
        tab,
        file: tab.dataset.llmFile || `${id}.md`,
        label: tab.dataset.llmLabel || id,
        panel: shell.querySelector(`[data-llm-panel="${id}"]`),
        doc,
        words: words ? formatWords(words) : "",
      };
    });
    const stepById = (id) => steps.find((s) => s.id === id);

    // Everything in the body before the first file belongs to the overview.
    const preamble = [];
    if (tContent) {
      for (const el of tContent.children) {
        if (el.classList.contains("llm-doc")) break;
        if (el.tagName !== "H1") el.setAttribute("data-llm-preamble", "");
        preamble.push(el);
      }
    }

    const ascii = canvas ? createAsciiField(shell, canvas) : null;
    const terminalState = { mode: "closed", height: 0 };
    let current = null;

    /* ---------------------------------------------------------- terminal */
    const applyTerminal = ({ animate = true } = {}) => {
      if (!animate) shell.classList.add("is-dragging");
      const open = terminalState.mode !== "closed";
      shell.classList.toggle("is-terminal-open", open);
      shell.classList.toggle("is-terminal-max", terminalState.mode === "max");
      shell.style.setProperty(
        "--llm-terminal-open-h",
        terminalState.mode === "open" && terminalState.height ? `${terminalState.height}px` : "",
      );
      if (tToggle) tToggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (tToggleLabel) tToggleLabel.textContent = open ? "Collapse" : "Expand";
      if (tMax) {
        const isMax = terminalState.mode === "max";
        const maxLabel = isMax ? "Restore terminal" : "Maximize terminal";
        tMax.setAttribute("aria-pressed", isMax ? "true" : "false");
        tMax.setAttribute("title", maxLabel);
        tMax.setAttribute("aria-label", maxLabel);
      }
      if (tBody) tBody.toggleAttribute("inert", !open);
      if (tHint && current) {
        tHint.textContent = `${current.words}${current.words ? " · " : ""}${open ? "drag the edge to resize" : "click to read"}`;
      }
      try {
        localStorage.setItem(STORAGE_KEY, terminalState.mode);
      } catch (_) {
        /* storage may be unavailable; the drawer still works */
      }
      if (!animate) {
        void shell.offsetHeight;
        shell.classList.remove("is-dragging");
      }
      if (ascii) {
        ascii.relayout();
        window.setTimeout(() => ascii.relayout(), 300);
      }
    };

    const setTerminal = (mode, options) => {
      terminalState.mode = mode;
      applyTerminal(options);
    };

    const openTerminal = () => {
      if (terminalState.mode === "closed") setTerminal("open");
      if (tBody) tBody.scrollTop = 0;
    };

    const scrollTerminalTo = (el) => {
      if (!el || !tBody) return;
      const top = el.getBoundingClientRect().top - tBody.getBoundingClientRect().top + tBody.scrollTop - 12;
      tBody.scrollTo({ top: Math.max(0, top), behavior: reduced ? "auto" : "smooth" });
    };

    /* ------------------------------------------------------------- steps */
    const centerTab = (tab) => {
      if (!tabList || tabList.scrollWidth <= tabList.clientWidth) return;
      const left = tab.offsetLeft - (tabList.clientWidth - tab.offsetWidth) / 2;
      tabList.scrollTo({ left: Math.max(0, left), behavior: reduced ? "auto" : "smooth" });
    };

    const setStep = (id, { updateHash = true, focusTab = false } = {}) => {
      const next = stepById(id) || steps[0];
      if (!next) return;
      const changed = next !== current;
      current = next;
      shell.dataset.step = next.id;
      for (const s of steps) {
        const active = s === next;
        s.tab.classList.toggle("is-active", active);
        s.tab.setAttribute("aria-selected", active ? "true" : "false");
        s.tab.tabIndex = active ? 0 : -1;
        if (s.panel) s.panel.hidden = !active;
        if (s.doc) s.doc.hidden = !active;
      }
      for (const el of preamble) el.hidden = next.id !== "overview";
      for (const el of tFiles) el.textContent = next.file;
      if (tHint) {
        const open = terminalState.mode !== "closed";
        tHint.textContent = `${next.words}${next.words ? " · " : ""}${open ? "drag the edge to resize" : "click to read"}`;
      }
      if (changed) {
        if (next.panel) next.panel.scrollTop = 0;
        if (tBody) tBody.scrollTop = 0;
      }
      if (updateHash) {
        const params = new URLSearchParams(location.search);
        params.delete("t");
        const qs = params.toString();
        const search = qs ? `?${qs}` : "";
        const url = next.id === "overview" ? `${location.pathname}${search}` : `${location.pathname}${search}#${next.id}`;
        try {
          history.replaceState(history.state, "", url);
        } catch (_) {
          /* ignore */
        }
      }
      if (focusTab) next.tab.focus({ preventScroll: true });
      centerTab(next.tab);
      if (ascii) ascii.setStep(next.id);
      shell.dispatchEvent(new CustomEvent("llm:step", { detail: { step: next.id } }));
      if (changed) {
        // The card is measured for the mark's position, so give it a frame to lay out.
        requestAnimationFrame(() => ascii && ascii.relayout());
      }
    };

    const stepIndex = () => steps.indexOf(current);
    const moveStep = (delta, focusTab) => {
      const idx = clamp(stepIndex() + delta, 0, steps.length - 1);
      setStep(steps[idx].id, { focusTab });
    };

    /* Deep links: `#tokenization` opens a stage; any other id is looked up in the files. */
    const jumpToAnchor = (id) => {
      if (!id) return false;
      if (stepById(id)) {
        setStep(id, { updateHash: false });
        return true;
      }
      if (!tContent) return false;
      let el = null;
      try {
        el = tContent.querySelector(`#${CSS.escape(id)}`);
      } catch (_) {
        el = null;
      }
      if (!el) return false;
      const doc = el.closest(".llm-doc");
      setStep(doc ? doc.dataset.llmDoc : "overview", { updateHash: false });
      openTerminal();
      window.setTimeout(() => scrollTerminalTo(el), 320);
      return true;
    };

    /* The page itself must never scroll; the stage and the drawer do. */
    const pinPage = () => {
      for (const el of [document.documentElement, document.body, document.querySelector(".md-container"), document.querySelector(".md-main")]) {
        if (el && el.scrollTop) el.scrollTop = 0;
      }
    };

    /* ------------------------------------------------------------ events */
    shell.addEventListener(
      "click",
      (e) => {
        const stepBtn = e.target.closest("[data-llm-step]");
        if (stepBtn && shell.contains(stepBtn)) {
          e.preventDefault();
          setStep(stepBtn.dataset.llmStep, { focusTab: stepBtn.classList.contains("llm-shell__tab") });
          return;
        }
        const openBtn = e.target.closest("[data-llm-terminal-open]");
        if (openBtn) {
          // Some of these are #chapter anchors (a no-JS fallback); keep instant navigation out.
          e.preventDefault();
          e.stopPropagation();
          const targetId = openBtn.dataset.llmTerminalOpen || (openBtn.getAttribute("href") || "").replace(/^#/, "");
          if (targetId && !stepById(targetId) && jumpToAnchor(targetId)) return;
          openTerminal();
          return;
        }
        if (e.target.closest("[data-llm-terminal-close]")) {
          e.preventDefault();
          setTerminal("closed");
        }
      },
      { signal },
    );

    if (tBar) {
      tBar.addEventListener(
        "click",
        (e) => {
          if (e.target.closest("button")) return;
          setTerminal(terminalState.mode === "closed" ? "open" : "closed");
        },
        { signal },
      );
    }
    if (tToggle) {
      tToggle.addEventListener("click", () => setTerminal(terminalState.mode === "closed" ? "open" : "closed"), { signal });
    }
    if (tMax) {
      tMax.addEventListener("click", () => setTerminal(terminalState.mode === "max" ? "open" : "max"), { signal });
    }

    if (tHandle && terminal) {
      tHandle.addEventListener(
        "pointerdown",
        (e) => {
          if (terminalState.mode !== "open") return;
          e.preventDefault();
          const startY = e.clientY;
          const startH = terminal.getBoundingClientRect().height;
          const maxH = shell.clientHeight - parseFloat(getComputedStyle(shell).getPropertyValue("--llm-rail-h")) * 16;
          shell.classList.add("is-dragging");
          tHandle.setPointerCapture(e.pointerId);
          const onMove = (ev) => {
            terminalState.height = clamp(startH + (startY - ev.clientY), 160, Math.max(160, maxH));
            shell.style.setProperty("--llm-terminal-open-h", `${terminalState.height}px`);
          };
          const onUp = () => {
            tHandle.removeEventListener("pointermove", onMove);
            tHandle.removeEventListener("pointerup", onUp);
            tHandle.removeEventListener("pointercancel", onUp);
            shell.classList.remove("is-dragging");
            if (ascii) ascii.relayout();
          };
          tHandle.addEventListener("pointermove", onMove);
          tHandle.addEventListener("pointerup", onUp);
          tHandle.addEventListener("pointercancel", onUp);
        },
        { signal },
      );
    }

    // Links between chapters are in-page anchors: switch stage and scroll the drawer instead of
    // the (pinned) page. Zensical's instant navigation listens on <body> and ignores
    // defaultPrevented, so stopping propagation is what keeps it out of the way.
    if (tContent) {
      tContent.addEventListener(
        "click",
        (e) => {
          const a = e.target.closest("a[href]");
          if (!a || a.target === "_blank") return;
          let url;
          try {
            url = new URL(a.getAttribute("href"), location.href);
          } catch (_) {
            return;
          }
          if (url.origin !== location.origin || url.pathname !== location.pathname || !url.hash) return;
          if (!jumpToAnchor(decodeURIComponent(url.hash.slice(1)))) return;
          e.preventDefault();
          e.stopPropagation();
          try {
            history.replaceState(history.state, "", `${location.pathname}${location.search}${url.hash}`);
          } catch (_) {
            /* ignore */
          }
        },
        { signal },
      );
    }

    document.addEventListener(
      "keydown",
      (e) => {
        if (!shell.isConnected || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
        const typing = e.target.closest && e.target.closest("input, textarea, select, [contenteditable]");
        if (typing) return;
        const inTabs = tabList && tabList.contains(e.target);
        const idle = document.activeElement === document.body || inTabs;
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          if (!idle) return;
          e.preventDefault();
          moveStep(e.key === "ArrowRight" ? 1 : -1, inTabs);
        } else if ((e.key === "Home" || e.key === "End") && inTabs) {
          e.preventDefault();
          setStep(e.key === "Home" ? steps[0].id : steps[steps.length - 1].id, { focusTab: true });
        } else if ((e.key === "t" || e.key === "T") && idle) {
          e.preventDefault();
          setTerminal(terminalState.mode === "closed" ? "open" : "closed");
        } else if (e.key === "Escape" && terminalState.mode !== "closed") {
          setTerminal(terminalState.mode === "max" ? "open" : "closed");
        }
      },
      { signal },
    );

    if (ascii) {
      shell.addEventListener("pointermove", ascii.onPointerMove, { signal, passive: true });
      shell.addEventListener("pointerleave", ascii.onPointerLeave, { signal });
      shell.addEventListener("pointerdown", ascii.onPointerDown, { signal });
      window.addEventListener("resize", () => ascii.resize(), { signal });
      document.addEventListener(
        "visibilitychange",
        () => {
          if (!document.hidden) ascii.relayout();
        },
        { signal },
      );
      if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(() => {
          if (shell.isConnected) ascii.refreshFont();
        });
      }
    }

    window.addEventListener(
      "hashchange",
      () => {
        pinPage();
        jumpToAnchor(location.hash.slice(1));
      },
      { signal },
    );

    const themeObserver = new MutationObserver(() => ascii && ascii.setTheme());
    themeObserver.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

    /* --------------------------------------------------------------- init */
    let stored = "closed";
    try {
      stored = localStorage.getItem(STORAGE_KEY) || "closed";
    } catch (_) {
      stored = "closed";
    }
    terminalState.mode = stored === "max" ? "open" : stored === "open" ? "open" : "closed";

    const urlParams = new URLSearchParams(location.search);
    const queryStep = urlParams.get("t");
    setStep(queryStep && stepById(queryStep) ? queryStep : "overview", { updateHash: false });
    applyTerminal({ animate: false });
    if (ascii) ascii.start();
    if (location.hash) {
      jumpToAnchor(location.hash.slice(1));
      window.setTimeout(pinPage, 0);
    }
    pinPage();

    return {
      shell,
      destroy() {
        controller.abort();
        themeObserver.disconnect();
        if (ascii) ascii.stop();
      },
    };
  };

  /* ================================================================ bootstrap */
  let active = null;

  const init = () => {
    if (active && !active.shell.isConnected) {
      active.destroy();
      active = null;
    }
    const shell = document.querySelector(SHELL);
    if (!shell || shell.dataset.llmShellReady === "true") return;
    shell.dataset.llmShellReady = "true";
    active = createShell(shell);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  }
})();
