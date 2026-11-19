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

/* Interactive Fine-Tuning Map & Pipeline Dropdown for docs/tuning/index.md.
   Uses strict DOM construction (zero innerHTML) for XSS safety. */
(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  const PIPELINE_GROUPS = [
    { group: null, items: ["all"] },
    { group: "Supervised", items: ["quickstart", "full", "moe", "tokens"] },
    { group: "Multimodal", items: ["vision", "audio"] },
    { group: "Reinforcement learning", items: ["preference", "reasoning"] },
    { group: "From base model", items: ["domain", "posttrain"] },
  ];

  const ROUTES = {
    all: { label: "All guides", steps: [] },
    quickstart: {
      label: "Quickstart",
      steps: ["best-practices", "datasets-formatting", "qlora", "export"],
    },
    full: {
      label: "Full fine-tuning",
      steps: ["best-practices", "datasets-formatting", "full-finetune", "scaling", "export"],
    },
    moe: {
      label: "MoE model",
      steps: ["best-practices", "datasets-formatting", "moe-finetune", "export"],
    },
    tokens: {
      label: "Custom tokens",
      steps: ["best-practices", "datasets-formatting", "unused-tokens", "export"],
    },
    vision: {
      label: "Vision",
      steps: ["best-practices", "datasets-formatting", "vision-qlora", "export"],
    },
    audio: {
      label: "Audio",
      steps: ["best-practices", "datasets-formatting", "audio-qlora", "export"],
    },
    preference: {
      label: "Preference tuning",
      steps: ["datasets-formatting", "qlora", "preference-alignment", "export"],
    },
    reasoning: {
      label: "Reasoning with RL",
      steps: ["datasets-formatting", "qlora", "reinforcement-learning", "export"],
    },
    domain: {
      label: "Domain adaptation",
      steps: ["best-practices", "continued-pretraining", "custom-chat-template", "export"],
    },
    posttrain: {
      label: "Full post-training",
      steps: [
        "best-practices",
        "datasets-formatting",
        "continued-pretraining",
        "custom-chat-template",
        "full-finetune",
        "preference-alignment",
        "reinforcement-learning",
        "export",
      ],
    },
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svgEl(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  }

  function dedupePoints(rawPts) {
    const out = [];
    for (const [x, y] of rawPts) {
      if (!out.length || Math.hypot(x - out[out.length - 1][0], y - out[out.length - 1][1]) > 1.5) {
        out.push([x, y]);
      }
    }
    return out;
  }

  function roundedPolylinePath(rawPts, radius) {
    const pts = dedupePoints(rawPts);
    if (pts.length < 2) return "";
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const [px, py] = pts[i - 1];
      const [cx, cy] = pts[i];
      const [nx, ny] = pts[i + 1];
      const v1x = cx - px;
      const v1y = cy - py;
      const v2x = nx - cx;
      const v2y = ny - cy;
      const len1 = Math.hypot(v1x, v1y);
      const len2 = Math.hypot(v2x, v2y);
      const r = Math.min(radius, len1 * 0.45, len2 * 0.45);
      if (r < 1.5) {
        d += ` L ${cx.toFixed(1)} ${cy.toFixed(1)}`;
      } else {
        const s1x = cx - (v1x / len1) * r;
        const s1y = cy - (v1y / len1) * r;
        const s2x = cx + (v2x / len2) * r;
        const s2y = cy + (v2y / len2) * r;
        d += ` L ${s1x.toFixed(1)} ${s1y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${s2x.toFixed(1)} ${s2y.toFixed(1)}`;
      }
    }
    const [lx, ly] = pts[pts.length - 1];
    d += ` L ${lx.toFixed(1)} ${ly.toFixed(1)}`;
    return d;
  }

  function arrowPolygonPoints(pts) {
    const clean = dedupePoints(pts);
    if (clean.length < 2) return null;
    const [px, py] = clean[clean.length - 2];
    const [tx, ty] = clean[clean.length - 1];
    const dx = tx - px;
    const dy = ty - py;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const nx = -uy;
    const ny = ux;
    const back = 7.5;
    const halfW = 4.2;
    const bx = tx - ux * back;
    const by = ty - uy * back;
    return `${tx.toFixed(1)},${ty.toFixed(1)} ${(bx + nx * halfW).toFixed(1)},${(by + ny * halfW).toFixed(1)} ${(bx - nx * halfW).toFixed(1)},${(by - ny * halfW).toFixed(1)}`;
  }

  function computeHopPoints(a, b, cRect) {
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    const aNodes = a.closest(".tmap-nodes") || a.parentElement || a;
    const bNodes = b.closest(".tmap-nodes") || b.parentElement || b;
    const nRect = aNodes.getBoundingClientRect();
    const bnRect = bNodes.getBoundingClientRect();

    const nLeft = nRect.left - cRect.left;
    const nRight = nRect.right - cRect.left;
    const aNodesBottom = nRect.bottom - cRect.top;
    const bNodesTop = bnRect.top - cRect.top;
    const leftMargin = nLeft;
    const rightMargin = cRect.width - nRight;
    const xLeft = leftMargin >= 20 ? nLeft - 12 : Math.max(8, leftMargin * 0.5);
    const xRight = rightMargin >= 20 ? nRight + 12 : cRect.width - Math.max(8, rightMargin * 0.5);

    const ax1 = ra.left - cRect.left;
    const ax2 = ra.right - cRect.left;
    const ay1 = ra.top - cRect.top;
    const ay2 = ra.bottom - cRect.top;
    const acx = (ax1 + ax2) / 2;
    const acy = (ay1 + ay2) / 2;

    const bx1 = rb.left - cRect.left;
    const bx2 = rb.right - cRect.left;
    const by1 = rb.top - cRect.top;
    const bcx = (bx1 + bx2) / 2;
    const bcy = (by1 + (rb.bottom - cRect.top)) / 2;

    const aStage = a.closest(".tmap-stage");
    const bStage = b.closest(".tmap-stage");
    const asBottom = aStage ? aStage.getBoundingClientRect().bottom - cRect.top : aNodesBottom + 16;
    const bsTop = bStage ? bStage.getBoundingClientRect().top - cRect.top : bNodesTop;

    const isDesktop = nLeft > 60;
    const aOnBottomRow = aNodesBottom - ay2 < 16 && isDesktop;
    const bOnTopRow = by1 - bNodesTop < 16 && isDesktop;

    // 1. Same-row adjacent cards
    if (Math.abs(ay1 - by1) < 24 && bx1 > ax2 && bx1 - ax2 < 40) {
      return [
        [ax2, acy],
        [bx1, bcy],
      ];
    }

    // 2. Directly adjacent stages separated only by one inter-stage corridor
    if (aOnBottomRow && bOnTopRow && bsTop - asBottom >= -4 && bsTop - asBottom < 36) {
      return [
        [acx, ay2],
        [acx, asBottom],
        [bcx, asBottom],
        [bcx, by1],
      ];
    }

    // 3. Multi-row schematic corridor routing (zero card overlap, including middle columns)
    const aOnLeft = ax1 - nLeft < 16;
    const aOnRight = nRight - ax2 < 16;
    const bOnLeft = bx1 - nLeft < 16;
    const bOnRight = nRight - bx2 < 16;

    const midGridX = (nLeft + nRight) / 2;
    const preferRight =
      (aOnRight && bOnRight && isDesktop) ||
      (aOnRight && !aOnLeft && !bOnLeft) ||
      (!aOnLeft && bOnRight && acx >= midGridX);
    let curLane = preferRight ? xRight : xLeft;

    const pts = [];
    if (aOnBottomRow) {
      pts.push([acx, ay2], [acx, asBottom], [curLane, asBottom]);
    } else if (!preferRight && aOnLeft) {
      pts.push([ax1, acy], [xLeft, acy]);
      curLane = xLeft;
    } else if (aOnRight) {
      pts.push([ax2, acy], [xRight, acy]);
      curLane = xRight;
    } else if (!aOnLeft && !aOnRight && isDesktop) {
      const xGutter = ax1 - 6;
      pts.push([ax1, acy], [xGutter, acy], [xGutter, asBottom]);
      if (bOnTopRow && bsTop - asBottom >= -4 && bsTop - asBottom < 36) {
        pts.push([bcx, asBottom], [bcx, by1]);
        return pts;
      }
      pts.push([curLane, asBottom]);
    } else {
      pts.push([ax1, acy], [xLeft, acy]);
      curLane = xLeft;
    }

    if (bOnTopRow) {
      const yIn = bsTop - 16;
      pts.push([curLane, yIn], [bcx, yIn], [bcx, by1]);
    } else if (!bOnLeft && !bOnRight && isDesktop) {
      const yStageAbove = bsTop - 16;
      const xGutter = bx1 - 6;
      pts.push([curLane, yStageAbove], [xGutter, yStageAbove], [xGutter, bcy], [bx1, bcy]);
    } else {
      const targetLane = bOnLeft ? xLeft : xRight;
      if (Math.abs(curLane - targetLane) > 2) {
        const yCross = bsTop - 16;
        pts.push([curLane, yCross], [targetLane, yCross]);
        curLane = targetLane;
      }
      const bEdgeX = bOnLeft ? bx1 : bx2;
      pts.push([curLane, bcy], [bEdgeX, bcy]);
    }

    return pts;
  }

  function mountMap(root) {
    if (!root || root.dataset.tmapReady === "1") return;
    const canvas = root.querySelector(".tmap-canvas");
    if (!canvas) return;
    root.dataset.tmapReady = "1";

    const nodesBySlug = new Map();
    for (const a of root.querySelectorAll("a.tmap-node[data-guide]")) {
      nodesBySlug.set(a.dataset.guide, a);
    }

    const bar = el("div", "tmap-bar");
    const barLeft = el("div", "tmap-bar__left");
    barLeft.appendChild(el("span", "tmap-bar__label", "Highlight a pipeline"));

    const dropdown = el("div", "tmap-dropdown");
    const trigger = el("button", "tmap-trigger");
    trigger.type = "button";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    const triggerText = el("span", "tmap-trigger__text", ROUTES.all.label);
    const caret = svgEl("svg", {
      class: "tmap-trigger__caret",
      viewBox: "0 0 16 16",
      fill: "none",
      stroke: "currentColor",
      "stroke-width": "2",
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "aria-hidden": "true",
    });
    caret.appendChild(svgEl("path", { d: "M4 6l4 4 4-4" }));
    trigger.appendChild(triggerText);
    trigger.appendChild(caret);

    const menu = el("div", "tmap-menu");
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", "Fine-tuning pipelines");

    const optionsByRoute = new Map();
    for (const grp of PIPELINE_GROUPS) {
      if (grp.group) {
        menu.appendChild(el("div", "tmap-menu__group", grp.group));
      }
      for (const routeId of grp.items) {
        const route = ROUTES[routeId];
        if (!route) continue;
        const opt = el("button", "tmap-option");
        opt.type = "button";
        opt.setAttribute("role", "option");
        opt.dataset.route = routeId;
        opt.appendChild(el("span", "tmap-option__label", route.label));
        const metaText = route.steps.length ? `${route.steps.length} steps` : "16 guides";
        opt.appendChild(el("span", "tmap-option__meta", metaText));
        menu.appendChild(opt);
        optionsByRoute.set(routeId, opt);
      }
    }

    dropdown.appendChild(trigger);
    dropdown.appendChild(menu);
    barLeft.appendChild(dropdown);
    bar.appendChild(barLeft);

    const resetBtn = el("button", "tmap-reset", "Reset");
    resetBtn.type = "button";
    resetBtn.hidden = true;
    bar.appendChild(resetBtn);
    root.insertBefore(bar, canvas);

    const svg = svgEl("svg", { class: "tmap-wires", "aria-hidden": "true" });
    canvas.appendChild(svg);

    let activeRoute = "all";

    function closeMenu() {
      dropdown.classList.remove("is-open");
      trigger.setAttribute("aria-expanded", "false");
    }

    function toggleMenu() {
      const next = !dropdown.classList.contains("is-open");
      dropdown.classList.toggle("is-open", next);
      trigger.setAttribute("aria-expanded", next ? "true" : "false");
    }

    function drawWires() {
      svg.replaceChildren();
      const route = ROUTES[activeRoute];
      if (!route || route.steps.length < 2) return;
      const cRect = canvas.getBoundingClientRect();
      if (cRect.width === 0 || cRect.height === 0) return;

      svg.setAttribute("viewBox", `0 0 ${cRect.width} ${cRect.height}`);

      for (let i = 0; i < route.steps.length - 1; i++) {
        const a = nodesBySlug.get(route.steps[i]);
        const b = nodesBySlug.get(route.steps[i + 1]);
        if (!a || !b) continue;
        const pts = computeHopPoints(a, b, cRect);
        const d = roundedPolylinePath(pts, 9);
        if (!d) continue;

        svg.appendChild(svgEl("path", { d, class: "tmap-wire-bg" }));
        svg.appendChild(svgEl("path", { d, class: "tmap-wire-fg" }));
        svg.appendChild(svgEl("circle", { cx: pts[0][0].toFixed(1), cy: pts[0][1].toFixed(1), r: 3.5, class: "tmap-wire-dot" }));
        const arrowPts = arrowPolygonPoints(pts);
        if (arrowPts) {
          svg.appendChild(svgEl("polygon", { points: arrowPts, class: "tmap-wire-arrow" }));
        }
      }
    }

    function setRoute(routeId) {
      if (!Object.prototype.hasOwnProperty.call(ROUTES, routeId)) return;
      activeRoute = routeId;
      const route = ROUTES[routeId];
      const steps = route.steps;
      root.classList.toggle("has-route", steps.length > 0);
      resetBtn.hidden = steps.length === 0;
      triggerText.textContent = steps.length ? `${route.label} · ${steps.length} steps` : route.label;

      for (const [id, opt] of optionsByRoute.entries()) {
        const on = id === routeId;
        opt.classList.toggle("is-active", on);
        opt.setAttribute("aria-selected", on ? "true" : "false");
      }

      for (const [slug, node] of nodesBySlug.entries()) {
        const oldBadge = node.querySelector(".tmap-step-badge");
        if (oldBadge) oldBadge.remove();
        const idx = steps.indexOf(slug);
        node.classList.toggle("is-step", idx !== -1);
        if (idx !== -1) {
          const top = node.querySelector(".tmap-node__top") || node;
          top.appendChild(el("span", "tmap-step-badge", String(idx + 1)));
        }
      }

      requestAnimationFrame(drawWires);
    }

    trigger.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleMenu();
    });

    menu.addEventListener("click", (ev) => {
      const opt = ev.target.closest("button.tmap-option");
      if (!opt) return;
      setRoute(opt.dataset.route);
      closeMenu();
    });

    resetBtn.addEventListener("click", () => {
      setRoute("all");
      closeMenu();
    });

    document.addEventListener("click", (ev) => {
      if (!dropdown.contains(ev.target)) closeMenu();
    });

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && dropdown.classList.contains("is-open")) {
        closeMenu();
        trigger.focus();
      }
    });

    if (typeof ResizeObserver === "function") {
      const ro = new ResizeObserver(() => requestAnimationFrame(drawWires));
      ro.observe(canvas);
    } else {
      window.addEventListener("resize", () => requestAnimationFrame(drawWires));
    }

    setRoute("all");
  }

  function initAllMaps() {
    for (const root of document.querySelectorAll("#gemma-tuning-map, .tmap")) {
      mountMap(root);
    }
  }

  window.__initGemmaTuningMap = initAllMaps;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initAllMaps);
  else initAllMaps();
  if (window.document$?.subscribe) window.document$.subscribe(initAllMaps);
})();
