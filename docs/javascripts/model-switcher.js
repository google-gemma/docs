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

/* Rewrite explicitly opted-in examples to a reader's preferred Gemma 4 model. */
(() => {
  "use strict";

  const STORAGE_KEY = "gemma-model";
  const QUERY_KEY = "model";
  const models = window.GEMMA_MODELS || {};
  const index = [];

  for (const [id, model] of Object.entries(models)) {
    for (const [engine, value] of Object.entries(model.ids || {})) {
      index.push({ text: value, engine, model: id });
    }
  }
  index.sort((a, b) => b.text.length - a.text.length);
  const byText = new Map(index.map((entry) => [entry.text, entry]));
  const escapeRegExp = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = index.length
    ? new RegExp(index.map((entry) => escapeRegExp(entry.text)).join("|"), "g")
    : null;

  const blocks = () => document.querySelectorAll(".model-swap code, code.model-swap");
  const displayName = (id) => models[id]?.fullName || models[id]?.name || id;

  const readStored = () => {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  };

  const writeStored = (value) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, value);
    } catch {
      // Storage may be disabled; the current page still works.
    }
  };

  const readUrl = () => new URL(window.location.href).searchParams.get(QUERY_KEY);

  const writeUrl = (value) => {
    const url = new URL(window.location.href);
    url.searchParams.set(QUERY_KEY, value);
    window.history.replaceState(window.history.state, "", url);
  };

  const enginesInUse = () => {
    const found = new Set();
    if (!pattern) return found;
    for (const block of blocks()) {
      for (const [match] of block.textContent.matchAll(pattern)) {
        const entry = byText.get(match);
        if (entry) found.add(entry.engine);
      }
    }
    return found;
  };

  const candidates = (requires) => {
    const engines = [...enginesInUse()];
    if (!engines.length) return [];
    return Object.keys(models).filter((id) => {
      const model = models[id];
      if (requires && !(model.inputs || []).includes(requires)) return false;
      return engines.every((engine) => (model.ids || {})[engine]);
    });
  };

  const detect = () => {
    if (!pattern) return null;
    for (const block of blocks()) {
      for (const [match] of block.textContent.matchAll(pattern)) {
        const entry = byText.get(match);
        if (entry) return entry.model;
      }
    }
    return null;
  };

  const rewriteNode = (node, target) => {
    if (!pattern) return;
    const original = node.__gemmaOriginal ?? node.nodeValue;
    const next = original.replace(pattern, (match) => {
      const entry = byText.get(match);
      return (entry && models[target]?.ids?.[entry.engine]) || match;
    });
    if (next === node.nodeValue) return;
    node.__gemmaOriginal = original;
    node.nodeValue = next;
  };

  const applyRequires = (target) => {
    const inputs = models[target]?.inputs || [];
    const article = document.querySelector(".md-content__inner");
    if (article) {
      const walker = document.createTreeWalker(article, NodeFilter.SHOW_COMMENT);
      const startRe = /^\s*notebook:requires-start\s+([a-z0-9_-]+)\s*$/i;
      const endRe = /^\s*notebook:requires-end\s*$/i;
      const starts = [];
      let comment;
      while ((comment = walker.nextNode())) {
        const match = comment.nodeValue.match(startRe);
        if (match) starts.push({ node: comment, modality: match[1].toLowerCase() });
      }
      for (const { node, modality } of starts) {
        const supported = inputs.includes(modality);
        let cur = node.nextSibling;
        while (cur) {
          if (cur.nodeType === Node.COMMENT_NODE && endRe.test(cur.nodeValue)) break;
          if (cur.nodeType === Node.ELEMENT_NODE) {
            cur.hidden = !supported;
            cur.style.display = supported ? "" : "none";
            if (cur.id && /^H[1-6]$/.test(cur.tagName)) {
              const targetHash = `#${cur.id}`;
              for (const link of document.querySelectorAll(".md-nav--secondary a.md-nav__link")) {
                if (link.hash !== targetHash) continue;
                const item = link.closest(".md-nav__item");
                if (item) {
                  item.hidden = !supported;
                  item.style.display = supported ? "" : "none";
                }
              }
            }
          }
          cur = cur.nextSibling;
        }
      }
    }
    for (const el of document.querySelectorAll("[data-requires]:not(.page-context)")) {
      const modality = el.dataset.requires?.trim().toLowerCase();
      if (!modality) continue;
      const supported = inputs.includes(modality);
      el.hidden = !supported;
      el.style.display = supported ? "" : "none";
    }
  };

  const apply = (target) => {
    if (!models[target]) return;
    for (const block of blocks()) {
      const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) rewriteNode(node, target);
    }
    applyRequires(target);
  };

  const positionMenu = (picker) => {
    const menu = picker.querySelector(".page-context__picker-menu");
    const strip = picker.closest(".page-context");
    if (!menu || menu.hidden || !strip) return;
    menu.style.left = "";
    menu.style.right = "";
    const stripRect = strip.getBoundingClientRect();
    const pickerRect = picker.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    if (menuRect.right > stripRect.right) {
      if (pickerRect.right - menuRect.width >= stripRect.left) {
        menu.style.left = "auto";
        menu.style.right = "0";
      } else {
        const offset = Math.max(
          stripRect.left - pickerRect.left,
          stripRect.right - pickerRect.left - menuRect.width
        );
        menu.style.left = `${offset}px`;
        menu.style.right = "auto";
      }
    }
  };

  const closePicker = (picker, returnFocus = false) => {
    const trigger = picker.querySelector(".page-context__picker-trigger");
    const menu = picker.querySelector(".page-context__picker-menu");
    trigger.setAttribute("aria-expanded", "false");
    menu.hidden = true;
    if (returnFocus) trigger.focus();
  };

  const updateColab = (target) => {
    const colabLink = document.querySelector(".page-context__colab");
    if (!colabLink) return;
    const slug = `${target.replace("/", "-")}`;
    colabLink.href = colabLink.href.replace(/-gemma-4-[a-z0-9-]+\.ipynb/, `-${slug}.ipynb`);
  };

  const choose = (picker, target) => {
    const trigger = picker.querySelector(".page-context__picker-trigger");
    trigger.querySelector(".page-context__picker-name").textContent = displayName(target);
    trigger.dataset.gemmaModel = target;
    trigger.setAttribute("aria-label", `Model used by this page's examples: ${displayName(target)}`);
    for (const option of picker.querySelectorAll("[role=option]")) {
      option.setAttribute("aria-selected", String(option.dataset.value === target));
    }
    updateColab(target);
  };

  const buildControl = (strip, available, initial, unavailable) => {
    const item = document.createElement("span");
    item.className = "page-context__item";

    const label = document.createElement("span");
    label.className = "page-context__label";
    label.textContent = "Examples";

    const picker = document.createElement("span");
    picker.className = "page-context__picker";

    const menuId = `gemma-model-menu-${Math.random().toString(36).slice(2, 9)}`;
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "page-context__picker-trigger";
    trigger.dataset.gemmaModel = initial;
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", menuId);
    trigger.innerHTML = '<span class="page-context__picker-name"></span>';

    const menu = document.createElement("span");
    menu.className = "page-context__picker-menu";
    menu.id = menuId;
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", "Model used by this page's examples");
    menu.hidden = true;

    if (unavailable && models[unavailable]) {
      const notice = document.createElement("span");
      notice.className = "page-context__picker-notice";
      notice.textContent = `${displayName(unavailable)} isn't available for this page.`;
      menu.appendChild(notice);
    }

    for (const id of available) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "page-context__picker-option";
      option.dataset.value = id;
      option.setAttribute("role", "option");
      option.innerHTML =
        `<span class="page-context__picker-option-name">${displayName(id)}</span>` +
        `<span class="page-context__picker-option-meta">${models[id].platform} · ${models[id].context} context</span>`;
      option.addEventListener("click", () => {
        choose(picker, id);
        writeStored(id);
        writeUrl(id);
        apply(id);
        closePicker(picker, true);
      });
      option.addEventListener("keydown", (event) => {
        const options = [...menu.querySelectorAll("[role=option]")];
        const current = options.indexOf(option);
        const destinations = {
          ArrowDown: options[(current + 1) % options.length],
          ArrowUp: options[(current - 1 + options.length) % options.length],
          Home: options[0],
          End: options.at(-1),
        };
        if (destinations[event.key]) {
          event.preventDefault();
          destinations[event.key].focus();
        } else if (event.key === "Escape") {
          event.preventDefault();
          closePicker(picker, true);
        }
      });
      menu.appendChild(option);
    }

    trigger.addEventListener("click", () => {
      const opening = menu.hidden;
      document.querySelectorAll(".page-context__picker").forEach((other) => closePicker(other));
      menu.hidden = !opening;
      trigger.setAttribute("aria-expanded", String(opening));
      if (opening) positionMenu(picker);
    });
    trigger.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closePicker(picker, true);
        return;
      }
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      positionMenu(picker);
      const options = [...menu.querySelectorAll("[role=option]")];
      (event.key === "ArrowDown" ? options[0] : options.at(-1))?.focus();
    });

    picker.append(trigger, menu);
    item.append(label, picker);
    const trailingItem = strip.querySelector(".page-context__item--colab, .page-context__item--markdown");
    if (trailingItem) strip.insertBefore(item, trailingItem);
    else strip.appendChild(item);
    choose(picker, initial);
  };

  const init = () => {
    const strip = document.querySelector(".page-context");
    const requested = readUrl() || readStored();

    if (!strip || document.querySelector("[data-gemma-model]")) return;

    const available = candidates(strip.dataset.requires || "");
    if (available.length < 2) {
      if (requested && available.includes(requested)) {
        apply(requested);
        updateColab(requested);
      }
      return;
    }

    const detected = detect();
    const initial = available.includes(requested)
      ? requested
      : available.includes(detected)
        ? detected
        : available[0];

    buildControl(strip, available, initial, requested && !available.includes(requested) ? requested : null);
    apply(initial);
  };

  let prevTabState = null;

  const syncTabIndicator = () => {
    const list = document.querySelector(".md-tabs__list");
    if (!list) return;
    const activeLink = list.querySelector(".md-tabs__item--active > .md-tabs__link, .md-tabs__link--active");
    if (!activeLink) return;

    const listRect = list.getBoundingClientRect();
    const linkRect = activeLink.getBoundingClientRect();
    const nextLeft = linkRect.left - listRect.left;
    const nextWidth = linkRect.width;
    const nextHeight = linkRect.height;

    let indicator = list.querySelector(".gemma-tab-indicator");
    if (!indicator) {
      indicator = document.createElement("span");
      indicator.className = "gemma-tab-indicator";
      list.appendChild(indicator);
    }
    list.classList.add("has-sliding-indicator");
    indicator.style.height = `${nextHeight}px`;

    if (prevTabState && (Math.abs(prevTabState.left - nextLeft) > 1 || Math.abs(prevTabState.width - nextWidth) > 1)) {
      indicator.style.transition = "none";
      indicator.style.transform = `translate3d(${prevTabState.left}px, -50%, 0)`;
      indicator.style.width = `${prevTabState.width}px`;
      void indicator.offsetWidth;
      requestAnimationFrame(() => {
        indicator.style.transition = "";
        indicator.style.transform = `translate3d(${nextLeft}px, -50%, 0)`;
        indicator.style.width = `${nextWidth}px`;
      });
    } else {
      indicator.style.transform = `translate3d(${nextLeft}px, -50%, 0)`;
      indicator.style.width = `${nextWidth}px`;
    }

    prevTabState = { left: nextLeft, width: nextWidth };
  };

  document.addEventListener("click", (event) => {
    if (event.target.closest?.(".md-search__button, [data-md-component=search]")) {
      setTimeout(initSearchModalDefaults, 20);
      setTimeout(initSearchModalDefaults, 150);
    }
    const tabLink = event.target.closest?.(".md-tabs__link");
    if (tabLink) {
      const list = tabLink.closest(".md-tabs__list");
      const currentActive = list?.querySelector(".md-tabs__item--active > .md-tabs__link, .md-tabs__link--active");
      if (list && currentActive) {
        const listRect = list.getBoundingClientRect();
        const linkRect = currentActive.getBoundingClientRect();
        prevTabState = { left: linkRect.left - listRect.left, width: linkRect.width };
      }
    }
    for (const picker of document.querySelectorAll(".page-context__picker")) {
      if (!picker.contains(event.target)) closePicker(picker);
    }
    const selectBtn = event.target.closest?.(".md-code__button[data-md-type=select]");
    if (selectBtn) {
      window.getSelection()?.removeAllRanges();
      const pre = selectBtn.closest("pre");
      if (pre) {
        for (const hll of pre.querySelectorAll(".hll")) {
          hll.replaceWith(...hll.childNodes);
        }
      }
      if (window.location.hash.startsWith("#__codelineno-")) {
        window.history.replaceState(
          window.history.state,
          "",
          window.location.pathname + window.location.search
        );
      }
    }
  });

  document.addEventListener(
    "mousedown",
    (event) => {
      if (event.button !== 0 || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
      const lineSpan = event.target.closest?.(".md-code__content > span[id^='__span']");
      if (!lineSpan) return;
      const code = lineSpan.parentElement;
      const pre = code?.closest("pre");
      const selectBtn = pre?.querySelector(".md-code__button[data-md-type=select]");
      if (!selectBtn?.classList.contains("md-code__button--active")) return;
      const selected = code.querySelectorAll(":scope > span[id^='__span']:has(.hll:not(.select))");
      if (selected.length === 1 && selected[0] === lineSpan) {
        event.preventDefault();
        event.stopPropagation();
        for (const hll of code.querySelectorAll(".hll")) {
          hll.replaceWith(...hll.childNodes);
        }
        if (window.location.hash.startsWith("#__codelineno-")) {
          window.history.replaceState(
            window.history.state,
            "",
            window.location.pathname + window.location.search
          );
        }
      }
    },
    true
  );

  window.addEventListener("keydown", (event) => {
    if (((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") || event.key === "/") {
      setTimeout(initSearchModalDefaults, 20);
      setTimeout(initSearchModalDefaults, 150);
    }
  });
  window.addEventListener("resize", () => {
    syncTabIndicator();
    document.querySelectorAll(".page-context__picker").forEach(positionMenu);
  });

  const DEFAULT_SEARCH_TOPICS = [
    {
      title: "Quickstart with Gemma 4",
      path: ["Get started", "Quickstart"],
      url: "get-started/",
      excerpt: "Choose a Gemma 4 model size (E2B to 31B) and run your first inference locally or in Google Colab."
    },
    {
      title: "Gemma 4 model overview & benchmarks",
      path: ["Models", "Gemma 4"],
      url: "models/gemma-4/",
      excerpt: "Architecture, context windows, multimodal inputs, and benchmark results across the Gemma 4 family."
    },
    {
      title: "Run Gemma locally with Ollama",
      path: ["Run", "Local"],
      url: "deployment/ollama/",
      excerpt: "Pull and run quantized Gemma 4 checkpoints on macOS, Linux, or Windows with a single command."
    },
    {
      title: "Fine-tune Gemma with QLoRA",
      path: ["Fine-tune", "Guides"],
      url: "tuning/qlora/",
      excerpt: "Memory-efficient 4-bit LoRA fine-tuning with Hugging Face TRL and PEFT on a single GPU."
    },
    {
      title: "Multimodal vision, audio, and video",
      path: ["Get started", "Multimodal"],
      url: "get-started/capabilities/vision/",
      excerpt: "Pass images, multi-frame video, and audio inputs to Gemma 4 and control token budgets."
    },
    {
      title: "Function calling & tool use",
      path: ["Get started", "Text"],
      url: "get-started/capabilities/function-calling/",
      excerpt: "Define structured tool schemas and parse model tool calls in multi-turn agent loops."
    }
  ];

  const resolveSiteBase = () => {
    try {
      const cfg = document.querySelector("#__config");
      if (cfg) {
        const parsed = JSON.parse(cfg.textContent);
        const raw = new URL(parsed.base || ".", window.location.href).toString();
        return raw.endsWith("/") ? raw : `${raw}/`;
      }
    } catch {
      // Fallback to root
    }
    return new URL("/", window.location.href).toString();
  };

  const initSearchModalDefaults = () => {
    let shadowRoot = null;
    for (const div of document.querySelectorAll("body > div")) {
      if (div.shadowRoot && div.shadowRoot.querySelector(".z")) {
        shadowRoot = div.shadowRoot;
        break;
      }
    }
    if (!shadowRoot || shadowRoot.__gemmaDefaultsInit) return;
    const resultsPane = shadowRoot.querySelector(".z");
    const input = shadowRoot.querySelector(".f input");
    if (!resultsPane || !input) return;
    shadowRoot.__gemmaDefaultsInit = true;

    const base = resolveSiteBase();
    const wrapper = document.createElement("div");
    wrapper.className = "gemma-search-defaults";
    wrapper.innerHTML =
      `<h3 class="B">Suggested topics</h3>` +
      `<ol class="b">` +
      DEFAULT_SEARCH_TOPICS.map((item, idx) => {
        const href = new URL(item.url, base).toString();
        const crumbs = item.path.map((p) => `<li>${p}</li>`).join("");
        return (
          `<li>` +
          `<a class="i${idx === 0 ? " h" : ""}" href="${href}">` +
          `<div class="C">` +
          `<div class="D"><menu class="t">${crumbs}</menu></div>` +
          `<h2 class="x">${item.title}</h2>` +
          `<div class="u">${item.excerpt}</div>` +
          `</div>` +
          `</a>` +
          `</li>`
        );
      }).join("") +
      `</ol>`;

    resultsPane.insertBefore(wrapper, resultsPane.firstChild);

    let activeIdx = 0;
    const links = [...wrapper.querySelectorAll("a.i")];
    const closeModal = () => shadowRoot.querySelector(".k button")?.click();
    const setHighlight = (next) => {
      links.forEach((a, i) => a.classList.toggle("h", i === next));
      activeIdx = next;
    };
    links.forEach((a, i) => {
      a.addEventListener("pointerenter", () => setHighlight(i));
      a.addEventListener("click", () => closeModal());
    });

    const syncVisibility = () => {
      const hasQuery = input.value.trim().length > 0;
      wrapper.style.display = hasQuery ? "none" : "";
    };

    input.addEventListener("input", syncVisibility);
    input.addEventListener("focus", syncVisibility);
    input.addEventListener("keydown", (e) => {
      if (input.value.trim().length > 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        e.stopPropagation();
        setHighlight((activeIdx + 1) % links.length);
        links[activeIdx]?.scrollIntoView({ block: "nearest" });
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        e.stopPropagation();
        setHighlight((activeIdx - 1 + links.length) % links.length);
        links[activeIdx]?.scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && links[activeIdx]) {
        e.preventDefault();
        e.stopPropagation();
        const dest = links[activeIdx].href;
        closeModal();
        window.location.href = dest;
      }
    });

    const observer = new MutationObserver(syncVisibility);
    observer.observe(resultsPane, { childList: true, subtree: true });
    syncVisibility();
  };

  const onPageReady = () => {
    init();
    syncTabIndicator();
    initSearchModalDefaults();
    setTimeout(initSearchModalDefaults, 250);
    if (typeof window.__initGemmaHome === "function") {
      window.__initGemmaHome();
    }
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", onPageReady);
  else onPageReady();
  if (window.document$?.subscribe) window.document$.subscribe(onPageReady);
})();
