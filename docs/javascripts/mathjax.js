// MathJax configuration for pymdownx.arithmatex (generic mode).
// Must load BEFORE the MathJax library (see extra_javascript in zensical.toml).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// With navigation.instant, pages are swapped in without a full reload, so
// MathJax's startup typeset never sees the new content. Re-typeset on every
// page change emitted by the theme's document$ observable.
if (window.document$ && typeof window.document$.subscribe === "function") {
  window.document$.subscribe(() => {
    if (!window.MathJax || !MathJax.startup || !MathJax.typesetPromise) return;
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
