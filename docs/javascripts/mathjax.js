// Configuration for pymdownx.arithmatex in "generic" mode, which emits
// \(...\) and \[...\] wrapped in .arithmatex elements. Without this, MathJax
// also rescans code blocks and can double-process them.
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
