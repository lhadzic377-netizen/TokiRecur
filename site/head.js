document.documentElement.classList.add("js-ready");

// If app.js never initialises (a blocked script, a syntax error, an old
// browser), do not leave the page as a set of invisible sections. The timer
// is a failure path, not the normal reveal mechanism.
window.setTimeout(function () {
  if (document.documentElement.dataset.siteReady) return;
  document.querySelectorAll(".reveal:not(.in-view)").forEach(function (el) {
    el.classList.add("in-view");
  });
}, 2500);
