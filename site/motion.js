/* The small motions that make a page feel made rather than assembled.

   Everything here is transform- or opacity-only, so it never triggers
   layout, and every one of them stands down when the visitor asks for
   reduced motion. */

(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ── the cards follow the pointer ────────────────────────────────────
     A card that lights up where the cursor is reads as a physical object.
     The position is written as two custom properties and the gradient is
     painted by the browser, so the cost is one style recalculation per
     frame at most. */
  const cards = Array.from(document.querySelectorAll(".l-panel, .feature"));
  if (cards.length && !reduceMotion.matches) {
    let queued = false;
    let pending = null;

    const paint = () => {
      queued = false;
      if (!pending) return;
      const { card, x, y } = pending;
      const box = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${((x - box.left) / box.width) * 100}%`);
      card.style.setProperty("--my", `${((y - box.top) / box.height) * 100}%`);
      pending = null;
    };

    cards.forEach((card) => {
      card.addEventListener(
        "pointermove",
        (event) => {
          if (event.pointerType === "touch") return;
          pending = { card, x: event.clientX, y: event.clientY };
          if (!queued) {
            queued = true;
            requestAnimationFrame(paint);
          }
        },
        { passive: true },
      );
    });
  }

  /* ── the hero drifts as you leave it ─────────────────────────────────
     A parallax on the live window and a slight fade on the copy. The
     transform is written once per frame, never per scroll event. */
  const hero = document.querySelector(".l-hero");
  const heroVisual = hero && hero.querySelector(".l-hero-media");
  const heroCopy = hero && hero.querySelector(".l-hero-copy");
  if (hero && heroVisual && heroCopy && !reduceMotion.matches) {
    let ticking = false;

    const update = () => {
      ticking = false;
      const offset = Math.min(window.scrollY, window.innerHeight);
      if (offset > window.innerHeight * 1.2) return;
      const progress = offset / Math.max(window.innerHeight, 1);
      heroVisual.style.transform = `translate3d(0, ${progress * 48}px, 0)`;
      heroCopy.style.transform = `translate3d(0, ${progress * 22}px, 0)`;
      heroCopy.style.opacity = String(Math.max(0, 1 - progress * 0.85));
    };

    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          ticking = true;
          requestAnimationFrame(update);
        }
      },
      { passive: true },
    );
    update();
  }
})();
