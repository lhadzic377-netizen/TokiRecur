/* The prefix matcher, drawn.

   A request arrives as a row of tokens. The matcher walks it against the
   prefix already in the KV cache. While the tokens match, they are
   replayed from memory; the first token that differs ends the prefix, and
   from there the model computes the rest.

   This is the mechanism the product is about, so it is drawn rather than
   described — and the numbers it counts are the measured ones.
*/

(() => {
  const stage = document.querySelector(".mech-stage");
  const cellsHost = stage && stage.querySelector("[data-mech-cells]");
  const scan = stage && stage.querySelector("[data-mech-scan]");
  const replayedOut = stage && stage.querySelector("[data-mech-replayed]");
  const computedOut = stage && stage.querySelector("[data-mech-computed]");
  if (!cellsHost || !scan) return;

  const CELLS = 30;
  // The prefix matches for 26 tokens, then the tail is new work.
  const MATCH = 26;
  const STEP_MS = 62;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const cells = [];
  for (let i = 0; i < CELLS; i += 1) {
    const cell = document.createElement("i");
    cell.className = "mech-cell";
    cellsHost.appendChild(cell);
    cells.push(cell);
  }

  // The matcher sits over the cells and advances one cell width at a time.
  // Measuring the pitch (cell + gap) keeps the line exactly on the token it
  // is comparing, whatever the width of the stage.
  const pitch = () => {
    const first = cells[0];
    if (!first) return 0;
    const styles = getComputedStyle(cellsHost);
    const gap = parseFloat(styles.columnGap || styles.gap || "0") || 0;
    return first.getBoundingClientRect().width + gap;
  };

  const setReadout = (replayed, computed) => {
    if (replayedOut) replayedOut.textContent = String(replayed);
    if (computedOut) computedOut.textContent = String(computed);
  };

  const paint = (index, state) => {
    for (let i = 0; i < CELLS; i += 1) {
      const cell = cells[i];
      if (state === "reset") {
        cell.className = "mech-cell";
        continue;
      }
      if (i < index) cell.className = "mech-cell is-replay";
      else if (i === index) cell.className = "mech-cell is-current";
      else cell.className = "mech-cell";
    }
  };

  const finalState = () => {
    for (let i = 0; i < CELLS; i += 1) {
      cells[i].className = i < MATCH ? "mech-cell is-replay" : "mech-cell is-compute";
    }
    setReadout(MATCH, CELLS - MATCH);
  };

  let timer = 0;
  let index = 0;
  let running = false;

  const tick = () => {
    index += 1;
    if (index <= MATCH) {
      cells[index - 1].className = "mech-cell is-replay";
      setReadout(index, 0);
    } else {
      // The mismatch: everything from here on has to be computed.
      for (let i = MATCH; i < CELLS; i += 1) {
        cells[i].className = "mech-cell is-compute";
      }
      setReadout(MATCH, CELLS - MATCH);
      index = 0;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        paint(0, "reset");
        setReadout(0, 0);
        timer = window.setTimeout(tick, 420);
      }, 1500);
      return;
    }
    scan.style.transform = `translateX(${index * pitch()}px)`;
    timer = window.setTimeout(tick, STEP_MS);
  };

  const start = () => {
    if (running || reduceMotion.matches) return;
    running = true;
    index = 0;
    paint(0, "reset");
    setReadout(0, 0);
    timer = window.setTimeout(tick, 260);
  };

  const stop = () => {
    running = false;
    window.clearTimeout(timer);
  };

  if (reduceMotion.matches) {
    finalState();
  } else {
    finalState();
    // Wait until the band is on screen: nobody should pay for an animation
    // they cannot see.
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              start();
              observer.disconnect();
            }
          });
        },
        { threshold: 0.35 },
      );
      observer.observe(stage);
    } else {
      start();
    }
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
      else start();
    });
  }
})();
