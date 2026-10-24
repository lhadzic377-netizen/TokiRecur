document.addEventListener("DOMContentLoaded", () => {
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.getElementById("nav-links");
  if (navToggle && navLinks) {
    const setNav = (open) => {
      document.body.classList.toggle("nav-open", open);
      navToggle.setAttribute("aria-expanded", String(open));
    };
    navToggle.addEventListener("click", () => {
      setNav(!document.body.classList.contains("nav-open"));
    });
    navLinks.addEventListener("click", (e) => {
      if (e.target.closest("a")) setNav(false);
    });
  }

  document.querySelectorAll("pre.code").forEach((pre) => {
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.type = "button";
    btn.textContent = "copy";
    btn.addEventListener("click", () => {
      const text = pre.querySelector("code").innerText;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "copied";
        btn.classList.add("done");
        setTimeout(() => {
          btn.textContent = "copy";
          btn.classList.remove("done");
        }, 1500);
      });
    });
    pre.appendChild(btn);
  });

  const stagger = (els, base = 0.08) =>
    els.forEach((el, i) => el.style.setProperty("--d", `${base * i}s`));

  const hero = document.querySelectorAll(".hero .reveal");
  stagger(hero, 0.12);
  hero.forEach((el) => el.classList.add("in-view"));

  const groups = document.querySelectorAll(
    ".cards .card, .steps .reveal, .stats .stat, #benchmarks tbody .reveal, .arch .reveal, #pricing .reveal, #watch .reveal, .cta-final .reveal, .code.reveal"
  );
  stagger(groups, 0.1);

  const tabs = document.querySelectorAll(".bench-tab");
  if (tabs.length) {
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => {
          t.classList.remove("active");
          t.setAttribute("aria-selected", "false");
        });
        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");
        document.querySelectorAll(".bench-panel").forEach((panel) => {
          panel.classList.toggle("active", panel.id === tab.dataset.model);
        });
        const shown = document.getElementById(tab.dataset.model);
        if (shown) {
          shown.querySelectorAll(".reveal").forEach((el, i) => {
            el.style.setProperty("--d", `${0.1 * i}s`);
            el.classList.add("in-view");
          });
        }
      });
    });
  }

  // Every reveal on the page, not just the ones a selector happened to
  // mention: a section heading that waits for a rule that forgot it is
  // a heading nobody ever sees.
  const reveals = document.querySelectorAll(".reveal");

  if (!("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("in-view"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.05 }
  );
  reveals.forEach((el) => io.observe(el));
});

/* ---------------------------------------------------------------------------
   The two bits of motion that are not reveals: the nav knows where you are,
   and the numbers count up once, when they first come into view.
   --------------------------------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
  const navLinks = Array.from(
    document.querySelectorAll("#nav-links a[href^='#']")
  ).filter((a) => a.getAttribute("href").length > 1);

  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);

  if (sections.length) {
    // The indicator follows the section you are actually in. An observer
    // fires on the edge of a section, so it lit "Architecture" while the
    // cards were on screen.
    const pick = () => {
      const line = window.scrollY + window.innerHeight * 0.4;
      let active = null;
      for (const s of sections) {
        if (s.offsetTop <= line) active = s;
        else break;
      }
      navLinks.forEach((a) => {
        a.classList.toggle(
          "active",
          Boolean(active) && a.getAttribute("href") === `#${active.id}`
        );
      });
    };
    pick();
    window.addEventListener("scroll", pick, { passive: true });
  }

  // The bar is a floating layer: it only grows a shadow once content is
  // actually passing under it.
  const topbar = document.querySelector(".topbar");
  if (topbar) {
    const mark = () => topbar.classList.toggle("scrolled", window.scrollY > 8);
    mark();
    window.addEventListener("scroll", mark, { passive: true });
  }

  // "+96.2%" counts to 96.2; "0" counts to nothing but still settles.
  const countUp = (el) => {
    const raw = el.textContent.trim();
    const match = raw.match(/(-?\d+(?:\.\d+)?)/);
    if (!match) return;
    const target = parseFloat(match[1]);
    if (!isFinite(target) || target === 0) return;
    const decimals = (match[1].split(".")[1] || "").length;
    const prefix = raw.slice(0, match.index);
    const suffix = raw.slice(match.index + match[1].length);
    const started = performance.now();
    const duration = 1100;
    const tick = (now) => {
      const p = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = `${prefix}${(target * eased).toFixed(decimals)}${suffix}`;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = raw;
    };
    requestAnimationFrame(tick);
  };

  const numbers = document.querySelectorAll(".stats .num");
  if (numbers.length && "IntersectionObserver" in window) {
    const counter = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          countUp(entry.target);
          counter.unobserve(entry.target);
        });
      },
      { threshold: 0.6 }
    );
    numbers.forEach((n) => counter.observe(n));
  }
});
