document.addEventListener("DOMContentLoaded", () => {
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.getElementById("nav-links");
  const mobileNav = window.matchMedia("(max-width: 900px)");

  if (navToggle && navLinks) {
    let navFocusTimer = 0;

    const setNav = (open, returnFocus = false) => {
      window.clearTimeout(navFocusTimer);
      const isMobile = mobileNav.matches;
      document.body.classList.toggle("nav-open", open);
      navToggle.setAttribute("aria-expanded", String(open));
      navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");

      if (isMobile) {
        navLinks.inert = !open;
        navLinks.setAttribute("aria-hidden", String(!open));
      } else {
        navLinks.inert = false;
        navLinks.removeAttribute("aria-hidden");
      }

      if (open && isMobile) {
        const first = navLinks.querySelector("a");
        // The menu transitions from visibility:hidden; focusing before the
        // style has committed silently fails, leaving focus on the button.
        window.clearTimeout(navFocusTimer);
        if (first) {
          navFocusTimer = window.setTimeout(() => {
            if (document.body.classList.contains("nav-open")) {
              first.focus({ preventScroll: true });
            }
          }, 240);
        }
      } else if (returnFocus && isMobile) {
        navToggle.focus({ preventScroll: true });
      }
    };

    const syncNav = () => {
      if (mobileNav.matches) {
        setNav(document.body.classList.contains("nav-open"));
      } else {
        document.body.classList.remove("nav-open");
        navToggle.setAttribute("aria-expanded", "false");
        navToggle.setAttribute("aria-label", "Open navigation");
        navLinks.inert = false;
        navLinks.removeAttribute("aria-hidden");
      }
    };

    navToggle.addEventListener("click", () => {
      setNav(!document.body.classList.contains("nav-open"));
    });

    navLinks.addEventListener("click", (e) => {
      if (e.target.closest("a")) setNav(false);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && document.body.classList.contains("nav-open")) {
        setNav(false, true);
      }
    });

    document.addEventListener("pointerdown", (e) => {
      if (!mobileNav.matches || !document.body.classList.contains("nav-open")) return;
      if (e.target.closest("#nav-links, .nav-toggle")) return;
      setNav(false, true);
    });

    if (mobileNav.addEventListener) mobileNav.addEventListener("change", syncNav);
    else mobileNav.addListener(syncNav);
    syncNav();
  }

  document.querySelectorAll("pre.code").forEach((pre) => {
    const code = pre.querySelector("code");
    if (!code) return;

    const btn = document.createElement("button");
    btn.className = "copy";
    btn.type = "button";
    btn.textContent = "copy";
    btn.setAttribute("aria-label", "Copy code to clipboard");

    const status = document.createElement("span");
    status.className = "copy-status";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");

    const reset = () => {
      window.setTimeout(() => {
        btn.textContent = "copy";
        btn.classList.remove("done");
        status.textContent = "";
      }, 1600);
    };

    btn.addEventListener("click", async () => {
      const text = code.innerText;
      try {
        if (!navigator.clipboard) throw new Error("clipboard unavailable");
        await navigator.clipboard.writeText(text);
        btn.textContent = "copied";
        btn.classList.add("done");
        status.textContent = "Copied to clipboard";
        reset();
      } catch {
        btn.textContent = "copy failed";
        btn.classList.remove("done");
        status.textContent = "Select the code and copy it manually";
        reset();
      }
    });

    pre.appendChild(status);
    pre.appendChild(btn);
  });

  // The stagger is a ripple, not a queue: without a cap the fortieth element
  // waits four seconds for its turn, and a page that scrolls fast looks
  // broken rather than choreographed. The cap is also the window in which
  // text is still fading in — keep it short enough that a number is never
  // sitting at half opacity while somebody is trying to read it.
  const stagger = (els, base = 0.08, cap = 0.24) =>
    els.forEach((el, i) =>
      el.style.setProperty("--d", `${Math.min(base * i, cap)}s`),
    );

  const hero = document.querySelectorAll(".hero .reveal");
  stagger(hero, 0.12, 0.42);
  hero.forEach((el) => el.classList.add("in-view"));

  const groups = document.querySelectorAll(
    ".cards .card, .steps .reveal, .stats .stat, #benchmarks tbody .reveal, .arch .reveal, #pricing .reveal, #watch .reveal, .cta-final .reveal, .code.reveal"
  );
  stagger(groups, 0.08, 0.24);

  const tabs = Array.from(document.querySelectorAll(".bench-tab"));
  const panels = Array.from(document.querySelectorAll(".bench-panel"));

  if (tabs.length) {
    const activateTab = (tab) => {
      tabs.forEach((t) => {
        const selected = t === tab;
        t.classList.toggle("active", selected);
        t.setAttribute("aria-selected", String(selected));
        t.tabIndex = selected ? 0 : -1;
      });
      panels.forEach((panel) => {
        const selected = panel.id === tab.dataset.model;
        panel.classList.toggle("active", selected);
        panel.hidden = !selected;
      });
      const shown = document.getElementById(tab.dataset.model);
      if (shown) {
        shown.querySelectorAll(".reveal").forEach((el, i) => {
          el.style.setProperty("--d", `${Math.min(0.08 * i, 0.24)}s`);
          el.classList.add("in-view");
        });
      }
    };

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => activateTab(tab));
      tab.addEventListener("keydown", (event) => {
        let next = null;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = tabs.length - 1;
        if (next === null) return;
        event.preventDefault();
        tabs[next].focus();
        activateTab(tabs[next]);
      });
      tab.tabIndex = tab.classList.contains("active") ? 0 : -1;
    });

    const initial = tabs.find((tab) => tab.classList.contains("active")) || tabs[0];
    activateTab(initial);
  }

  // The video is a poster until it is asked for, and it is loaded from
  // YouTube's no-cookie host. No third-party request before the click.
  document.querySelectorAll("[data-video]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const frame = document.createElement("iframe");
      frame.src = `https://www.youtube-nocookie.com/embed/${btn.dataset.video}?autoplay=1&rel=0`;
      frame.title = "Toki Recur — see it in action";
      frame.loading = "lazy";
      frame.allow =
        "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
      frame.referrerPolicy = "strict-origin-when-cross-origin";
      frame.allowFullscreen = true;
      btn.replaceWith(frame);
    });
  });

  // Every reveal on the page, not just the ones a selector happened to
  // mention: a section heading that waits for a rule that forgot it is
  // a heading nobody ever sees.
  const reveals = document.querySelectorAll(".reveal");

  if (!("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("in-view"));
    document.documentElement.dataset.siteReady = "1";
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        // Anything at or above the fold is revealed, even when the observer
        // says it is not intersecting: a jump to an anchor, a restored scroll
        // position, or a page that loads halfway down leaves those elements
        // behind the viewport, and a heading nobody can see is a heading
        // that does not exist.
        const past = entry.boundingClientRect.top < window.innerHeight;
        if (entry.isIntersecting || past) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.05 }
  );
  reveals.forEach((el) => io.observe(el));
  document.documentElement.dataset.siteReady = "1";
});

/* ---------------------------------------------------------------------------
   The two bits of motion that are not reveals: the nav knows where you are,
   and the numbers count up once, when they first come into view.
   --------------------------------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
    if (reduceMotion) return;
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
  if (numbers.length && "IntersectionObserver" in window && !reduceMotion) {
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
