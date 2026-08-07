document.addEventListener("DOMContentLoaded", () => {
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
    ".cards .card, .steps .reveal, .stats .stat, #benchmarks tbody .reveal, .arch .reveal, .cta-final .reveal"
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

  if (!("IntersectionObserver" in window)) {
    groups.forEach((el) => el.classList.add("in-view"));
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
    { threshold: 0.15 }
  );
  groups.forEach((el) => io.observe(el));
});

// ── wishlist ──────────────────────────────────────────────
// State lives only in the visitor's browser (localStorage) — no
// emails, no personal data, nothing stored server-side beyond the
// anonymous running total (a single integer).

const WL_KEY = "toki_wishlist";
const btn = document.getElementById("wishlist-btn");
const label = document.getElementById("wishlist-label");
const countEl = document.getElementById("wishlist-count");

function wlState() {
  return localStorage.getItem(WL_KEY) === "1";
}

function wlSet(on) {
  localStorage.setItem(WL_KEY, on ? "1" : "0");
  btn.setAttribute("aria-pressed", String(on));
  label.textContent = on ? "On the wishlist" : "Add to wishlist";
  btn.querySelector(".wl-icon").textContent = on ? "★" : "☆";
}

async function wlFetch(url, opts = {}) {
  try {
    const r = await fetch(url, opts);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

async function wlSyncCount() {
  const data = await wlFetch("/api/wishlist");
  if (!data || typeof data.count !== "number") return;
  countEl.hidden = false;
  countEl.textContent =
    data.count === 1 ? "1 on the wishlist" : `${data.count.toLocaleString()} on the wishlist`;
}

if (btn) {
  wlSet(wlState());
  wlSyncCount();
  btn.addEventListener("click", async () => {
    const on = !wlState();
    wlSet(on);
    const data = await wlFetch("/api/wishlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delta: on ? 1 : -1 }),
    });
    if (data && typeof data.count === "number") {
      countEl.hidden = false;
      countEl.textContent =
        data.count === 1 ? "1 on the wishlist" : `${data.count.toLocaleString()} on the wishlist`;
    }
  });
}
