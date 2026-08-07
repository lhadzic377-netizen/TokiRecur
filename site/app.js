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
