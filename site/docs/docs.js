document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("pre").forEach((pre) => {
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
      try {
        if (!navigator.clipboard) throw new Error("clipboard unavailable");
        await navigator.clipboard.writeText(code.innerText);
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

  const reveals = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("in-view"));
    document.documentElement.dataset.siteReady = "1";
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
    { threshold: 0.1 }
  );
  reveals.forEach((el) => io.observe(el));
  document.documentElement.dataset.siteReady = "1";
});
