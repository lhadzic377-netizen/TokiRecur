/* ---------------------------------------------------------------------------
   The toki launcher, live.

   The launcher is the front door of the product, so the front door of the
   site is the launcher itself: the same rows, the same colours, the same
   keys. Nothing here talks to a server. The lines follow the app's own
   rendering rules (launch_catalog.LAUNCH_APPS, the pinned model chip, the
   mode tag) so the demo cannot drift into being a picture of a launcher
   instead of the launcher.

   Keys: ↑ ↓ / j k move · 1-9 jump · m mode · → models · s settings ·
         enter launch · esc back
   --------------------------------------------------------------------------- */

(() => {
  const MODEL = "ornith-1.5-35b-a3b-gguf";
  const RAW_URL = "http://127.0.0.1:8112";
  const DAEMON_URL = "http://127.0.0.1:8811";

  /* The catalogue, in the order the launcher shows it. */
  const APPS = [
    { title: "Chat", desc: "Chat with models in the Toki TUI", cmd: `toki tui --raw-url ${RAW_URL} --daemon-url ${DAEMON_URL}`, model: true },
    { title: "Launch TokiRecur", desc: "The TokiRecur desktop app: chat, agents, voice, video", cmd: "tokirecur", model: false },
    { title: "Launch OpenCode", desc: "Anomaly's open-source coding agent", cmd: "opencode", model: true },
    { title: "Launch Hermes Agent", desc: "Self-improving AI agent built by Nous Research", cmd: "hermes", model: true },
    { title: "Launch OpenClaw", desc: "Personal AI with 100+ skills", cmd: "openclaw", model: true },
    { title: "Launch Codex", desc: "OpenAI CLI coding agent (Responses API)", cmd: "codex", model: true },
    { title: "Launch MiMo Code", desc: "Xiaomi's OpenCode-compatible coding agent", cmd: "mimo", model: true },
    { title: "Launch Qwen Code", desc: "Qwen's terminal coding agent", cmd: "qwen", model: true },
    { title: "Launch Factory Droid", desc: "Factory's AI coding agent in your terminal", cmd: "droid", model: true },
    { title: "Launch Cline CLI", desc: "Cline's terminal coding assistant", cmd: "cline", model: true },
    { title: "Launch DeepSeek Harness", desc: "DeepSeek Harness terminal agent", cmd: "dsh", model: true },
    { title: "Launch Grok Build", desc: "xAI's terminal coding agent", cmd: "grok", model: true },
    { title: "Launch Kilo Code", desc: "Kilo's terminal coding agent", cmd: "kilo", model: true },
    { title: "Launch Pi", desc: "Pi coding agent", cmd: "pi", model: true },
    { title: "Launch GitHub Copilot", desc: "GitHub Copilot CLI", cmd: "copilot", model: true },
    { title: "Launch Claude Code", desc: "Anthropic's terminal coding agent", cmd: "claude", model: true },
    { title: "Launch Kimi Code", desc: "Moonshot AI's terminal coding agent", cmd: "kimi", model: true },
    { title: "Launch OMP", desc: "Oh My Pi coding agent", cmd: "omp", model: true },
    { title: "Launch Web Dashboards", desc: "browser UIs: toki · openclaw · hermes", cmd: "toki web", model: true },
    { title: "Launch DeepSeek Harness Web", desc: "DeepSeek Harness browser UI on :3080", cmd: "dsh web", model: true },
  ];

  /* The registry, as the model picker shows it. */
  const MODELS = [
    { name: "9b", running: false },
    { name: "35b", tag: "vision", running: true },
    { name: "laguna", running: false },
    { name: "qwen3.5-0.8b-gguf", running: false },
    { name: "qwen3-4b-thinking-2507-gguf", running: false },
    { name: "qwen3-0.6b-gguf", running: false },
    { name: "kwaipilot_kat-coder-v2.5-dev-gguf", running: false },
    { name: "qwen3.5-2b-gguf", tag: "vision", running: false },
    { name: "lfm2.5-2.6b-gguf", running: false },
    { name: "ornith-1.5-35b-a3b-gguf", tag: "vision", running: true, pinned: true },
    { name: "qwen3-vl-4b", tag: "vision", running: false },
    { name: "gemma-4-e2b-it-gguf", tag: "vision", running: false },
  ];

  /* The settings pane, for the pinned model. */
  const SETTINGS = [
    ["gguf", "/models/qwen3.6-35b-a3b-ud-q4_k_m.gguf"],
    ["llama_port", "8089"],
    ["daemon_port", "8788"],
    ["ctx_size", "200000"],
    ["parallel", "1"],
    ["kv", "f16"],
    ["max_tokens", "reg:100000 · live:200000"],
    ["model_id", "qwen3.6-35b-a3b"],
    ["binary", "/opt/tokirecur/bin/llama-server"],
    ["always_on", "False"],
    ["boot", "False"],
    ["reasoning", "on"],
  ];

  const WORDMARK = [
    " _____ ___  _  _____ ",
    "|_   _/ _ \\| |/ /_ _|",
    "  | || | | | ' / | | ",
    "  | || |_| | . \\ | | ",
    "  |_| \\___/|_|\\_\\___|",
  ];

  const HINTS = {
    launchers: "↑/↓ navigate · 1-9 jump · → models · s settings",
    models: "↑/↓ navigate · 1-9 jump · enter select · esc back",
    settings: "↑/↓ navigate · enter edit · esc back",
  };

  const GUT = "  ";

  let root = null;
  let launchTimer = 0;

  const state = {
    pane: "launchers",
    index: 0,
    mode: "raw",
    model: 0,
    setting: 0,
    launched: null,
  };

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function line(cls) {
    const row = el("div", `tui-line${cls ? ` ${cls}` : ""}`);
    root.appendChild(row);
    return row;
  }

  function span(row, cls, text) {
    const node = el("span", cls, text);
    row.appendChild(node);
    return node;
  }

  /* The launchers pane, row for row with the app: the selected row carries
     the bar and its description, every other row is a name and a chip. */
  function renderLaunchers() {
    APPS.forEach((app, i) => {
      const active = i === state.index;
      const row = line(active ? "active" : null);
      row.dataset.row = String(i);
      span(row, "gutter", GUT);
      if (active) {
        span(row, "bar", "▌ ");
        span(row, "sel", app.title);
        if (app.model) {
          span(row, "pad", "  ");
          span(row, "chip", `(${MODEL})`);
        }
        const detail = line("dim");
        span(detail, null, GUT);
        span(detail, null, app.desc + (app.model ? ` · ${state.mode}` : ""));
      } else {
        span(row, "name", app.title);
        if (app.model) {
          span(row, "pad", "  ");
          span(row, "chip dim-chip", `(${MODEL})`);
        }
      }
    });
  }

  /* The model picker. */
  function renderModels() {
    const head = line("head");
    span(head, "gutter", GUT);
    span(head, "sel", "Select model");
    MODELS.forEach((model, i) => {
      const row = line(i === state.model ? "active" : null);
      row.dataset.row = String(i);
      span(row, "gutter", GUT);
      if (i === state.model) span(row, "bar", "▌ ");
      span(row, i === state.model ? "sel" : "name", model.name);
      if (model.pinned) span(row, "chip dim-chip", " << pinned");
      if (model.tag === "vision") span(row, "chip dim-chip", " [vision]");
      span(row, "dot", "  ");
      span(row, model.running ? "ok" : "off", "●");
      span(row, "dot", " ");
      span(row, model.running ? "ok" : "off", "●");
    });
  }

  /* The settings pane, grouped by model like the app's. */
  function renderSettings() {
    const head = line("head");
    span(head, "gutter", GUT);
    span(head, "sel", "Settings");
    const group = line("group");
    span(group, "gutter", GUT);
    span(group, "sel", "35b");
    SETTINGS.forEach(([key, value], i) => {
      const row = line(i === state.setting ? "active" : null);
      row.dataset.row = String(i);
      span(row, "gutter", GUT);
      if (i === state.setting) span(row, "bar", "▌ ");
      span(row, i === state.setting ? "sel" : "name", key);
      span(row, "pad", "  ");
      span(row, "value", value);
    });
  }

  /* The footer: the pinned model, the backend dots, the launch line. */
  function renderStatus() {
    const row = line("status");
    span(row, "gutter", GUT);
    span(row, "sel", `● ${MODEL}`);
    span(row, "dim", "  llama ");
    span(row, "ok", "●");
    span(row, "dim", "  daemon ");
    span(row, "ok", "●");
    if (state.launched) {
      span(row, "dim", "    ");
      span(row, "sel", `⟳ starting ${state.launched.title} …`);
    }
  }

  function renderHints() {
    const row = line("hints");
    span(row, "gutter", GUT);
    span(row, "dim", HINTS[state.pane]);
  }

  function render() {
    root.textContent = "";
    if (state.pane === "launchers") {
      WORDMARK.forEach((art) => {
        const row = line("wordmark");
        span(row, null, art);
      });
      renderLaunchers();
    } else if (state.pane === "models") {
      renderModels();
    } else {
      renderSettings();
    }
    if (state.launched) {
      const row = line("dim");
      span(row, "gutter", GUT);
      span(row, "dim", `↳ ${state.launched.cmd}`);
    }
    renderStatus();
    renderHints();
  }

  function move(delta) {
    if (state.pane === "launchers") {
      state.index = (state.index + delta + APPS.length) % APPS.length;
    } else if (state.pane === "models") {
      state.model = (state.model + delta + MODELS.length) % MODELS.length;
    } else {
      state.setting = (state.setting + delta + SETTINGS.length) % SETTINGS.length;
    }
    render();
  }

  function jump(index) {
    if (state.pane === "launchers" && index < APPS.length) state.index = index;
    else if (state.pane === "models" && index < MODELS.length) state.model = index;
    else return;
    render();
  }

  function launch() {
    if (state.pane !== "launchers") return;
    state.launched = APPS[state.index];
    window.clearTimeout(launchTimer);
    launchTimer = window.setTimeout(() => {
      state.launched = null;
      render();
    }, 2400);
    render();
  }

  function setPane(pane) {
    state.pane = pane;
    render();
  }

  function toggleMode() {
    if (state.pane !== "launchers") {
      setPane("launchers");
      return;
    }
    state.mode = state.mode === "raw" ? "managed" : "raw";
    render();
  }

  function onKey(event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key;
    let used = true;
    if (key === "ArrowDown" || key === "j") move(1);
    else if (key === "ArrowUp" || key === "k") move(-1);
    else if (key >= "1" && key <= "9") jump(Number(key) - 1);
    else if (key === "ArrowRight") setPane(state.pane === "launchers" ? "models" : state.pane);
    else if (key === "ArrowLeft" || key === "Escape") setPane("launchers");
    else if (key === "s") setPane(state.pane === "settings" ? "launchers" : "settings");
    else if (key === "m") toggleMode();
    else if (key === "Enter") launch();
    else used = false;
    if (!used) return;
    /* Tab is left alone: a keyboard visitor must be able to leave. */
    event.preventDefault();
  }

  document.querySelectorAll("[data-tui]").forEach((node) => {
    root = node;
    root.setAttribute("tabindex", "0");
    root.setAttribute("role", "application");
    root.addEventListener("keydown", onKey);
    root.addEventListener("focus", () => root.classList.add("focused"));
    root.addEventListener("blur", () => root.classList.remove("focused"));
    root.addEventListener("pointerdown", (event) => {
      root.focus({ preventScroll: true });
      const row = event.target.closest("[data-row]");
      if (!row) return;
      const index = Number(row.dataset.row);
      if (Number.isInteger(index)) jump(index);
    });
    render();
  });
})();
