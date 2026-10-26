/* ---------------------------------------------------------------------------
   toki, in the browser.

   The real terminal app, minus the part that talks to a model: the
   launcher, the settings, the model picker. Nothing here is live — the
   values are the app's own and no answer is anywhere — but the frame,
   the keys and the way a changed field says the server has to restart
   are the app's.

   The chat is not faked. The site shows it as a capture of the real
   terminal instead.

   Keys: ↑ ↓ / j k move · 1-9 jump · m mode · → models · s settings ·
         enter open · esc back
   --------------------------------------------------------------------------- */

(() => {
  const RAW_URL = "http://127.0.0.1:8112";
  const DAEMON_URL = "http://127.0.0.1:8811";

  /* Fields whose change needs llama-server restarted, as in the app. */
  const RESTART = new Set(["gguf", "ctx_size", "kv", "parallel", "binary", "llama_port", "daemon_port"]);

  const ENUMS = {
    kv: ["f16", "q8_0", "q4_0"],
    reasoning: ["on", "off", "auto"],
    always_on: ["True", "False"],
    boot: ["True", "False"],
    parallel: ["1", "2", "4"],
  };

  const APPS = [
    { key: "chat", title: "Chat", desc: "Chat with models in the Toki TUI", cmd: `toki tui --raw-url ${RAW_URL} --daemon-url ${DAEMON_URL}`, model: true },
    { key: "tokirecur", title: "Launch TokiRecur", desc: "The TokiRecur desktop app: chat, agents, voice, video", cmd: "tokirecur", model: false },
    { key: "opencode", title: "Launch OpenCode", desc: "Anomaly's open-source coding agent", cmd: "opencode", model: true },
    { key: "hermes", title: "Launch Hermes Agent", desc: "Self-improving AI agent built by Nous Research", cmd: "hermes", model: true },
    { key: "openclaw", title: "Launch OpenClaw", desc: "Personal AI with 100+ skills", cmd: "openclaw", model: true },
    { key: "codex", title: "Launch Codex", desc: "OpenAI CLI coding agent (Responses API)", cmd: "codex", model: true },
    { key: "mimo", title: "Launch MiMo Code", desc: "Xiaomi's OpenCode-compatible coding agent", cmd: "mimo", model: true },
    { key: "qwen", title: "Launch Qwen Code", desc: "Qwen's terminal coding agent", cmd: "qwen", model: true },
    { key: "droid", title: "Launch Factory Droid", desc: "Factory's AI coding agent in your terminal", cmd: "droid", model: true },
    { key: "cline", title: "Launch Cline CLI", desc: "Cline's terminal coding assistant", cmd: "cline", model: true },
    { key: "dsh", title: "Launch DeepSeek Harness", desc: "DeepSeek Harness terminal agent", cmd: "dsh", model: true },
    { key: "grok", title: "Launch Grok Build", desc: "xAI's terminal coding agent", cmd: "grok", model: true },
    { key: "kilo", title: "Launch Kilo Code", desc: "Kilo's terminal coding agent", cmd: "kilo", model: true },
    { key: "pi", title: "Launch Pi", desc: "Pi coding agent", cmd: "pi", model: true },
    { key: "copilot", title: "Launch GitHub Copilot", desc: "GitHub Copilot CLI", cmd: "copilot", model: true },
    { key: "claude", title: "Launch Claude Code", desc: "Anthropic's terminal coding agent", cmd: "claude", model: true },
    { key: "kimi", title: "Launch Kimi Code", desc: "Moonshot AI's terminal coding agent", cmd: "kimi", model: true },
    { key: "omp", title: "Launch OMP", desc: "Oh My Pi coding agent", cmd: "omp", model: true },
    { key: "web", title: "Launch Web Dashboards", desc: "browser UIs: toki · openclaw · hermes", cmd: "toki web", model: true },
    { key: "dshweb", title: "Launch DeepSeek Harness Web", desc: "DeepSeek Harness browser UI on :3080", cmd: "dsh web", model: true },
  ];

  const MODELS = [
    { name: "9b", ctx: "200k", vision: true, gguf: "/models/qwen3.5-9b-q4_k_m.gguf" },
    { name: "35b", ctx: "200k", vision: true, gguf: "/models/qwen3.6-35b-a3b-ud-q4_k_m.gguf" },
    { name: "laguna", ctx: "200k", vision: false, gguf: "/models/laguna/laguna-s-2.1-q4_k_m.gguf" },
    { name: "qwen3.5-0.8b-gguf", ctx: "128k", vision: true, gguf: "/models/qwen3.5-0.8b-q4_k_m.gguf" },
    { name: "qwen3-4b-thinking-2507-gguf", ctx: "128k", vision: false, gguf: "/models/qwen3-4b-thinking-2507-q4_k_m.gguf" },
    { name: "qwen3-0.6b-gguf", ctx: "40k", vision: false, gguf: "/models/qwen3-0.6b-q8_0.gguf" },
    { name: "kwaipilot-kat-coder-v2.5-dev-gguf", ctx: "256k", vision: false, gguf: "/models/kat-coder-pro-v1-q4_k_m.gguf" },
    { name: "qwen3.5-2b-gguf", ctx: "200k", vision: true, gguf: "/models/qwen3.5-2b-q4_k_m.gguf" },
    { name: "lfm2.5-2.6b-gguf", ctx: "32k", vision: false, gguf: "/models/lfm2.5-2.6b-q8_0.gguf" },
    { name: "qwen3.5-122b-gguf", ctx: "200k", vision: true, gguf: "/models/qwen3.5-122b-q4_k_m.gguf" },
    { name: "qwen3-8b-gguf", ctx: "128k", vision: false, gguf: "/models/qwen3-8b-q4_k_m.gguf" },
    { name: "ling-3.0-flash-gguf", ctx: "32k", vision: false, gguf: "/models/ling-3.0-flash-q4_k_m.gguf" },
    { name: "gemma-4-e2b-it-gguf", ctx: "128k", vision: true, gguf: "/models/gemma-4-e2b-it-q4_k_m.gguf" },
    { name: "gemma-4-e2b-gguf", ctx: "128k", vision: true, gguf: "/models/gemma-4-e2b-q4_k_m.gguf" },
    { name: "google_gemma-4-e2b-it-gguf", ctx: "128k", vision: true, gguf: "/models/gemma-4-e2b-it-q4_k_m.gguf" },
    { name: "muse-glimmer-30b-gguf", ctx: "128k", vision: true, gguf: "/models/muse-glimmer-30b-q4_k_m.gguf" },
    { name: "ling-3.0-tiny-gguf", ctx: "32k", vision: false, gguf: "/models/ling-3.0-tiny-q4_k_m.gguf" },
    { name: "nvidia-nemotron-3.5-lightning-30b-a3b-gguf", ctx: "128k", vision: true, gguf: "/models/nemotron-3.5-lightning-q4_k_m.gguf" },
    { name: "deepseek-v4-flash-0731-gguf", ctx: "128k", vision: false, gguf: "/models/deepseek-v4-flash-q4_k_m.gguf" },
    { name: "mistral-medium-3.5-128b-gguf", ctx: "128k", vision: false, gguf: "/models/mistral-medium-3.5-128b-q4_k_m.gguf" },
    { name: "llama-3.3-70b-instruct-abliterated-gguf", ctx: "128k", vision: false, gguf: "/models/llama-3.3-70b-abliterated-q4_k_m.gguf" },
    { name: "gpt-oss-20b-gguf", ctx: "128k", vision: false, gguf: "/models/gpt-oss-20b-mxfp4.gguf" },
    { name: "qwen3-8b-gguf", ctx: "128k", vision: false, gguf: "/models/qwen3-8b-q4_k_m.gguf" },
    { name: "ornith-1.5-35b-a3b-gguf", ctx: "200k", vision: true, gguf: "/models/ornith-1.5-35b-a3b-q4_k_m.gguf" },
    { name: "maple-preview-gguf", ctx: "64k", vision: false, gguf: "/models/maple-preview-q4_k_m.gguf" },
    { name: "ornith-1.5-35b-a3b-uncensored-gguf", ctx: "200k", vision: true, gguf: "/models/ornith-1.5-35b-a3b-q4_k_m.gguf" },
    { name: "qwen3-vl-4b", ctx: "128k", vision: true, gguf: "/models/qwen3-vl-4b-q4_k_m.gguf" },
    { name: "qwen3-vl-2b", ctx: "128k", vision: true, gguf: "/models/qwen3-vl-2b-q4_k_m.gguf" },
    { name: "lfm2-230m-gguf", ctx: "8k", vision: false, gguf: "/models/lfm2-230m-q8_0.gguf" },
    { name: "cloudflare_clef-flash-gguf", ctx: "128k", vision: true, gguf: "/models/cloudflare-clef-flash-q4_k_m.gguf" },
    { name: "qwen-image-2.1-turbo-gguf", ctx: "8k", vision: true, gguf: "/models/qwen-image-2.1-turbo-q4_k_m.gguf" },
    { name: "drama-3-gguf", ctx: "32k", vision: false, gguf: "/models/drama-3-q8_0.gguf" },
    { name: "breeze-tts-2.cpp", ctx: "8k", vision: false, gguf: "/models/breeze-tts-2-q4_k_m.gguf" },
    { name: "breeze-tts-2", ctx: "8k", vision: false, gguf: "/models/breeze-tts-2-q4_k_m.gguf" },
    { name: "ornith-1.5-122b-gguf", ctx: "128k", vision: false, gguf: "/models/ornith-1.5-122b-q4_k_m.gguf" },
    { name: "ornith-3.8-flash-next-gguf", ctx: "256k", vision: true, gguf: "/models/ornith-3.8-flash-next-q4_k_m.gguf" },
    { name: "spark-x2.6-4b-gguf", ctx: "64k", vision: false, gguf: "/models/spark-x2.6-4b-q4_k_m.gguf" },
    { name: "nvidia-nemotron-nano-9b-gguf", ctx: "128k", vision: true, gguf: "/models/nemotron-nano-9b-q4_k_m.gguf" },
  ];
  const PINNED = 23;

  /* The settings pane: three models, the fields the app exposes. */
  const SETTINGS = [
    {
      group: "9b",
      fields: [
        ["gguf", "/models/qwen3.5-9b-q4_k_m.gguf"],
        ["llama_port", "8088"],
        ["daemon_port", "8787"],
        ["ctx_size", "200000"],
        ["parallel", "2"],
        ["kv", "f16"],
        ["max_tokens", "reg:100000 · live:200000"],
        ["model_id", "qwen3.5-9b"],
        ["binary", "/opt/tokirecur/bin/llama-server"],
        ["always_on", "False"],
        ["boot", "False"],
        ["reasoning", "auto"],
      ],
    },
    {
      group: "35b",
      fields: [
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
      ],
    },
    {
      group: "laguna",
      fields: [
        ["gguf", "/models/laguna-s-2.1-q4_k_m.gguf"],
        ["llama_port", "8090"],
        ["daemon_port", "8789"],
        ["ctx_size", "200000"],
        ["parallel", "2"],
        ["kv", "f16"],
        ["max_tokens", "reg:100000 · live:131072"],
        ["model_id", "laguna-s-2.1"],
        ["binary", "/opt/tokirecur/bin/llama-server"],
        ["always_on", "False"],
        ["boot", "False"],
        ["reasoning", "on"],
      ],
    },
  ];

  const WORDMARK = [
    " _____ ___  _  _____ ",
    "|_   _/ _ \\| |/ /_ _|",
    "  | || | | | ' / | | ",
    "  | || |_| | . \\ | | ",
    "  |_| \\___/|_|\\_\\___|",
  ];

  const GUT = "  ";

  let root = null;
  let bodyEl = null;
  let timer = 0;
  let state = null;

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };


  function line(cls, text) {
    const row = el("div", `tui-line${cls ? ` ${cls}` : ""}`, text);
    (bodyEl || root).appendChild(row);
    return row;
  }

  function span(row, cls, text) {
    const node = el("span", cls, text);
    row.appendChild(node);
    return node;
  }

  /* --- state ------------------------------------------------------------- */

  function initial() {
    return {
      screen: "launcher",
      index: 0,
      mode: "cached",
      pinned: PINNED,
      model: PINNED,
      row: 0,
      launched: null,
      edit: null,
      restart: null,
      narrow: window.matchMedia("(max-width: 860px)").matches,
    };
  }

  /* --- launcher ---------------------------------------------------------- */

  function renderLauncher() {
    WORDMARK.forEach((art) => line("wordmark", art));
    line("wordmark-plain", "TOKI");
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
          span(row, "chip", `(${MODELS[state.pinned].name})`);
        }
        const detail = line("dim");
        span(detail, null, GUT);
        span(detail, null, app.desc + (app.model ? ` · ${state.mode}` : ""));
      } else {
        span(row, "name", app.title);
        if (app.model) {
          span(row, "pad", "  ");
          span(row, "chip dim-chip", `(${MODELS[state.pinned].name})`);
        }
      }
    });
  }

  /* --- models ------------------------------------------------------------ */

  function renderModels() {
    const head = line("head");
    span(head, "gutter", GUT);
    span(head, "sel", "Select model");

    MODELS.forEach((model, i) => {
      const active = i === state.model;
      const row = line(active ? "active" : null);
      row.dataset.row = String(i);
      span(row, "gutter", GUT);
      if (active) span(row, "bar", "▌ ");
      span(row, active ? "sel" : "name", model.name);
      if (model.vision) span(row, "chip dim-chip", " [vision]");
      if (i === state.pinned) span(row, "chip dim-chip", "  ◀ pinned");
      span(row, "dot", "  ");
      span(row, "ok", "●");
      span(row, "dim", " ●");
      if (active) {
        const detail = line("dim");
        span(detail, null, GUT + "  ");
        span(detail, null, `${model.ctx} ctx · :${8088 + (i % 3)}/:${8787 + (i % 3)} · ${model.gguf}`);
      }
    });
  }

  /* --- settings ---------------------------------------------------------- */

  const SETTING_ROWS = [];
  SETTINGS.forEach((group, gi) => {
    group.fields.forEach(([key, value], fi) => {
      SETTING_ROWS.push({ group: gi, key, value, first: fi === 0, editable: true });
    });
  });

  function renderSettings() {
    const head = line("head");
    span(head, "gutter", GUT);
    span(head, "sel", "Settings");
    span(head, "dim", "    ↑↓ move · enter edit · esc back");

    SETTINGS.forEach((group, gi) => {
      const title = line("group");
      span(title, "gutter", GUT);
      span(title, "sel", group.group);
      group.fields.forEach(([key, value], fi) => {
        const index = SETTING_ROWS.findIndex((r) => r.group === gi && r.key === key);
        const active = index === state.row;
        const row = line(active ? "active setting-row" : "setting-row");
        row.dataset.row = String(index);
        span(row, "gutter", GUT + (fi === 0 ? "" : "  "));
        span(row, "bar", active ? "▌ " : "  ");
        span(row, active ? "key sel" : "key", key);
        span(row, "value", value);
        if (ENUMS[key]) span(row, "dim", " ▾");
      });
    });
  }

  /* --- edit modal -------------------------------------------------------- */

  function openEdit(index) {
    const row = SETTING_ROWS[index];
    if (ENUMS[row.key]) {
      const values = ENUMS[row.key];
      const next = values[(values.indexOf(row.value) + 1) % values.length];
      SETTINGS[row.group].fields.find((f) => f[0] === row.key)[1] = next;
      row.value = next;
      state.row = index;
      state.restart = RESTART.has(row.key) ? `${row.key} → ${next}` : null;
      render();
      return;
    }
    state.edit = { index, key: row.key, group: SETTINGS[row.group].group, value: row.value };
    render();
  }

  function commitEdit(value) {
    const edit = state.edit;
    if (!edit) return;
    const row = SETTING_ROWS[edit.index];
    SETTINGS[row.group].fields.find((f) => f[0] === row.key)[1] = value;
    row.value = value;
    state.restart = RESTART.has(row.key) ? `${row.key} → ${value}` : null;
    state.edit = null;
    render();
    focusRoot();
  }

  function focusRoot() {
    if (root) root.focus({ preventScroll: true });
  }

  function renderEdit() {
    const edit = state.edit;
    const veil = el("div", "tui-veil");
    const box = el("div", "tui-modal");
    box.appendChild(el("div", "modal-title", `Edit ${edit.group} · ${edit.key}`));
    const input = el("input", "modal-input");
    input.type = "text";
    input.value = edit.value;
    input.setAttribute("aria-label", `Value for ${edit.key}`);
    box.appendChild(input);
    const actions = el("div", "modal-actions");
    const cancel = el("button", "modal-btn", "esc cancel");
    cancel.type = "button";
    const save = el("button", "modal-btn primary", "enter save");
    save.type = "button";
    actions.appendChild(cancel);
    actions.appendChild(save);
    box.appendChild(actions);
    veil.appendChild(box);
    root.appendChild(veil);

    input.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Enter") {
        event.preventDefault();
        commitEdit(input.value);
      } else if (event.key === "Escape") {
        event.preventDefault();
        state.edit = null;
        render();
        focusRoot();
      }
    });
    cancel.addEventListener("click", () => {
      state.edit = null;
      render();
      focusRoot();
    });
    save.addEventListener("click", () => commitEdit(input.value));
    window.requestAnimationFrame(() => {
      input.focus({ preventScroll: true });
      input.select();
    });
  }

  /* --- chrome ------------------------------------------------------------ */

  function renderStatus() {
    const row = line("status");
    span(row, "gutter", GUT);
    span(row, "sel", `● ${MODELS[state.pinned].name}`);
    span(row, "dim", "  llama ");
    span(row, "ok", "●");
    span(row, "dim", "  daemon ");
    span(row, "ok", "●");
    span(row, "dim", "  2 slots   ");
    if (state.launched) {
      span(row, "dim", "    ");
      span(row, "sel", `⟳ starting ${state.launched.title} …`);
    } else {
      span(row, "dim", state.restart ? `⟳ restart required: ${state.restart}` : "");
    }
  }

  function renderHints() {
    const row = line("hints");
    span(row, "gutter", GUT);
    const wide = {
      launcher: "↑/↓ move · 1-9 jump · m mode · → models · s settings · enter open",
      models: "↑/↓ move · 1-9 jump · enter pin · esc back",
      settings: "↑/↓ move · enter edit · esc back",
    };
    const narrow = {
      launcher: "tap a row, tap again to open it",
      models: "tap a model to pin it",
      settings: "tap a field to edit it",
    };
    span(row, "dim", state.narrow ? narrow[state.screen] || "" : wide[state.screen] || "");
  }

  function render() {
    root.textContent = "";
    root.dataset.screen = state.screen;
    bodyEl = el("div", "tui-body");
    root.appendChild(bodyEl);
    if (state.screen === "models") {
      renderModels();
    } else if (state.screen === "settings") {
      renderSettings();
    } else {
      renderLauncher();
    }

    const foot = el("div", "tui-foot");
    root.appendChild(foot);
    bodyEl = foot;
    if (state.screen === "launcher" && state.launched) {
      const row = line("dim");
      span(row, "gutter", GUT);
      span(row, "dim", `↳ ${state.launched.cmd}`);
    }
    renderStatus();
    renderHints();
    bodyEl = root;
    if (state.edit) renderEdit();
    const selected = root.querySelector(".tui-line.active");
    if (selected) selected.scrollIntoView({ block: "nearest" });
  }

  function setScreen(screen) {
    state.screen = screen;
    state.edit = null;
    if (screen !== "settings") state.restart = null;
    render();
    focusRoot();
  }

  /* --- keys -------------------------------------------------------------- */

  function move(delta) {
    if (state.screen === "launcher") {
      state.index = (state.index + delta + APPS.length) % APPS.length;
    } else if (state.screen === "models") {
      state.model = (state.model + delta + MODELS.length) % MODELS.length;
    } else if (state.screen === "settings") {
      state.row = (state.row + delta + SETTING_ROWS.length) % SETTING_ROWS.length;
      state.restart = null;
    }
    render();
  }

  function activate() {
    if (state.screen === "launcher") {
      const app = APPS[state.index];
      state.launched = app;
      render();
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        state.launched = null;
        render();
      }, 2200);
      return;
    }
    if (state.screen === "models") {
      state.pinned = state.model;
      setScreen("launcher");
      return;
    }
    if (state.screen === "settings") {
      openEdit(state.row);
    }
  }

  function onKey(event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.target.closest(".tui-modal, .tui-input")) return;
    const key = event.key;
    let used = true;
    if (state.edit) {
      used = false;
    } else if (key === "ArrowDown" || key === "j") move(1);
    else if (key === "ArrowUp" || key === "k") move(-1);
    else if (key >= "1" && key <= "9") {
      const index = Number(key) - 1;
      if (state.screen === "launcher") state.index = index;
      else if (state.screen === "models" && index < MODELS.length) state.model = index;
      render();
    } else if (key === "m") {
      state.mode = state.mode === "cached" ? "raw" : state.mode === "raw" ? "compare" : "cached";
      render();
    } else if (key === "s") setScreen(state.screen === "settings" ? "launcher" : "settings");
    else if (key === "ArrowRight") setScreen(state.screen === "launcher" ? "models" : state.screen);
    else if (key === "ArrowLeft" || key === "Escape") setScreen("launcher");
    else if (key === "Enter") activate();
    else used = false;
    if (!used) return;
    event.preventDefault();
  }

  /* --- mount ------------------------------------------------------------- */

  document.querySelectorAll("[data-tui]").forEach((node) => {
    root = node;
    state = initial();
    root.setAttribute("tabindex", "0");
    root.setAttribute("role", "application");
    root.setAttribute(
      "aria-label",
      "The toki launcher: the agents it can start, its settings, and the model list. Arrow keys move, Enter opens, s opens the settings, the right arrow opens the model list, Escape goes back."
    );
    root.addEventListener("keydown", onKey);
    root.addEventListener("focus", () => root.classList.add("focused"));
    root.addEventListener("blur", () => root.classList.remove("focused"));
    root.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".tui-input, .tui-modal, .tui-prompt")) return;
      const row = event.target.closest("[data-row]");
      if (row) {
        const index = Number(row.dataset.row);
        const already =
          (state.screen === "launcher" && state.index === index) ||
          (state.screen === "models" && state.model === index) ||
          (state.screen === "settings" && state.row === index);
        if (already) {
          activate();
        } else {
          if (state.screen === "launcher") state.index = index;
          else if (state.screen === "models") state.model = index;
          else state.row = index;
          render();
        }
        root.focus({ preventScroll: true });
        event.preventDefault();
        return;
      }
      root.focus({ preventScroll: true });
    });

    const query = window.matchMedia("(max-width: 860px)");
    const onChange = () => {
      if (query.matches !== state.narrow) {
        state.narrow = query.matches;
        render();
      }
    };
    if (query.addEventListener) query.addEventListener("change", onChange);
    else query.addListener(onChange);

    const reset = document.querySelector("[data-tui-reset]");
    if (reset) {
      reset.addEventListener("click", () => {
        window.clearTimeout(timer);
        state = initial();
        render();
      });
    }

    render();
  });
})();
