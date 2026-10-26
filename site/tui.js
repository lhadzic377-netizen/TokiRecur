/* ---------------------------------------------------------------------------
   toki, in the browser.

   A simulated session of the real terminal app: the launcher, the chat with
   its live panels, the settings, the model picker. Nothing here talks to a
   model — the answers are canned and the timings are invented — but the
   layout, the keys and the way the cache reports itself are the app's.

   The one thing worth trying: send the same message twice. The second turn
   comes back as a cache hit, because that is the whole product.

   Keys: ↑ ↓ / j k move · 1-9 jump · m mode · → models · s settings ·
         enter open · esc back · /help in the chat
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
    { name: "laguna", ctx: "200k", vision: false, gguf: "/models/laguna-s-2.1-q4_k_m.gguf" },
    { name: "qwen3.5-0.8b-gguf", ctx: "128k", vision: true, gguf: "/models/qwen3.5-0.8b-q4_k_m.gguf" },
    { name: "qwen3-4b-thinking-2507-gguf", ctx: "128k", vision: false, gguf: "/models/qwen3-4b-thinking-2507-q4_k_m.gguf" },
    { name: "qwen3-0.6b-gguf", ctx: "40k", vision: false, gguf: "/models/qwen3-0.6b-q4_k_m.gguf" },
    { name: "qwen3.5-2b-gguf", ctx: "200k", vision: true, gguf: "/models/qwen3.5-2b-q4_k_m.gguf" },
    { name: "lfm2.5-2.6b-gguf", ctx: "32k", vision: false, gguf: "/models/lfm2.5-2.6b-q8_0.gguf" },
    { name: "ornith-1.5-35b-a3b-gguf", ctx: "200k", vision: true, gguf: "/models/ornith-1.5-35b-a3b-q4_k_m.gguf" },
    { name: "ornith-1.5-122b-gguf", ctx: "128k", vision: false, gguf: "/models/ornith-1.5-122b-q4_k_m.gguf" },
    { name: "qwen3-vl-4b", ctx: "128k", vision: true, gguf: "/models/qwen3-vl-4b-q4_k_m.gguf" },
    { name: "gemma-4-e2b-it-gguf", ctx: "128k", vision: true, gguf: "/models/gemma-4-e2b-it-q4_k_m.gguf" },
    { name: "ornith-3.8-flash-next-gguf", ctx: "256k", vision: true, gguf: "/models/ornith-3.8-flash-next-q4_k_m.gguf" },
    { name: "spark-x2.6-4b-gguf", ctx: "64k", vision: false, gguf: "/models/spark-x2.6-4b-q4_k_m.gguf" },
    { name: "kwaipilot-kat-coder-pro-v1", ctx: "256k", vision: false, gguf: "/models/kat-coder-pro-v1-q4_k_m.gguf" },
    { name: "qwen3-8b-gguf", ctx: "128k", vision: false, gguf: "/models/qwen3-8b-q4_k_m.gguf" },
    { name: "gpt-oss-20b-gguf", ctx: "128k", vision: false, gguf: "/models/gpt-oss-20b-mxfp4.gguf" },
    { name: "llama-3.3-70b-instruct-abliterated-gguf", ctx: "128k", vision: false, gguf: "/models/llama-3.3-70b-abliterated-q4_k_m.gguf" },
    { name: "mistral-medium-3.5-128b-gguf", ctx: "128k", vision: false, gguf: "/models/mistral-medium-3.5-128b-q4_k_m.gguf" },
    { name: "maple-preview-gguf", ctx: "64k", vision: false, gguf: "/models/maple-preview-q4_k_m.gguf" },
    { name: "drama-3-gguf", ctx: "32k", vision: false, gguf: "/models/drama-3-q8_0.gguf" },
    { name: "qwen-image-2.1-turbo-gguf", ctx: "8k", vision: true, gguf: "/models/qwen-image-2.1-turbo-q4_k_m.gguf" },
    { name: "nvidia-nemotron-nano-9b-gguf", ctx: "128k", vision: true, gguf: "/models/nemotron-nano-9b-q4_k_m.gguf" },
    { name: "ling-3.0-tiny-gguf", ctx: "32k", vision: false, gguf: "/models/ling-3.0-tiny-q4_k_m.gguf" },
    { name: "muse-glimmer-30b-gguf", ctx: "128k", vision: true, gguf: "/models/muse-glimmer-30b-q4_k_m.gguf" },
  ];

  const PINNED = 8;

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

  const HELP = [
    "Commands: /help /quit /clear /mode [raw|cached|compare] /session [id] /session-new /precision /metrics /slots /reset /reprobe /max_tokens [n]",
    "Keys:    Ctrl+1 raw · Ctrl+2 cached · Ctrl+3 compare · Ctrl+L clear · Ctrl+R reprobe · Ctrl+P precision · Ctrl+U new session · Ctrl+M toggle metrics panel · Ctrl+T cycle max tokens",
  ];

  const REPLIES = {
    greet: {
      think: "A greeting. No need to reach for the tools.",
      text: "Hello! Ask me anything, or type /help for the commands this terminal understands.",
    },
    what: {
      think: "They are asking what this is and where it sits. Lead with the position, then the saving.",
      text: "TokiRecur is a daemon in front of llama-server that keeps the KV cache warm between requests. When two turns share a prefix — a system prompt, a file, a long chat — the second one reads the cache back instead of prefilling it again.",
    },
    cache: {
      think: "Explain the mechanism honestly: it is prefix matching, not magic, and a divergent token ends the reuse.",
      text: "Every turn tokenises to a prefix tree. When a new request arrives, the longest shared prefix is already in memory, so prefill skips it and only the new tokens are computed.\n\nThe first turn pays full price. The second, and every later turn that keeps the same prefix, pays for the tail only.",
    },
    maths: {
      think: "Trivial arithmetic. Answer straight.",
      text: "Four.",
    },
    colours: {
      think: "A list is enough here.",
      text: "1. Red\n2. Blue\n3. Green",
    },
    setup: {
      think: "They want the short path from nothing to a working cache.",
      text: "Three commands: install the daemon, pull a model, start the stack. Then point any OpenAI-compatible client at :8788 and it shares the cache with every other client doing the same.",
    },
    modes: {
      think: "Explain the three modes without overselling compare.",
      text: "raw sends each turn straight to llama-server. cached sends it through the daemon, which reuses the prefix. compare runs both and shows the two timings, so you can see the difference on your own machine.",
    },
    fallback: {
      think: "No canned answer for this one.",
      text: "In the real terminal that goes to your model. Here it is a simulation, so the answer is mine: try /help for the commands, /precision for a cache check, or ask me about the cache, the settings, or the model list.",
    },
  };

  const GUT = "  ";
  const ENTER = " ↵";

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

  const pad = (n) => String(n).padStart(2, "0");
  const clock = () => {
    const d = new Date();
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  };
  const number = (n) => n.toLocaleString("en-US");

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

  function freshChat() {
    return {
      messages: [
        { role: "system", at: "09:41:02", style: "cyan", text: "session #a7f2 · cached · 200k ctx" },
        { role: "user", at: "09:41:07", text: "Summarise what this daemon does, in two sentences." },
        {
          role: "ai",
          at: "09:41:12",
          mode: "cached",
          hit: false,
          think: "They want the pitch, not the internals. Two sentences, no bullets.",
          text: "It sits in front of llama-server and keeps the KV cache alive between requests. Agents that share a long prefix compute it once, then every later turn reads it back instead of prefilling again.",
          stats: "in 1,204 tok · ttft 3600ms · total 4400ms · 62 tok",
        },
        { role: "user", at: "09:41:40", text: "Ask the exact same thing again." },
        {
          role: "ai",
          at: "09:41:41",
          mode: "cached",
          hit: true,
          think: "Same prefix as the turn above. This should be a full hit.",
          text: "Same prefix, so the cache answers it: 1,204 prompt tokens read back in 210ms instead of prefilled for 3.6s. The words below are the first answer again, because a hit is a hit.",
          stats: "in 0 tok · ttft 210ms · total 380ms · 61 tok · reused 1204",
        },
        { role: "system", at: "09:41:41", style: "dim", text: "cache: 1,204 tok reused · hit rate 50% (1/2)" },
      ],
      seen: {},
      busy: false,
      live: { req: 14, hit: 8, miss: 6, skip: 9473, slots: 2, auth: 0 },
      session: { msgs: 2, gen: 123, reused: 1204, hits: 1, total: 2, ctx: 4 },
      precision: null,
      panel: true,
    };
  }

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
      chat: freshChat(),
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

  /* --- chat -------------------------------------------------------------- */

  /* The chat is the real app's chat: the sidebar panel, the bordered
     transcript with the brand header, the metrics panel, and the input
     row. Same words, same colours, same badges. */

  function renderChat() {
    const chat = state.chat;
    const wrap = el("div", "tui-chat");
    root.appendChild(wrap);

    /* left panel */
    const side = el("aside", "tui-side");
    wrap.appendChild(side);
    const sideBox = el("div", "panel-box");
    side.appendChild(sideBox);
    sideBox.appendChild(el("b", "panel-title", "Toki"));
    sideBox.appendChild(el("div", "rule"));
    sideBox.appendChild(el("div", `side-line mode-${state.mode}`, `mode ${state.mode}`));
    sideBox.appendChild(el("div", "side-line yellow", "ses  #a7f2"));
    sideBox.appendChild(el("div", "rule"));
    const status = (glyph, cls, text) => {
      const row = el("div", "side-line");
      row.appendChild(el("span", cls, glyph));
      row.appendChild(el("span", "dim", ` ${text}`));
      sideBox.appendChild(row);
    };
    status("●", "ok", "llama 127.0.0.1:8100");
    status("●", "ok", "daemon 127.0.0.1:8799");
    status("●", "ok", "auth ok");
    sideBox.appendChild(el("div", "rule"));
    sideBox.appendChild(el("b", "panel-title", "Slots"));
    sideBox.appendChild(el("div", "side-line dim", "  · s0 0%"));
    sideBox.appendChild(el("div", `side-line ${chat.busy ? "cyan" : "dim"}`, "  · s1 0%"));
    sideBox.appendChild(el("div", "side-line dim", "Ctrl+1/2/3 mode"));
    sideBox.appendChild(el("div", "side-line dim", "· Ctrl+U session"));

    /* transcript */
    const main = el("main", "tui-main");
    wrap.appendChild(main);
    const box = el("div", "chat-box");
    box.dataset.transcript = "1";
    main.appendChild(box);
    box.appendChild(el("b", "chat-wordmark", "TOKI"));
    box.appendChild(el("div", "chat-tagline", "recycled-prefill LLM stack · AMD Ryzen AI"));
    chat.messages.forEach((message) => box.appendChild(messageNode(message)));
    if (chat.busy) {
      const waiting = el("div", "msg ai");
      span(waiting, "msg-time", `[${clock()}]`);
      span(waiting, "msg-role", "ai");
      span(waiting, "msg-cursor", "▌");
      box.appendChild(waiting);
    }

    /* right panel */
    if (chat.panel) wrap.appendChild(panelNode());

    /* prompt */
    const form = el("form", "tui-prompt");
    const input = el("input", "tui-input");
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "Type a message and press Enter…  (try /help)";
    input.setAttribute("aria-label", "Message to the simulated toki session");
    form.appendChild(input);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        send(input.value);
      } else if (event.key === "Escape") {
        event.preventDefault();
        input.blur();
        setScreen("launcher");
      }
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      send(input.value);
    });
    wrap.appendChild(form);

    /* A phone should not grow a keyboard the visitor did not ask for. */
    if (!chat.busy && !window.matchMedia("(hover: none)").matches) {
      window.requestAnimationFrame(() => {
        input.focus({ preventScroll: true });
        scrollTranscript();
      });
    }
    window.requestAnimationFrame(scrollTranscript);
  }

  function messageNode(message) {
    const row = el("div", `msg ${message.role}`);
    if (message.think) {
      const think = el("div", "think");
      span(think, "think-text", `    think › ${message.think}`);
      row.appendChild(think);
    }
    const flow = el("div", "msg-line");
    span(flow, "msg-time", `[${message.at}] `);
    if (message.role === "user") {
      span(flow, "msg-you", "you ");
      span(flow, "msg-arrow", "▸ ");
      span(flow, "msg-text", message.text);
    } else if (message.role === "ai") {
      span(flow, "msg-ai", "ai  ");
      span(flow, "msg-mode", `[${message.mode || "cached"}] `);
      if (message.hit === true) span(flow, "hit", "✓ hit ");
      else if (message.hit === false) span(flow, "miss", "✗ miss ");
      else if (message.hit === "raw") span(flow, "rawbadge", "↯ raw ");
      span(flow, "msg-text", message.text);
      if (message.stats) span(flow, "msg-stats", `  [${message.stats}]`);
    } else if (message.role === "compare") {
      span(flow, "msg-compare", "compare ");
      span(flow, "compare-speed", `[${message.speedup}× `);
      span(flow, "dim", "cached vs raw");
      span(flow, "compare-speed", "]");
      span(flow, "dim", `  raw=${message.raw}`);
      span(flow, "dim", `  cached=${message.cached}`);
      row.appendChild(flow);
      if (message.body) {
        const body = el("div", "msg-line");
        span(body, message.identical ? "ok" : "warn", message.body);
        row.appendChild(body);
      }
      return row;
    } else {
      span(flow, "msg-system", "system ");
      span(flow, "msg-arrow magenta", "▸ ");
      span(flow, message.style || "warn", message.text);
    }
    row.appendChild(flow);
    row.body = flow;
    return row;
  }

  function panelNode() {
    const chat = state.chat;
    const live = chat.live;
    const rate = Math.round((live.hit / (live.hit + live.miss)) * 100);
    const panel = el("aside", "tui-info");
    const box = el("div", "panel-box");
    panel.appendChild(box);

    box.appendChild(el("b", "panel-title", "Live"));
    box.appendChild(el("div", "p-line white", `  req=${live.req} hit=${live.hit} miss=${live.miss}`));
    box.appendChild(el("div", "p-line cyan", `  rate=${rate}% skip=${live.skip}tok slots=${live.slots}`));
    if (live.auth) box.appendChild(el("div", "p-line red", `  auth_fail=${live.auth}`));

    box.appendChild(el("b", "panel-title", "Session"));
    const session = chat.session;
    box.appendChild(el("div", "p-line white", `  msgs=${session.msgs} gen=${session.gen}tok reused=${session.reused}tok`));
    const hits = session.total ? `${Math.round((session.hits / session.total) * 100)}%` : "—";
    box.appendChild(el("div", "p-line cyan", `  hits=${hits} ctx=${session.ctx}%`));

    box.appendChild(el("b", "panel-title", "Precision"));
    if (chat.precision) {
      const p = chat.precision;
      const bar = (pct) => {
        const width = 14;
        const filled = Math.round((pct / 100) * width);
        return "█".repeat(filled) + "░".repeat(width - filled);
      };
      const row = (label, pct) => {
        const line = el("div", "p-line");
        line.appendChild(el("span", "p-key", `  ${label}`));
        line.appendChild(el("span", pct >= 99 ? "ok" : pct >= 80 ? "warn" : "red", ` ${bar(pct)} ${pct}%`));
        box.appendChild(line);
      };
      row("baseline", p.baseline);
      row("recycled", p.recycled);
      row("match", p.match);
      box.appendChild(el("div", "p-line dim", "  (10 prompts)"));
    } else {
      box.appendChild(el("div", "p-line dim", "  (Ctrl+P to run)"));
    }
    return panel;
  }

  function scrollTranscript() {
    const main = root && root.querySelector("[data-transcript]");
    if (main) main.scrollTop = main.scrollHeight;
  }

  function refreshPanel() {
    const old = root.querySelector(".tui-panel");
    if (!old) return;
    old.replaceWith(panelNode());
  }

  /* --- sending ----------------------------------------------------------- */

  function intentFor(text) {
    const value = text.toLowerCase();
    if (/\b(hi|hello|hey|yo)\b/.test(value)) return "greet";
    if (/2\s*\+\s*2|what is 2\+2/.test(value)) return "maths";
    if (/colou?rs?/.test(value)) return "colours";
    if (/(raw|cached|compare)\s*(mode|vs)|what.*mode/.test(value)) return "modes";
    if (/cache|kv|prefix|reuse|hit rate/.test(value)) return "cache";
    if (/install|setup|set up|start|command|run it/.test(value)) return "setup";
    if (/tokirecur|toki\b|daemon|what.*(is|does) (this|it)/.test(value)) return "what";
    return "fallback";
  }

  function send(raw) {
    const text = (raw || "").trim();
    if (!text) return;
    const chat = state.chat;
    if (chat.busy) return;

    const input = root.querySelector(".tui-input");
    if (input) input.value = "";

    const command = text.toLowerCase();
    if (command === "/help") {
      push({ role: "system", at: clock(), style: "cyan", text: HELP[0] });
      push({ role: "system", at: clock(), style: "cyan", text: HELP[1] });
      return;
    }
    if (command === "/clear") {
      chat.messages = [{ role: "system", at: clock(), style: "dim", text: "chat cleared" }];
      render();
      return;
    }
    if (command === "/quit") {
      setScreen("launcher");
      return;
    }
    if (command === "/metrics") {
      chat.panel = !chat.panel;
      render();
      return;
    }
    if (command.startsWith("/mode")) {
      const next = text.split(/\s+/)[1];
      if (["raw", "cached", "compare"].includes(next)) {
        state.mode = next;
        push({ role: "system", at: clock(), style: "cyan", text: `mode → ${next}` });
      } else {
        push({ role: "system", at: clock(), style: "red", text: "unknown command: /mode  (try /help)" });
      }
      return;
    }
    if (command === "/precision") {
      push({ role: "system", at: clock(), style: "cyan", text: "running precision check…" });
      window.setTimeout(() => {
        chat.precision = { baseline: 33, recycled: 100, match: 33 };
        push({ role: "system", at: clock(), style: "ok", text: "precision: baseline=33%  recycled=100%  match=33%" });
      }, 900);
      return;
    }
    if (command === "/session-new") {
      chat.messages = [{ role: "system", at: clock(), style: "cyan", text: "new session: b3c8d4e1f0…" }];
      render();
      return;
    }

    /* A normal turn. */
    push({ role: "user", at: clock(), text });
    const key = text.toLowerCase().replace(/\s+/g, " ").trim();
    const rawMode = state.mode === "raw";
    const hit = rawMode ? "raw" : Boolean(chat.seen[key]);
    chat.seen[key] = true;
    const reply = REPLIES[intentFor(text)];
    chat.busy = true;

    const tokens = reply.text.split(/\s+/).length;
    const ttft = hit ? 180 + Math.round(Math.random() * 60) : 900 + Math.round(Math.random() * 900);
    const total = hit ? 380 + Math.round(Math.random() * 90) : ttft + 1400 + Math.round(Math.random() * 900);

    chat.live.req += 1;
    if (hit) chat.live.hit += 1;
    else chat.live.miss += 1;
    chat.session.msgs += 1;
    chat.session.total += 1;
    if (hit) chat.session.hits += 1;

    render();

    window.setTimeout(() => {
      if (state.screen !== "chat") return;
      const message = {
        role: "ai",
        at: clock(),
        mode: rawMode ? "raw" : "cached",
        hit,
        think: reply.think,
        text: "",
        stats: null,
      };
      chat.messages.push(message);
      const node = messageNode(message);
      const main = root.querySelector("[data-transcript]");
      if (main) {
        const waiting = main.querySelector(".msg-cursor");
        if (waiting) waiting.closest(".msg").remove();
        main.appendChild(node);
      }
      const body = node.querySelector(".msg-text");

      const words = reply.text.split(/(\s+)/);
      let i = 0;
      const tick = () => {
        i += 1;
        body.textContent += words.slice(i - 1, i).join("");
        message.text = body.textContent;
        scrollTranscript();
        if (i < words.length) {
          timer = window.setTimeout(tick, hit ? 6 : 22);
          return;
        }
        message.stats =
          `in ${hit === true ? 0 : number(1204)} tok · ttft ${ttft}ms · total ${total}ms · ${tokens} tok` +
          (hit === true ? " · reused 1204" : "");
        const stats = el("span", "msg-stats", `  [${message.stats}]`);
        node.body.appendChild(stats);
        chat.session.gen += tokens;
        if (hit) chat.session.reused += 1204;
        chat.live.skip += hit ? 1204 : 0;
        chat.busy = false;
        if (state.mode === "compare") {
          const rawTtft = ttft + 720 + Math.round(Math.random() * 400);
          const rawTotal = total + 900 + Math.round(Math.random() * 700);
          const ratio = (total / rawTotal).toFixed(2);
          const identical = hit !== "raw";
          const compare = {
            role: "compare",
            at: clock(),
            speedup: ratio,
            raw: `${rawTtft}/${rawTotal}ms`,
            cached: `${ttft}/${total}ms`,
            identical,
            body: identical
              ? "  outputs byte-identical"
              : "  outputs differ: 39 chars (first diff at char 19)",
          };
          chat.messages.push(compare);
          const transcript = root.querySelector("[data-transcript]");
          if (transcript) transcript.appendChild(messageNode(compare));
        }
        refreshPanel();
        scrollTranscript();
        const input = root.querySelector(".tui-input");
        if (input) input.focus({ preventScroll: true });
      };
      timer = window.setTimeout(tick, 0);
    }, ttft);
  }

  function push(message) {
    state.chat.messages.push(message);
    render();
    scrollTranscript();
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
    if (root && state.screen !== "chat") root.focus({ preventScroll: true });
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
    span(row, "dim", `  ${state.chat.live.slots} slots   `);
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
    if (state.screen === "chat") {
      bodyEl = root;
      renderChat();
      return;
    }

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
    const previous = state.screen;
    if (state.screen === "chat" && screen !== "chat" && state.chat.busy) {
      window.clearTimeout(timer);
      state.chat.busy = false;
    }
    state.screen = screen;
    state.edit = null;
    if (screen !== "settings") state.restart = null;
    render();
    if (screen !== "chat") focusRoot();
    else if (previous !== "chat") {
      const input = root.querySelector(".tui-input");
      if (input) input.focus({ preventScroll: true });
    }
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
      if (app.key === "chat") {
        setScreen("chat");
        return;
      }
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
    if (state.screen === "chat") return;
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
      "A simulated toki session: the launcher, chat, settings and model list. Arrow keys move, Enter opens, s opens settings, the right arrow opens the model list, Escape goes back. Inside the chat, type a message and press Enter, or type /help."
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
