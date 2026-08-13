// Toki Recur chat frontend

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

let ws = null;
let currentMode = "raw";
let history = [];        // [{role, content, mode?}]
let busy = false;
let abortController = null;

// ── Auth token persistence ───────────────────────────────────────────────

const AUTH_TOKEN_KEY = "toki_auth_token";
const SESSION_KEY = "toki_session_id";
const HISTORY_KEY = "toki_history";

function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY) || "";
}

function setAuthToken(t) {
  if (t) localStorage.setItem(AUTH_TOKEN_KEY, t);
  else localStorage.removeItem(AUTH_TOKEN_KEY);
}

function authHeaders(extra = {}) {
  const t = getAuthToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

function authFetch(url, opts = {}) {
  const t = getAuthToken();
  const headers = { ...(opts.headers || {}), ...(t ? { Authorization: `Bearer ${t}` } : {}) };
  return fetch(url, { ...opts, headers });
}

// ── Session ID ──────────────────────────────────────────────────────────

function getSessionId() {
  // User override takes priority.
  const override = $("#session-override")?.value?.trim();
  if (override) return override;
  let sid = localStorage.getItem(SESSION_KEY);
  if (!sid) {
    sid = (crypto.randomUUID && crypto.randomUUID()) ||
          "sess-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
    localStorage.setItem(SESSION_KEY, sid);
  }
  return sid;
}

function setSessionId(sid) {
  localStorage.setItem(SESSION_KEY, sid);
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
  const el = $("#session-id");
  if (el) el.textContent = "—";
}

function renderSessionId() {
  const sid = getSessionId();
  const el = $("#session-id");
  if (el) {
    el.textContent = sid.slice(0, 8) + "…";
    el.title = sid + " (click to copy)";
  }
}

// ── Connection ─────────────────────────────────────────────────────────

let daemonOk = false;        // last known daemon health
let fallbackNoted = false;   // avoid spamming the auto-fallback note

async function probeHealth() {
  const llamaEl = $("#llama-status");
  const daemonEl = $("#daemon-status");
  if (!llamaEl || !daemonEl) return;

  // Probe llama-server (no auth needed).
  try {
    const r = await authFetch("/v1/health/llama");
    const body = await r.json();
    llamaEl.textContent = body.status === "ok" ? "ok" : "down";
    llamaEl.className = "status-badge " + (body.status === "ok" ? "status-ok" : "status-unreachable");
  } catch {
    llamaEl.textContent = "down";
    llamaEl.className = "status-badge status-unreachable";
  }

  // Probe daemon (auth-aware: ok / wrong / required / down).
  let status = "down";
  try {
    const r = await authFetch("/v1/health/daemon");
    const body = await r.json();
    status = body.status || "down";
    if (status === "ok") {
      daemonOk = true;
      fallbackNoted = false;
      daemonEl.textContent = "ok";
      daemonEl.className = "status-badge status-ok";
    } else if (status === "auth_required") {
      daemonOk = false;
      // Distinguish "no token configured" from "token rejected".
      const auth = getAuthToken() ? "wrong" : "required";
      daemonEl.textContent = auth;
      daemonEl.className = "status-badge " +
        (auth === "wrong" ? "status-auth_wrong" : "status-auth_required");
    } else {
      daemonOk = false;
      daemonEl.textContent = "down";
      daemonEl.className = "status-badge status-unreachable";
    }
  } catch {
    daemonOk = false;
    daemonEl.textContent = "down";
    daemonEl.className = "status-badge status-unreachable";
  }

  // Auto-fallback: daemon unreachable + cache-requiring mode → raw.
  // Mirrors the TUI's probe behavior.
  if (!daemonOk && (currentMode === "cached" || currentMode === "compare")) {
    if (!fallbackNoted) {
      fallbackNoted = true;
      setMode("raw");
      renderSystemNote("daemon unreachable — mode auto-changed to raw", "warn");
    }
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/chat`);

  ws.addEventListener("open", () => {
    $("#connection-status").textContent = "connected";
    $("#connection-status").className = "connection-status connected";
  });

  ws.addEventListener("close", () => {
    $("#connection-status").textContent = "disconnected — retrying…";
    $("#connection-status").className = "connection-status error";
    // A drop while a request was in flight must not leave the input
    // disabled forever; the retry connection starts clean.
    setBusy(false);
    setTimeout(connect, 2000);
  });

  ws.addEventListener("error", () => {
    $("#connection-status").textContent = "connection error";
    $("#connection-status").className = "connection-status error";
  });

  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); }
    catch { return; }
    handleServerMessage(msg);
  });
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "start": onStart(msg); break;
    case "token": onToken(msg.content); break;
    case "reasoning": onReasoning(msg.content); break;
    case "done": onDone(msg); break;
    case "compare_start": onCompareStart(msg); break;
    case "compare_token": onCompareToken(msg); break;
    case "compare_done": onCompareDone(msg); break;
    case "summary": onSummary(msg); break;
    case "error": onError(msg.message || "unknown error"); break;
  }
}

// ── Session stats ─────────────────────────────────────────────────────

let sessionStats = { messages: 0, generated: 0, reused: 0, hits: 0, cachedMsgs: 0 };

function bumpSessionStats(metrics, cache) {
  sessionStats.messages += 1;
  sessionStats.generated += (metrics && metrics.tokens_generated) || 0;
  if (cache && cache.skipped_tokens) {
    sessionStats.reused += cache.skipped_tokens;
  }
  if (cache) {
    sessionStats.cachedMsgs += 1;
    if (cache.hit) sessionStats.hits += 1;
  }
  renderSessionStats();
}

function renderSessionStats() {
  const set = (id, v) => { const el = $("#" + id); if (el) el.textContent = v; };
  set("stat-messages", sessionStats.messages);
  set("stat-generated", sessionStats.generated + " tok");
  set("stat-reused", sessionStats.reused + " tok");
  set("stat-hitrate", sessionStats.cachedMsgs
    ? Math.round((sessionStats.hits / sessionStats.cachedMsgs) * 100) + "%"
    : "—");
}

function resetSessionStats() {
  sessionStats = { messages: 0, generated: 0, reused: 0, hits: 0, cachedMsgs: 0 };
  renderSessionStats();
}

// ── Single-mode state ──────────────────────────────────────────────────

let activeMsgEl = null;
let activeTextEl = null;
// Thinking-mode state for the in-flight assistant message. Reset on
// every onStart; populated by streaming "reasoning" events; collapsed
// at onDone so the next message isn't crowded.
let activeThinkDetails = null;
let activeThinkBody = null;
let activeThinkBytes = null;
let activeThinkText = "";

function onStart(msg) {
  if (msg.mode === "compare") return; // compare completes via compare_done/summary
  activeMsgEl = renderAssistantMessageShell(msg.mode);
  activeTextEl = activeMsgEl.querySelector(".text");
  activeTextEl.classList.add("streaming-cursor");
  // Per-message thinking state. Reset on every new message so a
  // long session doesn't accumulate across turns.
  activeThinkDetails = activeMsgEl.querySelector("details.thinking");
  activeThinkBody = activeMsgEl.querySelector(".think-body");
  activeThinkBytes = activeMsgEl.querySelector(".think-bytes");
  activeThinkText = "";
}

function onToken(chunk) {
  if (!activeTextEl) return;
  activeTextEl.textContent += chunk;
  scrollToBottom();
}

function onReasoning(chunk) {
  // Thinking-mode deltas stream in their own WS event. We render them
  // into a collapsed <details> block so the user can choose to look at
  // them but they don't dominate the screen while the answer streams.
  if (!activeThinkDetails) return;
  if (activeThinkDetails.hidden) activeThinkDetails.hidden = false;
  activeThinkText += chunk;
  activeThinkBody.textContent = activeThinkText;
  // Live byte count next to the toggle. Approximate token count would
  // be nicer but bytes are cheap and good enough for a glance.
  activeThinkBytes.textContent = `${activeThinkText.length} chars`;
  scrollToBottom();
}

function onDone(msg) {
  if (activeTextEl) {
    activeTextEl.classList.remove("streaming-cursor");
    activeTextEl.textContent = msg.text;
  }
  // Thinking-mode reasoning: use the final consolidated string from
  // the done message (in case any reasoning arrived after the last
  // streamed delta — or in case we connected after the stream started).
  if (activeThinkDetails) {
    if (typeof msg.reasoning === "string" && msg.reasoning.length > 0) {
      activeThinkText = msg.reasoning;
      activeThinkBody.textContent = msg.reasoning;
      activeThinkBytes.textContent = `${msg.reasoning.length} chars`;
      activeThinkDetails.hidden = false;
    }
    // After the response ends, fold the thinking block back up so it
    // doesn't crowd the next message. The user can still click the
    // summary to re-expand.
    activeThinkDetails.removeAttribute("open");
  }
  // Reasoning-but-no-answer case: the model burned the entire budget
  // thinking and produced no visible content. Surface a hint so the
  // user knows the reasoning is the only output.
  if (msg.no_answer && activeTextEl && !activeTextEl.textContent) {
    const note = document.createElement("div");
    note.className = "no-answer-note";
    note.textContent =
      "Model produced reasoning but no answer — likely ran out of " +
      "tokens while thinking. Try a higher max_tokens, or disable " +
      "thinking for this model.";
    activeMsgEl.querySelector(".body").appendChild(note);
  }
  if (activeMsgEl && msg.metrics) {
    appendMetrics(activeMsgEl, msg.metrics, msg.mode, msg.cache);
    // Capture TTFT for sparkline.
    if (typeof msg.metrics.ttft_ms === "number") {
      pushTtftSample(msg.metrics.ttft_ms);
    }
  }
  if (msg.aborted) markStopped(activeMsgEl);
  bumpSessionStats(msg.metrics, msg.cache);
  if (msg.text && currentMode !== "compare") {
    history.push({ role: "assistant", content: msg.text });
    persistHistory();
  }
  activeMsgEl = null;
  activeTextEl = null;
  activeThinkDetails = null;
  activeThinkBody = null;
  activeThinkBytes = null;
  activeThinkText = "";
  setBusy(false);
}

// ── Compare-mode state ─────────────────────────────────────────────────

let comparePair = null;
let comparePanels = {};
let compareMetrics = { raw: null, cached: null };
let compareFailed = { raw: false, cached: false };
let lastComparePrompt = "";

function onCompareStart(msg) {
  compareMetrics[msg.label] = null;
  compareFailed[msg.label] = false;
  const panel = comparePanels[msg.label];
  if (!panel) return;
  panel.textEl.textContent = "";
  panel.textEl.classList.add("streaming-cursor");
  panel.metricsEl.textContent = "";
}

function onCompareToken(msg) {
  const panel = comparePanels[msg.label];
  if (!panel) return;
  panel.textEl.textContent += msg.content;
  scrollToBottom();
}

function onCompareDone(msg) {
  const panel = comparePanels[msg.label];
  if (!panel) return;
  panel.textEl.classList.remove("streaming-cursor");
  panel.textEl.textContent = msg.text;
  if (msg.text) panel.fullText = msg.text;
  // Per-side metrics now come from server in the done payload.
  if (msg.metrics) {
    compareMetrics[msg.label] = msg.metrics;
    renderCompareSideMetrics();
  }
  if (msg.cache) renderCompareCacheBadge(msg.label, msg.cache);
  if (msg.error) {
    compareFailed[msg.label] = true;
    renderCompareSideMetrics();
  }
  if (msg.aborted) markStopped(comparePair);
  if (msg.label === "cached") bumpSessionStats(msg.metrics, msg.cache);
  if (msg.metrics && typeof msg.metrics.ttft_ms === "number") {
    pushTtftSample(msg.metrics.ttft_ms);
  }
}

function renderCompareCacheBadge(label, cache) {
  if (!comparePair) return;
  const header = comparePair.querySelector(`.compare-panel.${label} .label`);
  if (!header || header.querySelector(".cache-badge")) return;
  const badge = document.createElement("span");
  badge.className = "cache-badge " + (cache.hit ? "hit" : "miss");
  badge.textContent = cache.hit ? "hit" : "miss";
  if (cache.skipped_tokens > 0) badge.textContent += ` +${cache.skipped_tokens} tok`;
  header.appendChild(badge);
}

function markStopped(el) {
  if (!el) return;
  const note = document.createElement("span");
  note.className = "stopped-note";
  note.textContent = "stopped";
  el.appendChild(note);
}

function renderCompareSideMetrics() {
  if (!comparePair) return;
  let metricsBox = comparePair.querySelector(".compare-side-metrics");
  if (!metricsBox) {
    metricsBox = document.createElement("div");
    metricsBox.className = "compare-side-metrics";
    const left = comparePair.querySelector(".compare-panel.raw");
    const right = comparePair.querySelector(".compare-panel.cached");
    if (left) left.insertBefore(metricsBox, left.firstChild);
  }

  const hasError = compareFailed.raw || compareFailed.cached;
  metricsBox.classList.toggle("has-error", hasError);

  const raw = compareMetrics.raw || {};
  const cached = compareMetrics.cached || {};

  metricsBox.innerHTML = `
    <div class="compare-side-metric">
      <span class="compare-side-label">raw ttft</span>
      <span class="compare-side-value">${raw.ttft_ms != null ? fmtMs(raw.ttft_ms) : (compareFailed.raw ? "error" : "—")}</span>
    </div>
    <div class="compare-side-metric">
      <span class="compare-side-label">cached ttft</span>
      <span class="compare-side-value">${cached.ttft_ms != null ? fmtMs(cached.ttft_ms) : (compareFailed.cached ? "error" : "—")}</span>
    </div>
    <div class="compare-side-metric">
      <span class="compare-side-label">raw in</span>
      <span class="compare-side-value">${raw.prompt_tokens != null ? `${raw.prompt_tokens} tok` : "—"}</span>
    </div>
    <div class="compare-side-metric">
      <span class="compare-side-label">cached in</span>
      <span class="compare-side-value">${cached.prompt_tokens != null ? `${cached.prompt_tokens} tok` : "—"}</span>
    </div>
    <div class="compare-side-metric">
      <span class="compare-side-label">raw total</span>
      <span class="compare-side-value">${raw.total_ms != null ? fmtMs(raw.total_ms) : "—"}</span>
    </div>
    <div class="compare-side-metric">
      <span class="compare-side-label">cached total</span>
      <span class="compare-side-value">${cached.total_ms != null ? fmtMs(cached.total_ms) : "—"}</span>
    </div>
  `;

  if (hasError) {
    const rerun = document.createElement("button");
    rerun.className = "rerun-btn";
    rerun.textContent = "↻ retry failed side";
    rerun.onclick = () => rerunCompare();
    metricsBox.appendChild(rerun);
  }
}

function rerunCompare() {
  if (!lastComparePrompt) return;
  const sys = $("#system-prompt")?.value?.trim() || "";
  const payload = {
    mode: "compare",
    prompt: lastComparePrompt,
    system: sys || null,
    max_tokens: parseInt($("#max-tokens").value, 10),
    temperature: parseFloat($("#temperature").value),
    history: history,
    session_id: getSessionId(),
  };
  if (ws && ws.readyState === WebSocket.OPEN) {
    setBusy(true);
    renderComparePair();
    ws.send(JSON.stringify(payload));
  }
}

function onSummary(msg) {
  setBusy(false);
  if (!comparePair) return;

  const banner = document.createElement("div");
  banner.className = "summary-banner";

  if (msg.aborted) {
    banner.classList.add("neutral");
    const abortedEl = document.createElement("span");
    abortedEl.textContent = "stopped mid-stream";
    banner.appendChild(abortedEl);
    $("#messages").appendChild(banner);
    scrollToBottom();
    return;
  }

  // HONEST: if cached failed, don't fake a speedup.
  if (msg.cached_error) {
    banner.classList.add("error");
    const errEl = document.createElement("span");
    errEl.className = "error-text";
    errEl.textContent = `❌ Cached path failed: ${msg.cached_error}`;
    banner.appendChild(errEl);
    const rawInfo = document.createElement("span");
    rawInfo.textContent = ` (raw ran in ${fmtMs(msg.raw.total_ms)})`;
    banner.appendChild(rawInfo);
    $("#messages").appendChild(banner);
    scrollToBottom();
    return;
  }
  if (msg.raw_error) {
    banner.classList.add("error");
    const errEl = document.createElement("span");
    errEl.className = "error-text";
    errEl.textContent = `❌ Raw path failed: ${msg.raw_error}`;
    banner.appendChild(errEl);
    $("#messages").appendChild(banner);
    scrollToBottom();
    return;
  }

  if (msg.speedup == null) return;

  // Keep context for subsequent turns — use the raw answer, it is the
  // complete one (mirrors the TUI's compare behavior).
  if (comparePanels.raw && comparePanels.raw.fullText) {
    history.push({ role: "assistant", content: comparePanels.raw.fullText, mode: "compare" });
    persistHistory();
  }

  const speedup = msg.speedup;
  const speedupClass = speedup > 1.05 ? "good" : speedup < 0.95 ? "bad" : "neutral";
  banner.classList.add(speedupClass);

  const speedupEl = document.createElement("span");
  speedupEl.className = "speedup " + speedupClass;
  speedupEl.textContent = `${speedup.toFixed(3)}×`;
  banner.appendChild(speedupEl);

  const summaryText = document.createElement("span");
  summaryText.textContent = msg.identical
    ? "outputs byte-identical"
    : `${msg.diff_chars} chars differ (first at char ${msg.first_diff_char})`;
  banner.appendChild(summaryText);

  if (!msg.identical) {
    const warn = document.createElement("span");
    warn.className = "diverge-warn";
    warn.textContent = "⚠ cache changed the model's answer";
    banner.appendChild(warn);
  }

  $("#messages").appendChild(banner);
  scrollToBottom();
}

// ── Rendering ──────────────────────────────────────────────────────────

function clearWelcome() {
  const w = $(".welcome");
  if (w) w.remove();
}

function renderUserMessage(text) {
  clearWelcome();
  const el = document.createElement("div");
  el.className = "message user";
  el.innerHTML = `
    <div class="avatar">You</div>
    <div class="body">
      <div class="header"><span class="role">you</span></div>
      <div class="text"></div>
    </div>
  `;
  el.querySelector(".text").textContent = text;
  $("#messages").appendChild(el);
  scrollToBottom();
}

// Dim centered note (slash-command output, mode-fallback warnings).
function renderSystemNote(text, kind = "") {
  const el = document.createElement("div");
  el.className = "message system";
  el.innerHTML = `<div class="system-note${kind ? " " + kind : ""}"></div>`;
  el.querySelector(".system-note").textContent = text;
  $("#messages").appendChild(el);
  scrollToBottom();
}

function renderAssistantMessageShell(mode) {
  const el = document.createElement("div");
  el.className = "message assistant";
  const tag = mode ? `<span class="mode-tag mode-tag-${mode}">${mode}</span>` : "";
  // The <details class="thinking"> block is created closed and only
  // populated when reasoning tokens arrive. <summary> shows the byte
  // count and acts as the toggle. The block is omitted entirely if
  // no reasoning ever arrives, so non-thinking models see nothing.
  el.innerHTML = `
    <div class="avatar">AI</div>
    <div class="body">
      <div class="header">
        <span class="role">assistant</span>
        ${tag}
      </div>
      <details class="thinking" hidden>
        <summary><span class="think-label">thinking</span><span class="think-bytes"></span></summary>
        <pre class="think-body"></pre>
      </details>
      <div class="text"></div>
      <div class="metrics"></div>
    </div>
  `;
  $("#messages").appendChild(el);
  scrollToBottom();
  return el;
}

function renderComparePair() {
  clearWelcome();
  const pair = document.createElement("div");
  pair.className = "compare-pair";
  pair.innerHTML = `
    <div class="compare-panel raw">
      <div class="label">Raw (direct to llama-server)</div>
      <div class="text"></div>
      <div class="metrics"></div>
    </div>
    <div class="compare-panel cached">
      <div class="label">Cached (via toki daemon)</div>
      <div class="text"></div>
      <div class="metrics"></div>
    </div>
  `;
  $("#messages").appendChild(pair);
  comparePanels = {
    raw: {
      textEl: pair.querySelector(".compare-panel.raw .text"),
      metricsEl: pair.querySelector(".compare-panel.raw .metrics"),
      fullText: "",
    },
    cached: {
      textEl: pair.querySelector(".compare-panel.cached .text"),
      metricsEl: pair.querySelector(".compare-panel.cached .metrics"),
      fullText: "",
    },
  };
  comparePair = pair;
  compareMetrics = { raw: null, cached: null };
  compareFailed = { raw: false, cached: false };
  scrollToBottom();
  return pair;
}

function appendMetrics(el, m, mode, cache) {
  const wrap = el.querySelector(".metrics");
  if (!wrap) return;
  const items = [];
  if (m.prompt_tokens != null) items.push(["in", `${m.prompt_tokens} tok`]);
  if (m.ttft_ms != null) items.push(["ttft", fmtMs(m.ttft_ms)]);
  if (m.total_ms != null) items.push(["total", fmtMs(m.total_ms)]);
  if (m.prefill_tokens != null) items.push(["prefill", `${m.prefill_tokens} tok`]);
  if (m.tokens_generated != null) items.push(["gen", `${m.tokens_generated} tok`]);
  if (m.reuse_depth != null && m.reuse_depth > 0) {
    items.push(["reuse", `${m.reuse_depth} tok`]);
  }
  wrap.innerHTML = "";
  for (const [k, v] of items) {
    const s = document.createElement("span");
    s.textContent = `${k} ${v}`;
    wrap.appendChild(s);
  }
  if (cache) {
    const badge = document.createElement("span");
    badge.className = "cache-badge " + (cache.hit ? "hit" : "miss");
    badge.textContent = cache.hit ? "hit" : "miss";
    if (cache.skipped_tokens > 0) badge.textContent += ` +${cache.skipped_tokens} tok`;
    wrap.appendChild(badge);
  }
}

function fmtMs(ms) {
  if (ms == null) return "—";
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// ── Sending ────────────────────────────────────────────────────────────

function setBusy(b) {
  busy = b;
  $("#send-btn").disabled = b;
  $("#user-input").disabled = b;
  const stopBtn = $("#stop-btn");
  if (stopBtn) {
    stopBtn.hidden = !b;
    stopBtn.disabled = false;
  }
}

function abortRequest() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "abort" }));
  }
  const stopBtn = $("#stop-btn");
  if (stopBtn) stopBtn.disabled = true;
}

const TOKEN_PRESETS = [512, 1024, 2048, 4096];

// ── Slash commands (mirrors the TUI) ─────────────────────────────────

function handleSlash(text) {
  const parts = text.split(/\s+/);
  const cmd = (parts[0] || "").toLowerCase();
  const arg = parts.slice(1).join(" ").trim();
  const tokensEl = () => $("#max-tokens");
  switch (cmd) {
    case "/help":
    case "/?":
      renderSystemNote(
        "Commands: /help /clear /mode [raw|cached|compare] /raw [msg] /cached [msg] " +
        "/compare [msg] /session [id] /session-new /precision /metrics /slots " +
        "/reset /reprobe /max_tokens [64-8192]"
      );
      break;
    case "/clear":
      $("#clear-btn").click();
      renderSystemNote("chat cleared");
      break;
    case "/mode":
      if (arg) setMode(arg);
      else renderSystemNote(`current mode: ${currentMode}`);
      break;
    case "/raw":
    case "/cached":
    case "/compare":
      // TUI parity: switch mode and send the rest of the line.
      setMode(cmd.slice(1));
      if (arg) {
        $("#user-input").value = arg;
        sendMessage();
      } else {
        renderSystemNote(`mode: ${cmd.slice(1)}`);
      }
      break;
    case "/session":
      if (arg) {
        const o = $("#session-override");
        if (o) o.value = arg;
        renderSessionId();
        renderSystemNote(`session set to ${arg}`);
      } else {
        renderSystemNote(`current session: ${getSessionId()}`);
      }
      break;
    case "/session-new":
    case "/newsession":
      clearSession();
      const o = $("#session-override");
      if (o) o.value = "";
      getSessionId();
      renderSessionId();
      resetSessionStats();
      renderSystemNote("new session started");
      break;
    case "/precision":
      runPrecisionCheck(false);
      renderSystemNote("running precision check…");
      break;
    case "/metrics":
      fetchAndRenderMetrics();
      renderSystemNote("metrics refreshed");
      break;
    case "/slots":
      fetchAndRenderSlots();
      renderSystemNote("slot state refreshed");
      break;
    case "/reset":
      authFetch("/v1/cache/reset", { method: "POST" })
        .then(() => {
          fetchAndRenderSlots();
          renderSystemNote("cache reset ok");
        })
        .catch(() => renderSystemNote("cache reset failed", "error"));
      break;
    case "/reprobe":
      probeHealth();
      renderSystemNote("backend health re-probed");
      break;
    case "/max_tokens":
    case "/maxtokens":
      if (!arg) {
        renderSystemNote(`current max_tokens: ${tokensEl().value}`);
        break;
      }
      const v = parseInt(arg, 10);
      if (!v || v < 64 || v > 8192) {
        renderSystemNote("max_tokens must be 64-8192", "error");
        break;
      }
      tokensEl().value = v;
      syncTokenPreset();
      renderSystemNote(`max_tokens set to ${v}`);
      break;
    case "/quit":
    case "/exit":
      renderSystemNote("/quit is not available in the web UI — just close the tab");
      break;
    default:
      renderSystemNote(`unknown command: ${cmd}  (try /help)`, "error");
  }
}

function cycleMaxTokens() {
  const current = parseInt($("#max-tokens").value, 10) || 0;
  let next = TOKEN_PRESETS[0];
  for (const p of TOKEN_PRESETS) {
    if (p > current) { next = p; break; }
  }
  if (next === current) next = TOKEN_PRESETS[0];
  $("#max-tokens").value = next;
  syncTokenPreset();
  renderSystemNote(`max_tokens → ${next}`);
}

function syncTokenPreset() {
  const v = parseInt($("#max-tokens").value, 10);
  $$("#token-presets .token-preset-btn").forEach((b) => {
    b.classList.toggle("active", parseInt(b.dataset.tokens, 10) === v);
  });
}

function sendMessage() {
  if (busy) return;
  const prompt = $("#user-input").value.trim();
  if (!prompt) return;
  $("#user-input").value = "";

  if (prompt.startsWith("/")) {
    handleSlash(prompt);
    return;
  }

  renderUserMessage(prompt);

  if (currentMode === "compare") {
    renderComparePair();
    lastComparePrompt = prompt;
  } else {
    activeMsgEl = null;
    activeTextEl = null;
  }

  setBusy(true);
  history.push({ role: "user", content: prompt });
  persistHistory();

  const payload = {
    mode: currentMode,
    prompt,
    system: $("#system-prompt").value.trim() || null,
    max_tokens: Math.min(8192, Math.max(64, parseInt($("#max-tokens").value, 10) || 2048)),
    temperature: parseFloat($("#temperature").value),
    // Cap the context window at the last 20 messages so long sessions
    // stay well under the context budget (mirrors the TUI).
    history: history.slice(0, -1).slice(-20),
    session_id: getSessionId(),
    // No auth_token in the payload: the web server holds its own
    // credential and the WebSocket is gated by the session cookie.
    // The token never crosses the socket.
  };
  renderSessionId();
  try {
    ws.send(JSON.stringify(payload));
  } catch (e) {
    setBusy(false);
    onError("websocket closed");
  }
}

function onError(msg) {
  // Defensive: always clear busy state. The server may send `error`
  // without a corresponding `done` (e.g., upstream 401), so we reset
  // here to avoid the user getting stuck with a disabled send button.
  setBusy(false);
  if (activeTextEl) {
    activeTextEl.classList.remove("streaming-cursor");
    activeTextEl.classList.add("error");
    activeTextEl.textContent = `[error] ${msg}`;
  } else {
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = `
      <div class="avatar">AI</div>
      <div class="body"><div class="text error">[error] ${escapeHtml(msg)}</div></div>
    `;
    $("#messages").appendChild(el);
    scrollToBottom();
  }
  activeMsgEl = null;
  activeTextEl = null;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ── Autoscroll ─────────────────────────────────────────────────────────

let autoScroll = true;

function scrollToBottom() {
  if (!autoScroll) return;
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
}

function onMessagesScroll() {
  const m = $("#messages");
  autoScroll = m.scrollHeight - m.scrollTop - m.clientHeight < 40;
}

// ── UI plumbing ────────────────────────────────────────────────────────

function setMode(mode) {
  // Cache-requiring modes are useless with the daemon down (mirrors the
  // TUI's _set_mode guard).
  if (mode !== "raw" && mode !== "cached" && mode !== "compare") {
    renderSystemNote(`unknown mode: ${mode}  (raw | cached | compare)`, "error");
    return;
  }
  if ((mode === "cached" || mode === "compare") && !daemonOk) {
    renderSystemNote(`cannot switch to ${mode}: daemon unreachable`, "warn");
    return;
  }
  currentMode = mode;
  $$(".mode-btn").forEach((b) => {
    const active = b.dataset.mode === mode;
    b.setAttribute("aria-selected", active ? "true" : "false");
  });
  const pill = $("#status-pill");
  pill.className = `status-pill status-${mode}`;
  pill.textContent = mode;
}

function loadConfig() {
  fetch("/config").then((r) => r.json()).then((cfg) => {
    $("#raw-url").value = cfg.raw_url;
    $("#daemon-url").value = cfg.daemon_url;
  }).catch(() => {});
}

// ── Slot polling ──────────────────────────────────────────────────────

const SLOT_FETCH_INTERVAL_MS = 1500;
let slotFetchTimer = null;

function fetchAndRenderSlots() {
  authFetch("/v1/cache/state").then((r) => {
    if (!r.ok) throw new Error("status " + r.status);
    return r.json();
  }).then((state) => {
    renderSlots(state);
    if (state.daemon_status === "auth_required") {
      // Token is wrong or missing — surface a hint.
      const daemonEl = $("#daemon-status");
      const auth = getAuthToken() ? "wrong" : "required";
      if (daemonEl && daemonEl.className.indexOf("status-auth_") === -1) {
        daemonEl.textContent = auth;
        daemonEl.className = "status-badge " +
          (auth === "wrong" ? "status-auth_wrong" : "status-auth_required");
      }
    }
  }).catch(() => {
    renderSlots({ slots: [], sessions: {}, spill_count: 0 });
  });
}

function renderSlots(state) {
  const list = $("#slots-list");
  const slots = state.slots || [];
  const sessions = state.sessions || {};
  const spillCount = state.spill_count || 0;
  const maxCtx = state.max_context_tokens || 4096;

  updateCtxPill(state, slots, sessions, maxCtx);

  if (slots.length === 0) {
    list.innerHTML = `<div class="slots-empty">no active slots</div>`;
  } else {
    list.innerHTML = "";
    for (const s of slots) {
      const pct = maxCtx > 0 ? Math.min(100, (s.context_tokens / maxCtx) * 100) : 0;
      const high = pct > 75;
      const sessionForSlot = Object.entries(sessions).find(([_, sid]) => sid === s.slot_id);
      const sessionLabel = sessionForSlot ? `${sessionForSlot[0]}` : "";
      const card = document.createElement("div");
      card.className = "slot-card";
      card.innerHTML = `
        <div class="slot-card-header">
          <span class="slot-id">slot ${s.slot_id}${s.reserved ? " ⭐" : ""}</span>
          <span class="slot-pct" style="color: ${high ? "var(--raw)" : "var(--text-secondary)"}">${pct.toFixed(0)}%</span>
        </div>
        <div class="slot-context-bar">
          <div class="slot-context-fill ${high ? "high" : ""}" style="width: ${pct.toFixed(1)}%"></div>
        </div>
        <div class="slot-stats">
          <span>${s.context_tokens} tok</span>
          <span>${s.use_count} uses</span>
          ${sessionLabel ? `<span>session: ${escapeHtml(sessionLabel)}</span>` : ""}
        </div>
      `;
      list.appendChild(card);
    }
  }
  $("#slots-stats").textContent = `${slots.length} slot${slots.length === 1 ? "" : "s"} · ${spillCount} spill${spillCount === 1 ? "" : "s"}`;
}

function updateCtxPill(state, slots, sessions, maxCtx) {
  const pill = $("#ctx-pill");
  if (!pill) return;
  const slotId = sessions[getSessionId()];
  const slot = slots.find((s) => String(s.slot_id) === String(slotId));
  if (slot && maxCtx > 0) {
    const pct = Math.min(100, (slot.context_tokens / maxCtx) * 100);
    pill.textContent = `ctx ${pct.toFixed(0)}%`;
    pill.className = "ctx-pill" + (pct > 75 ? " ctx-high" : "");
  } else {
    pill.textContent = "ctx —";
    pill.className = "ctx-pill";
  }
}

function startSlotPolling() {
  if (slotFetchTimer) return;
  fetchAndRenderSlots();
  slotFetchTimer = setInterval(fetchAndRenderSlots, SLOT_FETCH_INTERVAL_MS);
}

// ── Live metrics ──────────────────────────────────────────────────────

const METRICS_FETCH_INTERVAL_MS = 2000;
let metricsFetchTimer = null;
const ttftSamples = [];  // rolling buffer of last 20 TTFTs

function fetchAndRenderMetrics() {
  authFetch("/v1/metrics/live").then((r) => {
    if (!r.ok) return null;
    return r.json();
  }).then((m) => {
    if (!m) return;
    $("#metric-requests").textContent = m.requests_total || 0;
    $("#metric-hits").textContent = m.cache_hits || 0;
    $("#metric-misses").textContent = m.cache_misses || 0;
    $("#metric-skipped").textContent = m.tokens_skipped || 0;
    $("#metric-slots").textContent = m.active_slots || 0;
  }).catch(() => {});
}

function startMetricsPolling() {
  if (metricsFetchTimer) return;
  fetchAndRenderMetrics();
  metricsFetchTimer = setInterval(fetchAndRenderMetrics, METRICS_FETCH_INTERVAL_MS);
}

function pushTtftSample(ttftMs) {
  ttftSamples.push(ttftMs);
  if (ttftSamples.length > 20) ttftSamples.shift();
  renderSparkline();
}

function renderSparkline() {
  const svg = $("#ttft-sparkline");
  if (!svg) return;
  svg.innerHTML = "";
  if (ttftSamples.length === 0) return;
  const maxV = Math.max(...ttftSamples, 1);
  const w = 200;
  const h = 40;
  const barW = w / 20;
  for (let i = 0; i < ttftSamples.length; i++) {
    const v = ttftSamples[i];
    const barH = Math.max(2, (v / maxV) * h);
    const x = i * barW + 1;
    const y = h - barH;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", barW - 2);
    rect.setAttribute("height", barH);
    rect.setAttribute("class", "spark-bar");
    svg.appendChild(rect);
  }
}

// ── Precision panel ──────────────────────────────────────────────────


async function runPrecisionCheck(silent = false) {
  const btn = $("#precision-btn");
  if (btn) btn.disabled = true;
  if (!silent) {
    $("#prec-hint").textContent = "running…";
  }
  try {
    const maxTokens = parseInt($("#max-tokens").value, 10) || 32;
    const r = await authFetch("/v1/precision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_tokens: Math.min(maxTokens, 64) }),
    });
    if (!r.ok) throw new Error("status " + r.status);
    const body = await r.json();
    if (body.error) {
      if (!silent) $("#prec-hint").textContent = `error: ${body.error}`;
      return;
    }
    renderPrecision(body);
  } catch (e) {
    if (!silent) {
      $("#prec-hint").textContent = `error: ${e.message}`;
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderPrecision(p) {
  const base = (p.baseline_deterministic || 0) * 100;
  const rec = (p.recycled_deterministic || 0) * 100;
  const cross = (p.byte_identical_to_baseline || 0) * 100;
  setPrecisionCell("prec-base", base);
  setPrecisionCell("prec-rec", rec);
  setPrecisionCell("prec-cross", cross);
  $("#prec-hint").textContent =
    `${p.prompts_tested || 3} prompts · 3 runs each · click to re-run`;
}

function setPrecisionCell(id, pct) {
  const el = $("#" + id);
  if (!el) return;
  el.textContent = `${pct.toFixed(0)}%`;
  el.classList.remove("prec-good", "prec-warn", "prec-bad");
  if (pct >= 99) el.classList.add("prec-good");
  else if (pct >= 80) el.classList.add("prec-warn");
  else el.classList.add("prec-bad");
}

// ── History persistence ──────────────────────────────────────────────

function persistHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch (e) { /* quota exceeded — ignore */ }
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return;
    history = JSON.parse(raw).slice(-100);
    // Re-render messages into the chat.
    if (history.length === 0) return;
    clearWelcome();
    for (const m of history) {
      if (m.role === "user") renderUserMessage(m.content);
      else if (m.role === "assistant") {
        const el = renderAssistantMessageShell("");
        el.querySelector(".text").textContent = m.content;
      }
    }
    scrollToBottom();
  } catch (e) { /* corrupt — ignore */ }
}

// ── Init ──────────────────────────────────────────────────────────────

// ── Guaranteed chatbox fallback ────────────────────────────────────────
// Browsers have repeatedly served stale mixes of old HTML + old CSS that
// hide or omit the input area, making it impossible to type. Before any
// other init runs, verify the chatbox actually exists and is visible;
// otherwise force it back into view or rebuild a working one. This runs
// first so all later listener wiring binds to the final elements.

function ensureChatbox() {
  const inputArea = $(".input-area");
  const input = $("#user-input");
  let problem = null;

  if (input) {
    const r = input.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) {
      problem = "input has zero size (hidden by CSS)";
    } else if (input.disabled) {
      problem = "input is disabled";
    }
  } else {
    problem = "input element missing";
  }
  if (problem === null && !inputArea) {
    problem = "input area missing";
  }

  if (problem) {
    console.warn("[tokirecur] chatbox fallback:", problem);
  }

  // Forced visibility: inline styles beat any stylesheet, stale or not.
  // Use `!important` too, so even extension-added rules cannot hide it.
  if (inputArea && problem) {
    for (const [prop, value] of [
      ["display", "block"],
      ["flex", "none"],
      ["flex-shrink", "0"],
      ["padding", "16px 24px 24px"],
      ["visibility", "visible"],
    ]) {
      inputArea.style.setProperty(prop, value, "important");
    }
  }

  // Element or container truly missing: rebuild a fixed chatbox pinned
  // to the bottom of the viewport. All existing handlers address these
  // elements by id at call time, so normal wiring keeps working.
  if (!input || !inputArea) {
    console.warn("[tokirecur] rebuilding chatbox");
    const area = document.createElement("div");
    area.className = "input-area";
    area.id = "input-area-fallback";
    area.innerHTML = `
      <div class="input-row">
        <textarea id="user-input" placeholder="Type a message…" rows="2"></textarea>
        <button id="stop-btn" class="ghost-btn" type="button" hidden>Stop</button>
        <button id="send-btn" class="primary-btn">Send</button>
      </div>`;
    for (const [prop, value] of [
      ["position", "fixed"],
      ["left", "0"],
      ["right", "0"],
      ["bottom", "0"],
      ["z-index", "1000"],
      ["display", "block"],
      ["background", "var(--bg-secondary, #13161d)"],
      ["border-top", "1px solid var(--border, #2a2d3a)"],
      ["padding", "16px 24px 24px"],
    ]) {
      area.style.setProperty(prop, value, "important");
    }
    document.body.appendChild(area);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  ensureChatbox();
  connect();
  loadConfig();
  loadHistory();
  renderSessionId();
  startSlotPolling();
  startMetricsPolling();
  probeHealth();
  setInterval(probeHealth, 5000);

  $$(".mode-btn").forEach((b) => {
    b.addEventListener("click", () => setMode(b.dataset.mode));
  });

  $("#send-btn").addEventListener("click", sendMessage);
  $("#user-input").addEventListener("keydown", (e) => {
    if ((e.key === "t" || e.key === "T") && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      cycleMaxTokens();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Max-token presets.
  $$("#token-presets .token-preset-btn").forEach((b) => {
    b.addEventListener("click", () => {
      $("#max-tokens").value = b.dataset.tokens;
      syncTokenPreset();
    });
  });
  syncTokenPreset();

  const stopBtn = $("#stop-btn");
  if (stopBtn) stopBtn.addEventListener("click", abortRequest);
  $("#messages").addEventListener("scroll", onMessagesScroll);

  $("#clear-btn").addEventListener("click", () => {
    history = [];
    localStorage.removeItem(HISTORY_KEY);
    resetSessionStats();
    $("#messages").innerHTML = `
      <div class="welcome">
        <h2>Welcome to Toki Recur</h2>
        <p>Conversation cleared. Type a prompt below.</p>
      </div>
    `;
    autoScroll = true;
  });

  $("#reset-cache-btn").addEventListener("click", () => {
    authFetch("/v1/cache/reset", { method: "POST" }).then(() => {
      fetchAndRenderSlots();
    });
  });

  $("#reset-session-btn").addEventListener("click", () => {
    clearSession();
    renderSessionId();
  });

  $("#copy-session-btn").addEventListener("click", () => {
    const sid = getSessionId();
    navigator.clipboard?.writeText(sid).then(() => {
      const btn = $("#copy-session-btn");
      const orig = btn.textContent;
      btn.textContent = "✓";
      setTimeout(() => { btn.textContent = orig; }, 1000);
    });
  });

  $("#session-id").addEventListener("click", () => {
    navigator.clipboard?.writeText(getSessionId());
  });

  $("#session-override").addEventListener("change", () => {
    renderSessionId();
  });

  // Auth token persistence.
  const tokenInput = $("#auth-token");
  if (tokenInput) {
    tokenInput.value = getAuthToken();
    tokenInput.addEventListener("change", () => {
      setAuthToken(tokenInput.value.trim());
      probeHealth();
      fetchAndRenderSlots();
      fetchAndRenderMetrics();
    });
  }

  // Test connection button.
  const testBtn = $("#test-conn-btn");
  if (testBtn) {
    testBtn.addEventListener("click", async () => {
      testBtn.disabled = true;
      const orig = testBtn.textContent;
      testBtn.textContent = "testing…";
      await probeHealth();
      // Also re-run precision if both backends are healthy.
      const llamaOk = $("#llama-status").classList.contains("status-ok");
      if (llamaOk) {
        await runPrecisionCheck(true);
      }
      testBtn.textContent = "✓";
      setTimeout(() => {
        testBtn.textContent = orig;
        testBtn.disabled = false;
      }, 1500);
    });
  }

  // Precision button.
  $("#precision-btn").addEventListener("click", () => runPrecisionCheck(false));
  $("#refresh-metrics-btn").addEventListener("click", () => fetchAndRenderMetrics());

  // Backend URL changes should re-probe.
  ["raw-url", "daemon-url"].forEach((id) => {
    const el = $("#" + id);
    if (el) el.addEventListener("change", () => probeHealth());
  });

  // ── Collapsible side panels (swipe in/out) ───────────────────────────

  const PANEL_KEYS = { left: "toki:panel:left", right: "toki:panel:right" };
  const PANEL_LIMITS = {
    left: { min: 160, max: 560, open: 340 },
    right: { min: 160, max: 480, open: 280 },
  };

  function panelCollapsed(side) {
    try {
      return localStorage.getItem(PANEL_KEYS[side]) === "1";
    } catch (e) {
      return false;
    }
  }

  function setPanelCollapsed(side, collapsed) {
    const appEl = $("#app");
    if (!appEl) return;
    appEl.classList.toggle(side === "left" ? "left-collapsed" : "right-collapsed", collapsed);
    const handle = $("#" + (side === "left" ? "edge-left" : "edge-right"));
    if (handle) {
      handle.setAttribute("aria-expanded", collapsed ? "false" : "true");
      handle.textContent =
        side === "left"
          ? (collapsed ? "»" : "‹")
          : (collapsed ? "«" : "»");
      // Anchor the handle to the panel's inner edge. The open width is
      // fixed by CSS (PANEL_LIMITS.open), so no measuring mid-transition.
      if (side === "left") {
        handle.style.left = (collapsed ? 0 : PANEL_LIMITS.left.open) + "px";
        handle.style.right = "";
      } else {
        handle.style.right = (collapsed ? 0 : PANEL_LIMITS.right.open) + "px";
        handle.style.left = "";
      }
    }
    try {
      localStorage.setItem(PANEL_KEYS[side], collapsed ? "1" : "0");
    } catch (e) { /* private mode etc. */ }
  }

  function initEdgeHandles() {
    ["left", "right"].forEach((side) => {
      const handle = $("#" + (side === "left" ? "edge-left" : "edge-right"));
      const panel = side === "left" ? $(".sidebar") : $("#slots-panel");
      const appEl = $("#app");
      if (!handle || !panel || !appEl) return;

      const limits = PANEL_LIMITS[side];
      let dragging = false;
      let dragged = false;
      let startX = 0;
      let startWidth = 0;

      handle.addEventListener("click", (e) => {
        if (dragged) {
          dragged = false;
          e.stopPropagation();
          return;
        }
        const collapsed = appEl.classList.contains(side === "left" ? "left-collapsed" : "right-collapsed");
        setPanelCollapsed(side, !collapsed);
      });

      handle.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 && e.pointerType === "mouse") return;
        dragging = true;
        dragged = false;
        startX = e.clientX;
        startWidth = panel.getBoundingClientRect().width;
        appEl.classList.add("dragging");
        try { handle.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      });

      handle.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = side === "left" ? e.clientX - startX : startX - e.clientX;
        if (Math.abs(e.clientX - startX) > 4) dragged = true;
        const width = Math.min(limits.max, Math.max(limits.min, startWidth + dx));
        panel.style.width = width + "px";
      });

      const finish = () => {
        if (!dragging) return;
        dragging = false;
        const width = panel.getBoundingClientRect().width;
        panel.style.width = "";
        appEl.classList.remove("dragging");
        // A pure click (no movement) is handled by the click listener.
        if (!dragged) return;
        setPanelCollapsed(side, width < (limits.min + limits.max) / 2);
      };

      handle.addEventListener("pointerup", finish);
      handle.addEventListener("pointercancel", finish);

      // Restore the saved state on load.
      setPanelCollapsed(side, panelCollapsed(side));
    });
  }

  // ── Stale-HTML repair ────────────────────────────────────────────────
  // If a browser serves a cached copy of an older index.html where the
  // stats panel lived inside <main>, the chat input gets squeezed out of
  // view. Move the panel back into the grid on load so the layout always
  // works, even without a hard refresh.
  (function repairSlotsPanelPlacement() {
    const slots = $("#slots-panel");
    const main = $(".main");
    const appEl = $("#app");
    if (!slots || !main || !appEl) return;
    if (main.contains(slots)) {
      main.removeChild(slots);
      appEl.appendChild(slots);
    }
  })();

  initEdgeHandles();
});
