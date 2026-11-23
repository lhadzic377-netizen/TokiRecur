# Changelog

All notable changes to TokiRecur are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.24] - 2026-11-23

### Added
- **`p` in the launcher: a public HTTPS origin for this machine.** The
  custom-endpoint form in other apps (arkenOS among them) refuses a
  loopback base URL - it wants a public HTTPS URL on port 443 - so the
  one thing it cannot be given is the daemon it should talk to. `p` opens
  a public-server screen; Enter starts a `cloudflared` quick tunnel to
  the pinned model's daemon and the screen then shows the credentials for
  all three surfaces on that hostname: OpenAI (`/v1/chat/completions`,
  `/v1/responses`), Anthropic (`/v1/messages`) and Google generative AI
  (`/v1beta/models/<id>:generateContent` and `:streamGenerateContent`),
  plus the daemon's bearer token and model id. `c` copies the block. The
  hostname is random and dies with the process; closing the screen (or
  quitting toki) stops the tunnel, and every request still needs the key.
  The screen starts nothing on open - an outbound tunnel is a deliberate
  act - and says so when `cloudflared` is missing or the daemon is down.

## [0.3.23] - 2026-11-22

### Fixed
- **A harness that walks away no longer keeps the GPU generating.** Killed
  coding-agent clients left their requests streaming into llama-server
  with nobody reading: each abandoned turn ran to its full token budget
  (measured ~2k tokens, ~90s on Inkling) and pinned a slot. Two of them
  wedged every harness behind the same pair of slots - health checks
  stayed green, completions queued forever, and only a stack restart
  recovered. Three things were wrong at once: the shared, pooled httpx
  client kept the upstream TCP socket open when its producer was
  cancelled, so llama-server never saw the disconnect; Starlette's
  `StreamingResponse` cancels the body task on disconnect but leaves its
  async generator suspended at the last `yield`, so the route's cleanup
  never ran; and the engine left its slot claimed when finalization was
  skipped. The streaming path now uses a client dedicated to one request
  (cancelling it closes the socket), every SSE route closes its relay
  through a `ClosingStreamingResponse`, and the engine releases its slot
  bookkeeping on cancellation or backend failure. Measured on the pinned
  build: a slot is free 0.25s after the client aborts, three aborts in a
  row leave both slots usable, and the next turn answers normally.
- **Slot routing uses the field llama-server actually reads.** Toki sent
  `slot_id` in every completion payload, but llama-server reads
  `id_slot`; it silently ignored the routing decision and auto-assigned
  a slot. That defeated prefix reuse and made a later slot reset hit a
  different slot than the one the engine meant. All completion payloads
  now send `id_slot`.
- **`clear_slot` calls the endpoint llama-server supports.** `?action=clear`
  is rejected with HTTP 400 ("Invalid action"); `?action=erase` is the
  real KV reset. Note that an erase is queued behind a task still
  generating on that slot, so it is not a cancellation primitive - the
  streaming path cancels by closing the HTTP connection instead.

- **Reasoning reaches codex again.** The Responses stream carried the
  model's reasoning as deltas attached to the assistant *message* item,
  and codex's parser drops those: the model thought, the turn looked
  empty, and nothing was on screen. The stream now opens a real
  reasoning item (index 0) with a summary part, sends the
  spec-shaped `reasoning_summary_text.delta` events (`delta` plus
  `summary_index`, addressed to the reasoning item), closes that item
  - before the assistant message opens, the order the Responses API
  uses - and then emits the message as index 1. Tool calls move past
  the reasoning item. A turn that thinks out loud is now visible while
  it happens.
- **A ROCm install no longer carries a CUDA knob.** The ROCm setup
  wrote `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` into the model's
  environment. That variable belongs to the CUDA backend, but the HIP
  backend reads the same name and turns on a unified-memory path that
  silently corrupts inference on some AMD GPUs: the server answers,
  every token is repeated garbage, and the reply comes back empty -
  with nothing in the logs to explain it. The variable is gone from
  new installs, and the registry repair takes it out of the entries
  that older installs already wrote (only when the model's build is
  ROCm, so a CUDA box keeps the knob it wants).
- **The registry repair no longer throws away a pinned build.** A
  model that pins its own `llama-server` - because it needs a
  different llama.cpp, or a locally tuned one - lost that pin the
  next time the repair ran, and silently started on the default build
  instead. Repairing the environment now leaves the pinned binary
  alone.
- **A model the installed build cannot load says so.** A llama.cpp
  build only knows the architectures that existed when it was compiled.
  Inkling 2 Mini asks for `inkling2_loop`, which arrived upstream in
  ggml-org/llama.cpp#32694; a build from before that merge dies
  inside the server with "unknown model architecture", which reads
  like a truncated download and sends people hunting for the wrong
  file. The launcher now refuses before the process starts, naming the
  architecture the GGUF asks for and the two ways out: a newer
  llama.cpp release, or a rebuild from source.

## [0.3.22] - 2026-11-20

### Added
- **Goose and Pool run on the local model.** Two more harnesses join
  the launcher, the client-config family and the theme list - Block's
  goose and Poolside's pool. Neither is an opencode clone, so neither
  takes a provider block in a JSON file. goose reads a provider from
  `~/.config/goose/custom_providers/tokirecur.json` and picks it with
  `GOOSE_PROVIDER` and `GOOSE_MODEL` in `config.yaml`; its `base_url`
  is the full request path - goose appends nothing to it, so a URL
  that stops at `/v1` answers 404 - and its key is read from the
  `TOKI_AUTH_TOKEN` environment variable, so no secret is written to
  disk. pool only speaks to an OpenAI-compatible endpoint in
  standalone mode, and only the environment selects that mode: the
  launcher starts it with `POOLSIDE_STANDALONE_BASE_URL`,
  `POOLSIDE_API_KEY`, `POOLSIDE_STANDALONE_MODEL` and
  `POOLSIDE_STANDALONE_CONTEXT_LENGTH`, and
  `~/.config/poolside/settings.yaml` keeps the saved API URL next to
  the user's own agent servers and MCP servers. Both were verified
  against a local llama-server: goose answered "HELLO FROM GOOSE" and
  pool answered "HELLO FROM POOL".
- **Antigravity runs on the local model.** Google's agentic IDE has a
  terminal CLI (`agy`), and it is the one harness that speaks neither
  the OpenAI API nor the Anthropic one: it asks for
  `/v1beta/models/<model>:generateContent` in Gemini's shape. The
  daemon serves that route now - the request is translated into the
  engine's shape and the answer back into Gemini's, and the stream
  ends when the server closes it, because Gemini has no `[DONE]`
  sentinel and a client that reads one as JSON fails on it. The key is
  accepted in `x-goog-api-key`, which is where the Google SDKs carry
  it. Antigravity is pointed at the daemon with a custom model in
  `~/.gemini/antigravity-cli/settings.json` and
  `GOOGLE_GEMINI_BASE_URL` in the environment; the user's own
  permissions, colour scheme and models stay as they are. It always
  routes through the daemon: the Gemini route lives there, not on
  llama-server, and its launcher row says `daemon` for the same
  reason.
- **A theme for each of them.** `goose` wears the blue, teal and
  violet its binary carries on a warm near-black; `pool` wears the
  blue of its logo on the dark it ships with; `antigravity` wears the
  mint, the amber and the blue of the Tokyo Night it ships with. All
  three are in the settings pane's theme list like every other theme.

### Fixed
- **Antigravity starts on the local model, not a hosted fallback.**
  The CLI reads `model` from its settings file as an alias name, and
  "tokirecur" is not one - so a plain `agy` start resolves it to a
  hosted Gemini model and says the model "is no longer available".
  The launcher names the custom model on the command line
  (`agy --model tokirecur`), which selects it without the warning.
- **A client whose API lives on the daemon cannot be told to go
  raw.** codex speaks the Responses API and antigravity the Gemini
  API, and both routes exist only on the daemon. Those launcher rows
  say `daemon` instead of whatever mode was last stored, `m` explains
  the pin instead of storing `raw`, and `toki model mode <client> raw`
  refuses rather than writing a mode that cannot work.

## [0.3.21] - 2026-11-19

### Added
- **GTA VI says its name again.** The lettering under the wordmark is
  back - "GTA VI" - with the horizon under it, the way the theme was
  first drawn. It names the mood; it is not an edition.

### Fixed
- **An adapter finds its base model on Windows.** The family of a model
  is read from the last segment of its path, but only a POSIX separator
  was split on - so on Windows the family came out as the whole
  directory, and installing an adapter said "no installed model matches
  this adapter's target" for a base that was installed and named
  correctly. A path is split on either separator now, and the same
  applies to the model's own name when it is compared with the name
  llama-server reports: a Windows path never matched, so the launcher
  said a running model was not running.
- **The install tests look for the build the installer looks for.** The
  installer reads `llama-server.exe` on Windows and `llama-server`
  everywhere else; the tests planted the POSIX name on every platform,
  so on a Windows box the installer never saw the build the test had
  planted and the test measured the wrong thing.
- **The settings tests wait for the commit.** The launcher writes the
  registry in a worker - a restart waits for llama-server - so the
  value is not there the moment the modal closes. On a slow machine the
  test read the spec before the worker had written it, and reported a
  commit that had happened. The tests now wait for the worker, and the
  stack work the commit does is stubbed: a restart that waits a minute
  for a server that is not there is not what those tests are about.

## [0.3.20] - 2026-11-15

### Added
- **A theme per harness, listed in the settings pane.** The launcher,
  the chat TUI, the CLI and the web dashboard now share one palette
  that a theme can replace. Every harness toki launches has a theme of
  its own, taken from the colours that harness actually wears -
  Claude's warm clay, opencode's cream on charcoal, Kimi's electric
  blue, openclaw's amber on black, hermes' petrol teal, Qwen's violet,
  MiMo's orange, Grok's Tokyo Night purple - plus the house style and
  GTA VI, which is a mood like the others and not a limited edition.
  The settings pane lists them under `theme` and picking one repaints
  the launcher where it stands; `toki theme` lists the same set from
  the shell, `toki theme <name>` switches the whole box, and `TOKI_THEME`
  applies a theme for one run. The slots keep their names whatever the
  theme - `YELLOW` is the primary accent - so a theme is a palette and
  not a rename, and the palette is read at the moment it is used, so
  the switch takes effect without a restart. The Textual stylesheet is
  rendered from the active palette, so a theme reaches the widgets that
  are styled in CSS as well as the ones styled in Python. GTA VI is more
  than a palette: the wordmark is drawn in a heavier face and painted
  in a sunset - the sun at the top of the letters, the sand through the
  middle, the sea at the foot - with the horizon under it drawn as a
  rule in the same colours.
- **MiniMax Code runs on the local model.** `mcode` joins the launcher
  and the model-capable client family. `toki model use` writes a
  `tokirecur` provider under `custom_provider` in
  `~/.minimax/config.yaml` - the user's own MiniMax provider, key and
  preferences are left alone - and the session model is
  provider-qualified (`custom_provider:tokirecur/<model-id>`), because a
  bare model id resolves against the signed-in MiniMax account instead
  of the local route. The launcher starts it with
  `mcode --model custom_provider:tokirecur/<model-id>`; the token
  travels in the config, which stays owner-only.

### Fixed
- **The doctor says why a model will not start.** "The model is not
  running" is not an answer: a build that does not know the model's
  architecture refuses the file in milliseconds and the daemon never
  starts, so the model looks like it is simply off. The doctor reads
  the model's own log now and says what went wrong - the architecture
  it does not know, and that the model needs a build that supports it
  (often a fork), which the settings' backend dropdown can point at.
- **OpenClaw starts again.** The installed OpenClaw moved its agent
  roster from a list to a map keyed by the agent's id, and it refuses
  to start on the old shape - "OpenClaw config is invalid". toki is
  what writes that file, so it wrote a config its own launcher could
  not open. The roster is migrated to the keyed shape now, and the
  keys the installed version has retired are repaired with it (the
  ownership marker a multi-agent roster needs, the retired
  `gateway.tailscale.resetOnExit`, the old name for the gateway's
  reload mode). The roster itself is the user's and is kept.
- **The backend is a dropdown now, and it follows the hardware.** The
  settings pane asked for a path and never said what the build was, so
  a model could be pointed at a CPU build and keep answering - ten to
  thirty times slower, with nothing in the output to say why. The
  field is a list of the backends this machine can actually serve:
  the card's own runtime first (ROCm on an AMD box), then the portable
  ones whose driver is installed here (Vulkan, when the matching ICD
  is present), then the CPU build that always works. Picking one
  installs that build for the model - the download, the env block and
  the self-check are the installer's job - and points the model at it;
  a backend this machine cannot run is not offered at all. The free
  text path is still there for a build from somewhere else.
- **The settings pane no longer lets a model point at something that
  is not a build.** The backend build is the one setting that can make
  a stack unstartable, and a path that is not a build was written to
  the registry like any other value - the launcher then reported a
  restart failure instead of the mistake. The path is checked before
  the registry is written, the settings row says what the build
  actually is (ROCm, CUDA, Vulkan, CPU - the path alone does not say
  whether the model will run on the GPU), and a swap onto a CPU-only
  build says what it means: the model still answers, ten to thirty
  times slower, with nothing in the output to say why.
- **Two requests no longer wait for each other.** The daemon held one
  lock across the whole generation, so a second client was answered
  only after the first had finished - one request at a time, whatever
  the number of slots llama-server had. The engine now locks only its
  routing state, claims a slot for the length of the request and
  releases it afterwards, so the clients overlap as far as the server
  and the hardware allow. Measured on this box with the model resident
  on the GPU and four concurrent clients: 651 ms (4.03x one request -
  no gain at all over one client at a time) became 456 ms (2.72x one
  request, 1.47x the throughput of a single client); two clients went
  from 2.06x to 1.38x. The same holds on a CPU-served model: 638 ms
  became 444 ms (1.40x the throughput of one client). The streaming
  path never took that lock and mutated the slot map without any lock
  at all, so two streams raced on it; the routing state is locked now,
  and a slot with a request in flight is never spilled. In tokens per
  second, on the same GPU stack: two clients at once went from 28.5
  tok/s of aggregate output (the daemon answered them one after the
  other) to 49.3 tok/s - the same per-request rate, twice the work
  done. A single request decodes at 29 tok/s and prefills at 1077
  tok/s on this box.
- **The hot-prefix pool asks the server that is actually running.** It
  looked for a free slot on port 8088 - where nothing listens on a
  machine whose model sits on its own port - so it never found one and
  the pool never claimed a slot.
- **A prompt that does not fit is a 413, not a 500.** llama.cpp words
  the overflow "request (N tokens) exceeds the available context size
  (M tokens)", and the daemon did not know that wording - so an
  oversized prompt came back as a 500 whose body no OpenAI-compatible
  client can read ("413 status code (no body)" in MiniMax Code), and a
  streaming client saw the turn die mid-stream and concluded the
  provider was gone. The overflow is recognised now, it is checked
  before the response headers go out - so the client gets the status
  and can compact and retry - it is counted the way llama-server builds
  the prompt (the chat template plus the tool schemas, which on an
  agent client are more tokens than the conversation), and the error
  body carries `error.message`, the shape OpenAI clients read.
- **The prompt-size probe no longer 404s.** Clients that show the
  prompt size before a turn (MiniMax Code does) ask the daemon to count
  the tokens of a request. The route did not exist, so the client got a
  404, retried it, and never showed the number. `POST
  /v1/responses/input_tokens` counts the prompt now, and a Responses
  input item no longer has to carry a `type` - clients that count
  tokens send `{"role": ..., "content": ...}` - and a string body is
  read as the message it is.

### Changed
- **The overflow check stopped counting the whole prompt.** Tokenizing
  a rendered prompt costs tens of milliseconds on an agent-sized one
  (39 ms for 106k characters on this box), and the count is only
  needed to tell "fits" from "does not fit". The prompt is now sampled
  - spread over its length, so a prompt that is prose at the front and
  tool schemas in the middle is represented - and scaled: 39 ms became
  19 ms here and 77 ms became 15 ms on a larger one, with the
  estimate within 1% of the true count. A prompt that lands inside
  the margin around the window is still counted exactly.
- **The daemon accepts both of the machine's tokens.** A daemon
  started with `TOKI_AUTH_TOKEN` exported and a client configured
  from the auth file used to disagree, and every request between them
  was a 401 - the "Reconnecting" class of failure. Both are the
  owner's own tokens and the file is owner-only, so either one is
  accepted now.
- **One tokenization per request instead of two.** The routing text was
  tokenized for the slot match and then tokenized again inside the
  embedding, which is a round-trip to llama-server for nothing. The
  token list is handed over now.
- **The streaming path keeps its connection.** It opened a new HTTP
  client - and a new TCP connection - for every request, so the
  keep-alive that would have made the next request cheap went with it.
  The slot watcher shares it too.
- **The prefix comparison is done in blocks.** A 20k-token prompt was
  compared one token at a time in Python, per slot: 0.78 ms each. The
  comparison now runs on slices, 0.16 ms for the same prompt, and only
  the block holding the first difference is walked token by token.
- **The tap measures its own directory every 25 writes.** It stat-ed
  every file in the tap directory on every request to enforce a
  ceiling that is a soft one anyway.

## [0.3.16] - 2026-11-13

### Fixed
- **Pinning a model no longer leaves the clients on the old one.** The
  launcher's model picker moved the active marker and stopped there,
  while the model-capable clients read the active model out of their own
  config files. A client started outside toki - from a shell, not from
  the launcher - kept talking to the model that was active before the
  pin, over a port that was no longer served: "Reconnecting", or a
  silent fall back to another provider. A pin now refreshes every
  client config, off the event loop so the launcher keeps answering
  keys.

## [0.3.15] - 2026-11-13

### Added
- **Adapters are selectable modes of their base model.** An adapter is
  a second GGUF tied to one target: a DSpine/DFlash2 draft reads the
  base's hidden states, a LoRA patches its weights, and neither can
  serve a prompt alone. The registry now keeps an `adapters:` catalog
  per model and one active `adapter_mode:`; the active entry is
  projected onto the flat `draft*` fields, so launch and fit read one
  thing and a hand-written `draft:` registry still resolves as a mode.
  The launcher's settings pane lists `base model (no adapter)` plus
  every installed adapter — and any adapter GGUF found beside the
  model — and picking one restarts the stack with it. `toki model
  adapter add|list|use|remove` manages the catalog; `toki model pull`
  attaches a DSpine adapter to its installed target as a mode and, when
  the target is missing, says which base to pull first. An adapter that
  cannot be attached (missing file, stale id, unknown kind, build
  without DSpine) is refused before the launch. The fit planner and
  RAM gate reserve the active adapter's weights, its own 8k KV window,
  and the target's recurrent-state copies per drafted token. `--draft`
  fetches and registers the mode even on a stock build but keeps the
  model on its base; `--draft-quant` and `--draft-n-max` choose the
  file and block size (default 15 on CUDA/ROCm/SYCL, 3 on Vulkan). The
  model row shows the active mode's badge.

### Fixed
- **Two harnesses no longer keep talking to a model that is gone.** OMP
  keeps the providers in `models.yml` and the model each role uses in
  `config.yml`, so writing only the provider left the role pointing at
  whatever was active when it was written - and when that model was
  removed, OMP quietly fell back to its own Ollama on port 11434.
  Droid keeps the model a session starts with under a slug of the
  custom model's display name, so it kept naming the old one and
  answered "No active subscription found" instead of using the local
  route. Both writers now refresh the selection as well as the
  provider list.
- **A mid-conversation system message no longer 500s the model.** The
  chat templates raise on a system message that is not the first one -
  "System message must be at the beginning" - and the request fails
  with a Jinja error that reads like a server fault. The clients that
  trigger it inject context mid-turn: Claude Code sends a hook result as
  a system message, and Codex sends its work state the same way. Both
  are really user-visible context, so they are demoted to a user turn;
  the leading instruction preamble stays where it was.
- **Qwen Code 1.0 launches on the local model again.** The 1.0 preview
  replaced the `modelProviders` settings section with `providers` and
  removed the `--openai-base-url` and `--auth-type` flags, so the old
  route was silently ignored and the launch exited 1 on an unknown
  option. The writer now fills `providers` as well: a provider entry
  with `protocol`, `baseUrl`, `apiKey` and a `models` list keyed by
  `model` (not `id`). The token is written literally, because 1.0
  rejects `${VAR}` as unresolved and passes a bare `$VAR` through
  untouched; the settings file stays owner-only (0600). The saved model
  name is provider-qualified (`tokirecur/<model>`), since 1.0 resolves a
  bare id against the signed-in account and fails with `belongs to
  provider "qoder"`. The launcher detects the installed version and
  passes `qwen --model tokirecur/<model>` on 1.0, keeping the endpoint
  flags for the releases that still accept them. The pre-1.0 section is
  still written, so a downgrade keeps a live route.
- **The model picker no longer crashes on a small registry.** With one
  or two models registered — a normal state after `toki model rm`, or
  with a custom registry — the picker windowed three rows over a
  two-item list, walked off the end, and took the whole launcher down
  with `IndexError`. The window is now clamped to the list, the way the
  settings pane's already was.
- **A settings edit no longer freezes the launcher.** Changing a
  restart-required field stopped and started llama-server, which waits
  up to a minute for the server to come back — and it ran on the event
  loop, so the launcher stopped rendering and stopped answering keys
  for the whole restart. The write now runs in a worker.
- **A failed turn no longer locks the input.** The streaming loop was
  guarded, but the finalize path was not: an error writing the history
  left the input disabled for good, and no key reset it.
- **The web UI answers only the hosts it serves.** Comparing `Origin`
  against `Host` is not a defence on its own — under DNS rebinding
  both are the attacker's, so the check passed and the attacker's page
  got to chat on the victim's GPU and read the cache state through a
  proxy that attaches the daemon's own token. The `Host` header is now
  checked against the names we serve.
- **A non-loopback web bind asks for the token.** The UI is open by
  design and authenticates its own proxied calls with the daemon's
  credential, which is fine on loopback and an open, token-lending
  proxy anywhere else. `toki web --host 0.0.0.0` now requires the
  token and tells the user how to pass it.
- **`model remove --purge` can no longer delete the models tree.** The
  purge target came from the raw registry string, so an entry with a
  bare filename resolved to the current directory — running the command
  from `~/models` deleted every model's weights. The target is now the
  resolved weights directory, and it has to be a per-model subdirectory.
- **A pulled split model is registered under its own name.** The loop
  over the linked parts rebound the model name, so the entry was saved
  under a part's filename, `--name` was silently dropped, and pulling
  the same repository again created a duplicate.
- **A corrupt model file cannot crash the fit planner.** The GGUF
  header parser read a length straight from the file — a hostile header
  asked for exabytes and raised `MemoryError`, which no except clause
  caught. The length is capped now, and the memory errors are handled.
- **The reserved slots are reserved again.** The persistent pool's
  slots are keyed by cache id, but the check looked an int slot id up in
  that string-keyed map, so it was never true and the pool's hot-prefix
  slots could be spilled over.
- **An AMD GPU is not demoted to the CPU over a CSV parse.** The VRAM
  reader required the label and the number on one line, and the real
  `rocm-smi --csv` puts them in a header and a data row, so the GPU
  came back with no VRAM and failed the 4 GB gate.
- **A broken cache index no longer crashes every command.** A corrupt
  SQLite file took `toki stats` and the daemon down with it; the index
  now quarantines it and rebuilds, the way a corrupt registry already
  did.
- **A GitHub token no longer sits in the process list.** The curl
  download path passed the bearer token in argv, where every local user
  could read it for the length of the download. It travels in a 0600
  config file instead.
- **The dsh web token no longer lands in a world-readable log.** The
  log holds a one-time token that opens the web UI, and it was created
  0644 under the usual umask.
- **A tarball is no longer extracted blindly.** On the stock LTS
  Pythons the `filter=` argument does not exist, and the fallback
  extracted without it — the path-traversal class of bug. Every member
  is checked now.
- **The dashboard renders report fields as text.** The model name, the
  workload name and the grade came from report JSON and were written
  into the row as HTML, so a model file named `<img onerror=...>` ran
  script in the dashboard's origin. The cells are built as text now,
  and the dashboard sends a content-security policy.
- **One bad report no longer blanks the dashboard.** A null where a
  number was expected threw in the first arithmetic call, and nothing
  rendered at all.
- **A model's answer is no longer read as markup.** The REPL printed
  the answer through the rich console, which ate `[i]` as an italic
  tag and raised on a stray `[/red]`, aborting the stream mid-answer.
- **The tap no longer loses a rolled file.** The daemon and the TUI
  write the same day file and the lock is per process, so both could
  roll into the same name and the second rename overwrote the first
  with the fresh base. The name is claimed before the rename, and a
  short write is now completed instead of truncating the line.
- **A split GGUF published under another name is fetched and linked.**
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder ships five parts named
  `-of-00006` and puts the sixth — the 110 GB Engram table — in the
  repository root as `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`. The pull
  fetched the five it could name and stopped, leaving a model whose
  header still said `split.count = 6` and whose sixth part did not
  exist. llama-server then refused to start:
  `failed to open GGUF file '...-00006-of-00006.gguf'`. The missing
  part is now found by reading the header of the repository's other
  GGUFs — the part says what it is — and linked under the name
  llama.cpp opens.
- **A model with a tensor too big to hold resident keeps it on disk.**
  The Engram table is 110 GB on its own; loaded eagerly it needs more
  memory than the machine has and the server dies in the OOM killer
  during model load, with nothing to read. llama.cpp reads such
  tensors on demand when `--lazy-mode` is on, and the launch now
  turns that on for any model carrying one.
- **A split is sized by all of its parts, not the first.** The fit
  planner read the weights from the file the registry points at, which
  for a split is the metadata shard: 5.8 MB standing in for 72 GB of
  weights. The launch therefore claimed the model needed no headroom
  and asked for a context window that did not fit.

## [0.3.14] - 2026-11-08

### Fixed
- **A unified-memory GPU is judged on the pool it shares.** The
  firmware reserves a slice for the iGPU (0.5 GB on a Ryzen AI Max+),
  sysfs reports that slice as VRAM, and the `vram_gb >= 4` gate threw
  the part out — so a Strix Halo box that runs 35B models on its
  Radeon 8060S was profiled as CPU-only, `ngl=0`, and told it had no
  usable GPU. The carve is now only what the reason line reports; the
  number the gate reads is the shared pool, as it already was for
  AMD parts that report no carve at all and for Apple silicon. The
  profile schema moves to 6 so a cached CPU recommendation is
  recomputed, and `toki hardware` says `shared=` for these parts
  instead of calling the pool VRAM.

## [0.3.13] - 2026-11-08

### Added
- **A slow turn is a number, not a mystery.** The daemon now measures
  how fast the machine prefills per model, projects the cost of the
  next prompt from that rate, and says so before the wait: the estimate
  rides in the first event of the Anthropic stream, the measured
  prefill is in the non-streaming response, and a prefill that will
  take more than 20 seconds is logged with its numbers.

### Fixed
- **A slow client is no longer abandoned.** Claude Code gives a local
  endpoint ten minutes and treats 300 seconds of silence as a dead
  stream; a CPU-only laptop can spend that on a prefill. The daemon
  waited 600 s for llama-server, the same number as the client's
  budget, so whichever gave up first turned a slow turn into a 500 or
  a timeout. The upstream timeout is now 1800 s
  (`TOKI_BACKEND_TIMEOUT`), and on a box the profile calls CPU-only the
  launcher gives Claude Code a 25-minute budget
  (`TOKI_CLAUDE_TIMEOUT_MS` overrides it).
- **An install no longer dies over a GPU build it did not ask for.** The
  hardware profile recommended CPU on a laptop whose llama-server was a
  working Vulkan build, and the install refused to overwrite it — which
  is right, and also aborted the whole install, so the agent-config
  wiring and the verification never ran. An auto-detected CPU backend
  now keeps the existing GPU build and lets the install finish; an
  explicit `--backend cpu` still refuses, and both messages name the
  exact command (`toki model install-llama --backend vulkan`) instead
  of a bare flag.
- **The hardware profile read RAM in the wrong unit.** `/proc/meminfo`
  reports kibibytes and the profile divided by 1024**3, so every Linux
  box reported a thousand times too little: an 8 GB machine showed
  `ram=0GB`, and that number feeds the KV and thread sizing. It also
  now never reports zero, and a profile cached with no RAM in it is
  re-profiled instead of trusted.
- **An Intel iGPU is no longer called a discrete GPU.** A laptop with an
  integrated Arc part (no dedicated VRAM) was described as
  `vram=0.0GB use=no (discrete GPU)`: the VRAM clause was skipped when
  the number was zero. Intel parts with no dedicated VRAM are now
  reported as integrated, Lunar Lake (0x7d) is in the iGPU list, and
  the reason line says why the CPU build is recommended.
- **The installers agree about the extras.** The README, `install.bat`
  and the curl/irm one-liners install the embeddings extra by default;
  `install.sh` did not, so the benchmark's semantic similarity quietly
  degraded to its proxy on Linux. It does now, and `TOKI_EXTRAS=` still
  means no extras.
- **Claude Code talks to the daemon again.** Claude Code sends the
  `ANTHROPIC_API_KEY` from the environment only when that key is on its
  approved list in `~/.claude.json`; a key it has never seen gets a
  prompt, and a key that was declined is never sent again. Claude Code
  then fell back to `ANTHROPIC_AUTH_TOKEN` — on a machine with another
  provider configured, that token was not ours — and the daemon answered
  `401 invalid token` for every request, while Claude Code showed
  "Please run /login" and retried ten times against a server that was
  working. The local route now travels as `ANTHROPIC_AUTH_TOKEN`, which
  Claude Code sends without asking, a stale rejection is cleared, the
  401 names the fix (`toki model use <name>`), and `toki doctor` reports
  a key Claude Code will not send.

## [0.3.12] - 2026-11-07

### Added
- **Windows runs the real launcher and TUI.** Bare `toki` opens the Textual
  launcher and `toki tui` the full-screen chat, instead of falling back to the
  numbered menu and REPL. All process differences live in `winproc.py`
  (`OpenProcess` instead of `os.kill`, detached process groups, `.cmd`/`.bat`
  through `cmd.exe`), and status polling runs off the event loop.
  `TOKI_CLI=1` still selects the plain fallbacks on any platform.
- **The OS/arch/backend matrix is complete.** Linux x86_64 and aarch64:
  CUDA 12.8/13.4, ROCm 10.0, SYCL, OpenVINO 2026.4.1, Vulkan, CPU. Windows
  x86_64: CUDA, SYCL, Vulkan, ROCm/HIP, OpenVINO, CPU; Windows ARM64: CUDA,
  Vulkan, CPU. macOS: one universal Metal build that runs CPU-only with
  `-ngl 0`.
- **Installs check the runtime they claim.** CUDA fetches the matching
  `cudart`/`cublas` and checks the library major; Linux SYCL verifies DPC++,
  oneMKL and Level Zero; ROCm verifies `libamdhip64.so`. A missing runtime
  refuses the install rather than leaving llama-server to fall back to CPU
  while reporting success. The device line comes from
  `llama-server --list-devices`; a machine whose driver enumerates no device
  gets a loud warning, not a silent success.
  (Corrected after release: this entry originally said the install
  *required* a real `CUDA0` device line. The code has always warned
  instead of refusing — only a missing runtime refuses.)
- **A public site and docs at [tokirecur.com](https://tokirecur.com)**, built
  and deployed by the repo, with an installer that owns only marked regions of
  `/etc/caddy/Caddyfile` so other sites on the same machine are untouched.
- **CI gates for the site**: axe against WCAG 2.2 A/AA at three widths, link
  crawling, contrast, wire weight, LCP and CLS.

### Fixed
- CPU mode is honest: a registry-pinned `-ngl 0` is a true CPU launch,
  `toki up` CPU mode no longer asks for a GPU, and an explicit CPU launch
  does not warn about an unused GPU.
- Assistant messages are normalized and tool schemas flattened before they
  reach llama-server, so clients that send either shape work.
- Slow prefills keep the stream alive, and a busy backend is no longer
  reported as wedged.
- The harness advertises the model id clients actually use.
- Loading a second model beside a running one warns instead of silently
  competing for the GPU.
- Windows: Store Python aliases are no longer mistaken for a real
  installation, the batch installer has CRLF endings and an ASCII-safe path,
  and the suite no longer asserts POSIX or macOS behaviour on Windows.
- Unit tests no longer read the developer's own model registry.

### Changed
- The repository is public, so the install path no longer asks for a token.
- The README was rewritten around install, platforms, interfaces and
  commands; the docs code blocks lost their separate black header strip.

## [0.3.6] - 2026-10-01

### Added
- **A fresh-install test that walks the whole chain** in a temporary home:
  register a model, sync every client config, build every launch plan, check
  that every plan carries the daemon token, and authenticate against a real
  daemon with the token that ends up in those configs.

  This exists because every fault found while wiring up DeepSeek Harness had
  the same shape - correct when written, silently wrong later - and none of
  them was *guarded*. A config naming a dead port, a profile that was never
  bound, a token the daemon does not accept: all three surface as an error
  inside somebody else's application, so neither the installer job nor the
  unit tests could ever see them. It runs on all three CI platforms, because
  the config paths are not the same everywhere.

  It found nothing new in the product, which is the point - it locks in the
  chain that took five releases to get right.

## [0.3.5] - 2026-10-01

### Added
- **The DeepSeek Harness TUI and web UI now open together.** Selecting dsh
  starts the browser UI in the background and opens it, then hands the
  terminal to the TUI. Both surfaces are the same model on the same daemon,
  so there was no reason to make them two separate selections.

  The companion is started *before* the foreground launch on purpose: a
  foreground launch `exec`s and replaces the process, so anything spawned
  afterwards never exists. A companion that fails to start is reported and
  the TUI launches anyway - a browser that will not open must not cost you
  the terminal.

  dsh-web still works as its own entry for when you only want the browser.

## [0.3.4] - 2026-10-01

### Fixed
- **DeepSeek Harness Web opens even when one is already running.** A second
  `dsh web` dies with `EADDRINUSE: address already in use 127.0.0.1:3080` and
  never prints a URL, so the launcher had nothing to open and reported a
  failure the user could not act on - while a perfectly good instance was
  serving on 3080. It now reuses the running instance's URL instead.
- **A launch no longer picks up the previous run's token.** The log is opened
  in append mode and the URL scan started at offset 0, so the first match was
  last run's URL: the browser opened a token for a session that no longer
  existed. Scanning now starts at the file size recorded when the process was
  spawned.

## [0.3.3] - 2026-10-01

### Added
- **DeepSeek Harness Web is in the launcher.** `dsh web` takes over the
  terminal and only prints its URL, so from tokirecur it was unreachable:
  the one web row is a hardcoded toki entry and `launch_cmd` had no web
  branch for dsh. There is now a "Launch DeepSeek Harness Web" entry that
  starts it detached, reads the tokenised URL it prints on boot, and opens
  it in the browser. A hardcoded `http://127.0.0.1:3080/` would have 401'd -
  the fence accepts the URL-plus-token it prints, nothing else.

## [0.3.2] - 2026-10-01

### Fixed
- **The DeepSeek Harness web UI works.** `dsh web` served its UI, showed the
  model selector, and then failed every turn with `API key is invalid`.
- **Every dsh profile is now bound, not just `tui`.** `write_dsh_config`
  hardcoded the `tui` profile, so `web` and `headless` kept the empty patch
  layer and fell through to `deepseek-official/deepseek-flash`, which demands
  a `DEEPSEEK_API_KEY` this machine does not have. It now walks the profiles
  that exist and writes each one. A profile directory with no
  `cordis.patch.yml` is skipped rather than created — inventing a user patch
  layer is not ours to do.
- **`auth_token()` now reads `~/.config/tokirecur/auth`.** It read the env var
  and otherwise invented a fresh random token, so it never agreed with
  `api.app._get_auth_token`, which reads env then that file. With no env var
  set, two calls returned two different tokens and neither matched the
  daemon — so a written client config could not work, and a stray
  `TOKI_AUTH_TOKEN` in the environment silently overrode the configured token
  on the daemon side. The two resolvers now have the same order, and
  `test_security.py` is hermetic so it cannot pass by reading the machine's
  real auth file again.

Verified in a browser: `http://127.0.0.1:3080/?token=…` loads the Harness UI,
lists the local `Qwen3.5-0.8B-Q4_K_M`, and a turn completes in ~2s.

## [0.3.1] - 2026-10-01

### Fixed
- **The DeepSeek Harness TUI actually starts.** Two independent faults, and
  the first one was hiding the second.

  0.3.0 fixed a stale `~/.dsh/settings.yaml` and dsh stopped reporting
  `TRANSPORT: Connection error` — which turned out to be only the first
  layer. What remained was
  `ERR_PACKAGE_PATH_NOT_EXPORTED` on
  `@deepseek-ai/dsh-tool-subagent/model-selection-settings`, thrown by
  `@tomowang/dsh-tui@0.8.1`: that bundle is built against
  `@deepseek-ai/dsh-*` `^0.1.2-rc.1` and the installed dsh was `0.1.1-rc.2`.
  Aligning to dsh `0.1.7-rc.2` with `@tomowang/dsh-tui` `0.9.0` (an exact peer
  match) boots the TUI. `dsh --profile headless` still answers on both.

  dsh 0.1.7 migrates `settings.yaml` aside to `settings.yaml.imported` and
  expects a named profile to be bound by that profile's own
  `cordis.patch.yml`. `settings.yaml` is the config for `headless` only:
  `dsh --profile tui --dump-config` and `--dump-default-config` came back
  identical, so the profile ignored it and fell through to its bundle default
  of `deepseek-official/deepseek-flash`, then asked for a
  `DEEPSEEK_API_KEY` this machine does not have. `write_dsh_config` now
  writes that profile layer too, so both layers are set by one call and
  cannot drift.

## [0.3.0] - 2026-10-01

### Fixed
- **DeepSeek Harness now starts.** `dsh` was pinned in `~/.dsh/settings.yaml`
  to a model id and a daemon port that no longer existed, so it failed with a
  bare `TRANSPORT: Connection error` and its TUI profile rendered nothing. The
  file was only ever written by `toki model use`, so switching the active
  model by any other path — or adding and removing models — left every
  config-owning client pointing at a dead endpoint.
- Client configs are now rewritten at launch time, not just on model switch.
  Launching is the last moment the answer is known and cannot go stale
  afterwards. A failed write warns and still launches rather than blocking.

### Removed
- **Meta Code, Cursor Agent and Devin are no longer offered in the launcher.**
  All three authenticate against their own account rather than a local stack,
  so none of them could use a model bound here. `protocol="account"` existed
  only for them and is gone with it; `openai` and `anthropic` remain.

## [0.2.33] - 2026-08-13

### Changed
- Hardware profiling drives backend detection and CPU tuning (`toki doctor`,
  `--profile`/`--no-profile`, 30-day profile TTL).
- Installers no longer download a model at install time; model selection is
  a separate, explicit step.
