# llama-server KV endpoints

## Status: not needed

> License: this patch modifies llama.cpp, MIT (c) 2023-2026 The ggml
> authors. Full text: [`LICENSE.llama-cpp`](../LICENSE.llama-cpp).

As of llama.cpp b3524 (mid-2025), the upstream `llama-server` already
exposes KV cache save/restore endpoints, gated by the `--slot-save-path`
flag:

```
POST /slots/{id_slot}?action=save     body: {"filename": "..."}
POST /slots/{id_slot}?action=restore  body: {"filename": "..."}
POST /slots/{id_slot}?action=erase    body: {"filename": "..."}
```

The Debian-packaged `llama-server` 9413 (which is what this project
targets end-to-end) already has these endpoints. Toki Recur uses
them directly via `LlamaServerBackend`.

The patch file in this directory is preserved as historical reference
for older llama.cpp versions and as a starting point for an upstream
PR that adds:
  - a content-addressed path (cache_id) parameter
  - a JSON return shape consistent with the rest of the recycling API
  - an `/info` endpoint that returns n_past + model_id without a save

To use the existing upstream endpoints, start `llama-server` with:

```
llama-server -m model.gguf --slot-save-path /var/lib/tokirecur/kv ...
```

The cache directory will hold one `.bin` file per cache_id.
