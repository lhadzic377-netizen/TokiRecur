# llama-server KV endpoints

## Status: not needed, and no longer used

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
targets end-to-end) already has these endpoints.

**Toki Recur does not use them.** The recycling engine routes a request
to the llama-server slot that already holds the matching prefix and lets
llama-server reuse that in-memory KV state (`cache_prompt=true`). It
never asks the server to write KV to disk and never restores it.

Disk persistence was tried and removed: restoring disk KV corrupted
precision — the benchmark measured 88% substantive divergence from the
no-cache baseline. `LlamaServerBackend.cache_save` / `cache_load` and
the `.trkv` sidecar still exist in the tree, but nothing in the request
path calls them.

What is left here is historical:

- `llama-server-kv-endpoints.diff` is kept as a reference for older
  llama.cpp builds and as a starting point for an upstream PR that adds
  - a content-addressed path (cache_id) parameter
  - a JSON return shape consistent with the rest of the recycling API
  - an `/info` endpoint that returns n_past + model_id without a save
- `scripts/build_llama_cpp.sh` still references the patch, but only to
  decide whether to apply it: a checkout that already has the endpoints
  is built unpatched.

Do not start `llama-server` with `--slot-save-path` for Toki Recur. The
daemon does not read those files, and the directory would only collect
KV blobs nobody restores.
