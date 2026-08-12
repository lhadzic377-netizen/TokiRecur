# Integration tests

These tests spin up a real `llama-server` against a real GGUF. They are
skipped gracefully if no model is found, but to actually run them you need:

1. **A GGUF on disk.** `tests/integration/conftest.py` looks for:
   - `$TOKI_TEST_MODEL` (env var, exact path), or
   - `~/Downloads/Qwen3.5-9B-Q4_K_M.gguf`, or
   - `~/.lmstudio/models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf` (fallback for fast CI).
2. **A throwaway tokirecur daemon on port 8799.** Don't touch the long-running
   one on 8094:

   ```bash
   nohup .venv/bin/toki serve --host 127.0.0.1 --port 8799 \
       --backend-url http://127.0.0.1:8094 \
       >/tmp/opencode/toki-serve-smoke.log 2>&1 &
   ```

3. **Run the suite:**

   ```bash
   .venv/bin/pytest -q tests/integration
   ```

4. **Tear down:**

   ```bash
   pkill -f "toki serve --port 8799"
   ```

Tests skip with `pytest.skip(...)` (not a failure) when the GGUF or daemon
isn't available, so a clean checkout's `pytest` run without a model
silently passes — that's expected.

If a test fails:

- Check the daemon log: `tail -100 /tmp/opencode/toki-serve-smoke.log`.
- Check the llama-server log: it's under `~/.local/state/tokirecur/model/<name>/llama.log`.
- Re-run just that test: `.venv/bin/pytest -q tests/integration/test_foo.py -k "test_bar"`.
