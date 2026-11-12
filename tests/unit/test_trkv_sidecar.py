"""The ``.trkv`` sidecar must stay small: it records a digest of the
llama-server ``.bin``, never a copy of it.

Regression: ``_write_trkv_sidecar`` used to ``write_payload(bin.read_bytes())``,
embedding the entire multi-GB KV blob into the sidecar. That doubled disk
usage and spiked RSS by the full payload size on every save. The sidecar
now streams the ``.bin`` through SHA-256 and stores only the digest.
"""

from __future__ import annotations

import hashlib

import pytest

from tokirecur.backend import LlamaServerBackend
from tokirecur.cache.blob import BlobReader, BlobWriter, blob_path

CACHE_ID = "abc123def456"


def _backend(tmp_path):
    return LlamaServerBackend("http://127.0.0.1:1", cache_dir=tmp_path)


def _write_bin(tmp_path, data: bytes):
    (tmp_path / f"{CACHE_ID}.bin").write_bytes(data)


class TestSidecarIsSmall:
    def test_sidecar_does_not_embed_bin(self, tmp_path):
        backend = _backend(tmp_path)
        # 5 MiB spans several 1 MiB streaming chunks.
        payload = bytes(range(256)) * (5 * 1024 * 1024 // 256)
        _write_bin(tmp_path, payload)
        backend._write_trkv_sidecar(
            CACHE_ID, n_past=42, info={"model_id": "m", "vocab_size": 100},
        )
        trkv = blob_path(tmp_path, CACHE_ID)
        # Header + small JSON meta + 32-byte digest only — no payload.
        assert trkv.stat().st_size < 1024
        reader = BlobReader(trkv)
        assert reader.read_payload() == b""
        assert reader.digest == hashlib.sha256(payload).digest()
        assert reader.meta["n_past"] == 42

    def test_verify_accepts_matching_bin(self, tmp_path):
        backend = _backend(tmp_path)
        _write_bin(tmp_path, b"some kv bytes" * 1000)
        backend._write_trkv_sidecar(
            CACHE_ID, n_past=1, info={"model_id": "m"},
        )
        # Must not raise.
        backend._verify_trkv_sidecar(CACHE_ID)

    def test_verify_rejects_tampered_bin(self, tmp_path):
        backend = _backend(tmp_path)
        _write_bin(tmp_path, b"original kv bytes" * 1000)
        backend._write_trkv_sidecar(
            CACHE_ID, n_past=1, info={"model_id": "m"},
        )
        # Corrupt the .bin after the sidecar was written.
        bin_path = tmp_path / f"{CACHE_ID}.bin"
        data = bytearray(bin_path.read_bytes())
        data[0] ^= 0xFF
        bin_path.write_bytes(bytes(data))
        with pytest.raises(RuntimeError, match="digest mismatch"):
            backend._verify_trkv_sidecar(CACHE_ID)


class TestBackwardCompat:
    def test_legacy_embedded_sidecar_still_verifies(self, tmp_path):
        """Old daemons embedded the .bin bytes and hashed them. Because
        the embedded payload equalled the .bin, a streaming digest of the
        .bin still matches the legacy stored digest — no migration needed.
        """
        backend = _backend(tmp_path)
        payload = b"legacy kv payload" * 500
        _write_bin(tmp_path, payload)
        trkv = blob_path(tmp_path, CACHE_ID)
        with BlobWriter(trkv) as w:
            w.write_header(model_id="m", n_past=7, vocab_size=100)
            w.write_payload(payload)  # legacy: embed the bytes
            w.finalize()
        # New streaming verify must accept the legacy sidecar.
        backend._verify_trkv_sidecar(CACHE_ID)
