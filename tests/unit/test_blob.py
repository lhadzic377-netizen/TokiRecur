"""Tests for the .trkv sidecar format."""

import hashlib
from pathlib import Path

import pytest

from tokirecur.cache.blob import (
    MAGIC,
    VERSION,
    BlobReader,
    BlobWriter,
    blob_path,
    compute_sha256,
)


class TestRoundTrip:
    def test_write_then_read(self, tmp_path):
        path = tmp_path / "abc.trkv"
        payload = b"some test payload bytes" * 100
        with BlobWriter(path) as w:
            w.write_header(model_id="qwen3.5-9b", n_past=2048, vocab_size=152064)
            w.write_payload(payload)
            n = w.finalize()
        assert n == len(payload)
        assert path.exists()
        reader = BlobReader(path)
        assert reader.meta["model_id"] == "qwen3.5-9b"
        assert reader.meta["n_past"] == 2048
        assert reader.meta["vocab_size"] == 152064
        assert reader.read_payload() == payload
        assert reader.digest == hashlib.sha256(payload).digest()

    def test_digest_mismatch_detected(self, tmp_path):
        path = tmp_path / "abc.trkv"
        with BlobWriter(path) as w:
            w.write_header(model_id="m", n_past=10, vocab_size=100)
            w.write_payload(b"original bytes")
            w.finalize()
        # Tamper with the payload portion (between meta and trailing
        # digest).
        data = bytearray(path.read_bytes())
        reader = BlobReader(path)
        data[reader.payload_offset : reader.payload_offset + 4] = b"\xff" * 4
        path.write_bytes(bytes(data))
        # Re-parse — the reader returns a different digest than the
        # recomputed one, which is what ``LlamaServerBackend.
        # _verify_trkv_sidecar`` uses to refuse the restore.
        reader2 = BlobReader(path)
        recomputed = compute_sha256(reader2.read_payload())
        assert recomputed != reader2.digest


class TestMagicAndVersion:
    def test_constants(self):
        assert MAGIC == b"TRKV"
        assert VERSION == 1

    def test_bad_magic_rejected(self, tmp_path):
        path = tmp_path / "bad.trkv"
        # Construct a file with valid structure but wrong magic.
        import struct
        path.write_bytes(b"XXXX" + struct.pack(">I", 1) + struct.pack(">I", 2)
                         + b'{"model_id":"m","n_past":1,"vocab_size":1}'
                         + b"\x00" * 32)
        with pytest.raises(ValueError, match="bad magic"):
            BlobReader(path)

    def test_short_file_rejected(self, tmp_path):
        path = tmp_path / "tiny.trkv"
        path.write_bytes(b"TRKV")
        with pytest.raises(ValueError, match="file too short"):
            BlobReader(path)


class TestBlobPath:
    def test_extension_is_trkv(self):
        p = blob_path(Path("/tmp/cache"), "abc123")
        assert str(p).endswith(".trkv")
        assert p.name == "abc123.trkv"
