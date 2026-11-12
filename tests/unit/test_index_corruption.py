"""A corrupt cache DB must not crash every command that opens it.

The index is regenerable scratch: on a non-DB file or a truncated
embedding blob (disk-full / interrupted write) BruteForceIndex quarantines
the bad file and rebuilds, instead of letting sqlite3.DatabaseError /
ValueError escape as a traceback in `toki stats`, `toki evict`, or the
daemon startup path.
"""

from __future__ import annotations

import sqlite3

from tokirecur.index import BruteForceIndex


def test_non_db_file_quarantined_and_rebuilt(tmp_path):
    db = tmp_path / "cache.sqlite"
    db.write_bytes(b"this is definitely not a sqlite database")
    idx = BruteForceIndex(db)  # must not raise
    try:
        assert idx.stats()["entries"] == 0
    finally:
        idx.close()
    assert any(
        p.name.startswith("cache.sqlite.corrupt-") for p in tmp_path.iterdir()
    )


def test_truncated_embedding_blob_quarantined(tmp_path):
    db = tmp_path / "cache.sqlite"
    idx = BruteForceIndex(db)
    idx.close()
    # Write a row whose embedding blob is not a multiple of 4 bytes, so
    # np.frombuffer(dtype=float32) raises ValueError on the next open.
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO cache_entries
            (cache_id, prompt, token_ids, embedding, n_past, model_id,
             bytes_on_disk, created_at, last_used_at, hit_count, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("deadbeef", "p", b"", b"\x00\x00\x00", 1, "m", 0, 0.0, 0.0, 0, 0),
    )
    conn.commit()
    conn.close()

    idx2 = BruteForceIndex(db)  # must quarantine + rebuild, not raise
    try:
        assert idx2.stats()["entries"] == 0
    finally:
        idx2.close()
    assert any(
        p.name.startswith("cache.sqlite.corrupt-") for p in tmp_path.iterdir()
    )


def test_healthy_db_unchanged(tmp_path):
    db = tmp_path / "cache.sqlite"
    idx = BruteForceIndex(db)
    idx.close()
    idx2 = BruteForceIndex(db)
    try:
        assert idx2.stats()["entries"] == 0
    finally:
        idx2.close()
    # No quarantine file for a healthy DB.
    assert not any(".corrupt-" in p.name for p in tmp_path.iterdir())
