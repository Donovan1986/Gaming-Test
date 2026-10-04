"""Synthetic snapshot integrity checks; no network or research inputs are read."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import restore_inputs as restore


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RestorationTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name) / "synthetic-uap"
        self.raw = self.root / "data/raw"
        self.raw.mkdir(parents=True)
        self.manifest = self.root / "data/manifest.jsonl"
        self.payload = b"synthetic-verified-snapshot\n"
        self.entry = {"dest": "data/raw/input.csv", "url": "https://example.org/input.csv",
                      "ok": True, "sha256": digest(self.payload)}
        self.write_manifest([self.entry])
        for name, value in (("ROOT", self.root), ("RAW", self.raw),
                            ("MANIFEST", self.manifest),
                            ("QUARANTINE", self.raw / ".restoration_divergent")):
            mock = patch.object(restore, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        network = patch.object(restore, "session", side_effect=AssertionError("Network is forbidden in this test"))
        self.network = network.start()
        self.addCleanup(network.stop)

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_atomic_publication_creates_verified_target(self):
        part, dest = self.raw / "input.part", self.raw / "input.csv"
        part.write_bytes(self.payload)
        restore.publish_preserving_existing(part, dest, digest(self.payload))
        self.assertEqual(dest.read_bytes(), self.payload)
        # The link publishes completed bytes in one filesystem operation.
        self.assertEqual(part.stat().st_ino, dest.stat().st_ino)
        self.assertTrue(part.exists())

    def test_publication_preserves_conflicting_destination_and_partial(self):
        part, dest = self.raw / "input.part", self.raw / "input.csv"
        part.write_bytes(self.payload)
        dest.write_bytes(b"preserve-another-writers-input\n")
        before = dest.stat().st_ino
        with self.assertRaisesRegex(ValueError, "Concurrent destination differs"):
            restore.publish_preserving_existing(part, dest, digest(self.payload))
        self.assertEqual(dest.read_bytes(), b"preserve-another-writers-input\n")
        self.assertEqual(dest.stat().st_ino, before)
        self.assertEqual(part.read_bytes(), self.payload)

    def test_identical_existing_destination_is_safe_to_resume(self):
        part, dest = self.raw / "input.part", self.raw / "input.csv"
        part.write_bytes(self.payload)
        dest.write_bytes(self.payload)
        before = dest.stat().st_ino
        restore.publish_preserving_existing(part, dest, digest(self.payload))
        self.assertEqual(dest.stat().st_ino, before)
        self.assertEqual(dest.read_bytes(), self.payload)

    def test_historical_selection_preserves_snapshot_and_ignores_restoration(self):
        initial = {**self.entry, "sha256": digest(b"older-historical-snapshot")}
        accepted = {**self.entry, "sha256": digest(b"reviewed-current-snapshot"),
                    "restoration": "accepted_current", "historical_sha256": self.entry["sha256"]}
        quarantined = {**self.entry, "dest": "data/raw/.restoration_divergent/input.csv.sha256-other"}
        failed = {**self.entry, "ok": False, "sha256": digest(b"failed-download")}
        self.write_manifest([initial, self.entry, accepted, quarantined, failed])
        before = self.manifest.read_bytes()
        rows, expected = restore.historical_entries()
        self.assertEqual(len(rows), 5)
        self.assertEqual(expected, {self.entry["dest"]: self.entry})
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_historical_raw_path_escape_is_rejected(self):
        self.write_manifest([{**self.entry, "dest": "data/raw/../../escape.csv"}])
        with self.assertRaisesRegex(ValueError, "Unsafe raw destination"):
            restore.historical_entries()
        self.assertFalse((self.root / "escape.csv").exists())

    def test_existing_checksum_mismatch_is_preserved_without_network(self):
        dest = self.root / self.entry["dest"]
        dest.write_bytes(b"existing-unreviewed-content\n")
        before = self.manifest.read_bytes()
        result = restore.retrieve(self.entry, retries=3)
        self.assertEqual(result["status"], "existing_mismatch")
        self.assertEqual(result["expected_sha256"], self.entry["sha256"])
        self.assertEqual(result["sha256"], digest(b"existing-unreviewed-content\n"))
        self.assertEqual(dest.read_bytes(), b"existing-unreviewed-content\n")
        self.assertEqual(self.manifest.read_bytes(), before)
        self.network.assert_not_called()

    def test_exact_cached_snapshot_skips_download_and_provenance_append(self):
        dest = self.root / self.entry["dest"]
        dest.write_bytes(self.payload)
        inode, before = dest.stat().st_ino, self.manifest.read_bytes()
        result = restore.retrieve(self.entry, retries=3)
        self.assertEqual(result["status"], "verified_existing")
        self.assertEqual(result["sha256"], self.entry["sha256"])
        self.assertEqual(dest.stat().st_ino, inode)
        self.assertEqual(dest.read_bytes(), self.payload)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
