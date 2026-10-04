"""Synthetic runner integrity checks; no scientific inputs or stages are read."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_investigation as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        for name in ("uap/src", "uap/data/raw", "uap/results"):
            (self.root / name).mkdir(parents=True)
        (self.root / "uap/src/synthetic.py").write_text("# synthetic source\n")
        (self.root / "uap/requirements.txt").write_text("# synthetic, no research dependencies\n")
        self.raw = self.root / "uap/data/raw/input.csv"
        self.raw.write_bytes(b"synthetic-input\n")
        self.entry = {"ok": True, "dest": "data/raw/input.csv", "url": "https://example.org/input.csv",
                      "sha256": runner.digest(self.raw)}
        self.manifest = self.root / "uap/data/manifest.jsonl"
        self.manifest.write_text(json.dumps(self.entry) + "\n")
        self.args = argparse.Namespace(python=sys.executable, restore=False, resume=False,
                                       preflight_only=False, feature_chunk=50_000, permutations=30)
        self.runtime = {"python": "synthetic-pinned-runtime", "pins": {}}
        self.stages = [{"name": name, "command": ["synthetic", name], "outputs": [f"results/{name}.csv"]}
                       for name in ("first", "second")]

    def execute(self, invoke):
        with patch.object(runner, "versions", return_value=self.runtime), \
                patch.object(runner, "plan", return_value=self.stages), \
                patch.object(runner, "invoke", side_effect=invoke), contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.args, self.root)

    def test_stale_restoration_cannot_make_missing_data_ready(self):
        self.raw.unlink()
        (self.root / "uap/results/input_restoration.json").write_text(json.dumps({"counts": {"verified_existing": 1}}))
        historic = self.root / "uap/results/discovery_results.csv"
        historic.write_bytes(b"historical-result\n")
        with patch.object(runner, "invoke") as invoke, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(self.args, self.root), 2)
        invoke.assert_not_called()
        self.assertEqual(historic.read_bytes(), b"historical-result\n")
        state = json.loads((self.root / "uap/results/investigation_status.json").read_text())
        self.assertEqual(state["scientific_conclusion"], "NOT_EVALUATED")
        self.assertFalse(state["research_executed"])

    def test_checksum_mismatch_blocks_unreviewed_replacement(self):
        self.raw.write_bytes(b"different-source-snapshot\n")
        result = runner.preflight(self.root)
        self.assertFalse(result["ready"])
        self.assertEqual(result["mismatched_files"], 1)
        approval = {**self.entry, "sha256": runner.digest(self.raw), "restoration": "accepted_current",
                    "historical_sha256": self.entry["sha256"]}
        with self.manifest.open("a") as stream:
            stream.write(json.dumps(approval) + "\n")
        approved = runner.preflight(self.root)
        self.assertTrue(approved["ready"])
        self.assertEqual(approved["accepted_current_snapshots"], [self.entry["dest"]])

    def test_manifest_path_escape_is_rejected(self):
        self.entry["dest"] = "data/raw/../../../escape.csv"
        self.manifest.write_text(json.dumps(self.entry) + "\n")
        with self.assertRaisesRegex(ValueError, "Unsafe manifest"):
            runner.preflight(self.root)

    def test_plan_freezes_after_discovery_before_all_holdout_checks(self):
        stages = runner.plan(self.args, {"code": "synthetic", "data": "synthetic"})
        names = [stage["name"] for stage in stages]
        for name in ("ml_discovery", "sequences", "spatial_discovery", "positive_controls"):
            self.assertLess(names.index(name), names.index("freeze"))
        for name in ("validate", "starlink_integrity", "solar_cycle_systems", "convergence", "media_waves"):
            self.assertGreater(names.index(name), names.index("freeze"))
        freeze = next(stage for stage in stages if stage["name"] == "freeze")
        self.assertIn("splits_manifest_sha256", freeze["command"][-1])

    def test_failure_stops_and_historical_outputs_are_backed_up(self):
        interim = self.root / "uap/data/interim"
        interim.mkdir()
        cache = interim / "report_context.parquet"
        cache.write_bytes(b"old-generated-cache")
        old = self.root / "uap/results/first.csv"
        old.write_text("old-result\n")
        calls = []
        def fake(command, root, log):
            calls.append(command[-1])
            return 7
        self.assertEqual(self.execute(fake), 1)
        self.assertEqual(calls, ["first"])
        backups = list((self.root / "uap/logs").glob("*first-first.csv.backup.log"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "old-result\n")
        self.assertFalse(cache.exists())
        cache_backups = list((self.root / "uap/logs").glob("*cache-report_context.parquet.backup.log"))
        self.assertEqual(cache_backups[0].read_bytes(), b"old-generated-cache")
        state = json.loads((self.root / "uap/results/investigation_status.json").read_text())
        self.assertEqual((state["state"], state["exit_code"]), ("FAILED_STAGE", 7))

    def test_resume_verifies_outputs_and_skips_only_completed_stages(self):
        calls = []
        def initial(command, root, log):
            name = command[-1]
            calls.append(name)
            if name == "second":
                return 1
            (root / f"uap/results/{name}.csv").write_text("fresh-first\n")
            return 0
        self.assertEqual(self.execute(initial), 1)
        self.args.resume = True
        def finish(command, root, log):
            name = command[-1]
            calls.append(name)
            (root / f"uap/results/{name}.csv").write_text("fresh-second\n")
            return 0
        self.assertEqual(self.execute(finish), 0)
        self.assertEqual(calls, ["first", "second", "second"])
        (self.root / "uap/results/first.csv").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "output/command changed"):
            self.execute(finish)

    def test_resume_rejects_changed_code_and_retains_journal_when_blocked(self):
        def failure(command, root, log):
            return 1
        self.assertEqual(self.execute(failure), 1)
        status_path = self.root / "uap/results/investigation_status.json"
        journal = json.loads(status_path.read_text())["journal"]
        raw_bytes = self.raw.read_bytes()
        self.raw.unlink()
        self.assertEqual(self.execute(failure), 2)
        self.assertEqual(json.loads(status_path.read_text())["journal"], journal)
        self.raw.write_bytes(raw_bytes)
        self.args.resume = True
        (self.root / "uap/src/synthetic.py").write_text("# changed source\n")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            self.execute(failure)

    def test_freeze_and_split_hash_guard(self):
        base = self.root / "uap/results"
        (base / "splits_manifest.json").write_text('{"master_seed":20261004}')
        identity = {"code": "same", "data": "same"}
        payload = json.dumps({"runner_identity": identity,
                              "splits_manifest_sha256": runner.digest(base / "splits_manifest.json")}).encode()
        (base / "frozen_hypotheses.json").write_bytes(payload)
        (base / "frozen_hypotheses.sha256").write_text(hashlib.sha256(payload).hexdigest())
        runner.check_freeze(self.root, identity)
        (base / "splits_manifest.json").write_text('{"master_seed":0}')
        with self.assertRaisesRegex(ValueError, "Split manifest changed"):
            runner.check_freeze(self.root, identity)
        (base / "frozen_hypotheses.json").write_bytes(payload + b" ")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            runner.check_freeze(self.root, identity)

    def test_supplementary_manifest_append_resumes_and_checks_new_measurements(self):
        def fresh(command, root, log):
            (root / f"uap/results/{command[-1]}.csv").write_text("fresh\n")
            return 0
        self.assertEqual(self.execute(fresh), 0)
        initial_prefix = self.manifest.read_bytes()
        extra = self.root / "uap/data/raw/magnetometer.json"
        extra.write_bytes(b"synthetic supplementary measurement")
        row = {"ok": True, "dest": "data/raw/magnetometer.json", "url": "https://example.org/measurement",
               "sha256": runner.digest(extra)}
        with self.manifest.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.args.resume = True
        with patch.object(runner, "invoke") as invoke:
            self.assertEqual(self.execute(invoke), 0)
            invoke.assert_not_called()
        extra.write_bytes(b"mutated measurement")
        with self.assertRaisesRegex(ValueError, "Supplementary raw input missing or changed"):
            self.execute(fresh)
        extra.write_bytes(b"synthetic supplementary measurement")
        content = self.manifest.read_bytes()
        self.manifest.write_bytes(initial_prefix.replace(b"example.org", b"example.net") + content[len(initial_prefix):])
        with self.assertRaisesRegex(ValueError, "Initial manifest prefix changed"):
            self.execute(fresh)


if __name__ == "__main__":
    unittest.main()
