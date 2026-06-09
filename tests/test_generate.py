"""Generator tests ([[RFC-0001:C-CONFIG-SOURCE]], [[RFC-0001:C-SEARCH-STRATEGY]])."""

import json
import tempfile
import unittest
from pathlib import Path

from sglbench.argsearch.generate import (
    config_hash,
    focused_grid_manifest,
    generate_grid,
    generate_ofat,
    is_valid,
    load_config,
    write_dir,
    write_grid_manifest,
)
from sglbench.argsearch.schema import SearchConfig

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nemotron_v3_ultra.yaml"


def _pin_branch():
    """Branch whose focused_grid pins a non-gridded candidate away from its baseline."""
    return SearchConfig.model_validate({
        "model": "m",
        "precision_branches": [{
            "name": "b",
            "fixed": {"quantization": "modelopt_fp4"},
            "candidate": [
                {"name": "g", "values": [1, 2]},
                {"name": "p", "values": ["lo", "hi"]},
            ],
            "baseline": {"g": 1, "p": "lo"},
            "focused_grid": {
                "args": ["g"],
                "rationale": "g interacts; p pinned to its OFAT-best",
                "pins": {"p": "hi"},
            },
        }],
    }).branch("b")


class GenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        cls.branch = cls.cfg.branch("nvfp4")

    def test_seeded_config_loads(self):
        self.assertEqual(len(self.branch.candidate), 9)
        self.assertEqual(self.branch.fixed["quantization"], "modelopt_fp4")

    def test_ofat_count_and_baseline(self):
        pts = generate_ofat(self.branch)
        labels = [p.label for p in pts]
        self.assertEqual(len(pts), 7)
        self.assertEqual(labels[0], "baseline")

    def test_ofat_prunes_coupled_singletons(self):
        labels = [p.label for p in generate_ofat(self.branch)]
        self.assertNotIn("ep-size=4", labels)
        self.assertNotIn("moe-a2a-backend=deepep", labels)
        self.assertNotIn("dp-size=4", labels)
        self.assertNotIn("enable-dp-attention=True", labels)

    def test_focused_grid_ep_pair(self):
        pts = generate_grid(self.branch, ["ep-size", "moe-a2a-backend"])
        self.assertEqual(len(pts), 2)
        seen = {(p.args["ep-size"], p.args["moe-a2a-backend"]) for p in pts}
        self.assertEqual(seen, {(1, "none"), (4, "deepep")})

    def test_focused_grid_dp_pair(self):
        pts = generate_grid(self.branch, ["dp-size", "enable-dp-attention"])
        self.assertEqual(len(pts), 2)
        seen = {(p.args["dp-size"], p.args["enable-dp-attention"]) for p in pts}
        self.assertEqual(seen, {(1, False), (4, True)})

    def test_all_emitted_configs_valid(self):
        cons = self.branch.constraints
        for p in generate_ofat(self.branch):
            self.assertTrue(is_valid(p.args, cons), p.label)
        for p in generate_grid(self.branch, ["ep-size", "moe-a2a-backend"]):
            self.assertTrue(is_valid(p.args, cons), p.label)

    def test_grid_rejects_non_candidate(self):
        with self.assertRaises(KeyError):
            generate_grid(self.branch, ["not-an-arg"])

    def test_grid_uses_declared_focused_grid_when_args_omitted(self):
        pts = generate_grid(self.branch)
        seen = {
            (p.args["ep-size"], p.args["moe-a2a-backend"],
             p.args["dp-size"], p.args["enable-dp-attention"])
            for p in pts
        }
        self.assertIn((1, "none", 1, False), seen)
        self.assertIn((4, "deepep", 4, True), seen)

    def test_grid_pins_non_gridded_to_ofat_best(self):
        branch = _pin_branch()
        pts = generate_grid(branch)
        self.assertEqual({p.args["g"] for p in pts}, {1, 2})
        self.assertTrue(all(p.args["p"] == "hi" for p in pts))
        self.assertTrue(all(p.label.startswith("g=") for p in pts))

    def test_manifest_records_admitted_set_and_rationale(self):
        branch = _pin_branch()
        pts = generate_grid(branch)
        m = focused_grid_manifest(branch, ["g"], pts)
        self.assertEqual(m["admitted_args"], ["g"])
        self.assertEqual(m["pins"], {"p": "hi"})
        self.assertTrue(m["rationale"])
        self.assertEqual(m["config_hashes"], [p.config_hash for p in pts])

    def test_write_grid_manifest_creates_file(self):
        branch = _pin_branch()
        pts = generate_grid(branch)
        m = focused_grid_manifest(branch, ["g"], pts)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_grid_manifest(m, tmp)
            self.assertEqual(path.name, "focused_grid.json")
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["admitted_args"], ["g"])
            self.assertEqual(loaded["rationale"], m["rationale"])

    def test_config_hash_stable_and_distinct(self):
        pts = generate_ofat(self.branch)
        hashes = [p.config_hash for p in pts]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(pts[0].config_hash, config_hash(pts[0].args))

    def test_write_dir_creates_dir_and_files(self):
        pts = generate_ofat(self.branch)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "out"
            self.assertFalse(target.exists())
            write_dir(pts, target)
            self.assertTrue(target.is_dir())
            files = sorted(p.name for p in target.glob("*.json"))
            self.assertEqual(files, sorted(f"{p.config_hash}.json" for p in pts))
            index = [json.loads(line) for line in (target / "index.jsonl").read_text().splitlines()]
            self.assertEqual(len(index), len(pts))
            one = json.loads((target / f"{pts[0].config_hash}.json").read_text())
            self.assertEqual(one["config_hash"], pts[0].config_hash)

    def test_to_cli_includes_fixed_skips_none(self):
        base = generate_ofat(self.branch)[0]
        cli = base.to_cli()
        self.assertIn("--quantization modelopt_fp4", cli)
        self.assertIn("--trust-remote-code", cli)        # bool flag
        self.assertNotIn("moe-a2a-backend", cli)         # baseline value "none" -> omitted


if __name__ == "__main__":
    unittest.main()
