"""Generator tests ([[RFC-0001:C-CONFIG-SOURCE]], [[RFC-0001:C-SEARCH-STRATEGY]])."""

import unittest
from pathlib import Path

from sglbench.argsearch.generate import (
    config_hash,
    generate_grid,
    generate_ofat,
    is_valid,
    load_config,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nemotron_v3_ultra.yaml"


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

    def test_config_hash_stable_and_distinct(self):
        pts = generate_ofat(self.branch)
        hashes = [p.config_hash for p in pts]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(pts[0].config_hash, config_hash(pts[0].args))

    def test_to_cli_includes_fixed_skips_none(self):
        base = generate_ofat(self.branch)[0]
        cli = base.to_cli()
        self.assertIn("--quantization modelopt_fp4", cli)
        self.assertIn("--trust-remote-code", cli)        # bool flag
        self.assertNotIn("moe-a2a-backend", cli)         # baseline value "none" -> omitted


if __name__ == "__main__":
    unittest.main()
