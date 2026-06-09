"""Schema validation tests ([[RFC-0001:C-SCOPE]], [[RFC-0001:C-PRECISION-BRANCH]])."""

import unittest

from pydantic import ValidationError

from sglbench.argsearch.schema import PrecisionBranch, SearchConfig


def _branch(**over):
    base = dict(
        name="b",
        fixed={"quantization": "modelopt_fp4"},
        candidate=[{"name": "a", "values": [1, 2]}],
        constraints=[],
        baseline={"a": 1},
    )
    base.update(over)
    return base


def _config(branch_over=None):
    return SearchConfig.model_validate(
        {"model": "m", "precision_branches": [_branch(**(branch_over or {}))]}
    )


class SchemaTest(unittest.TestCase):
    def test_valid_branch(self):
        cfg = _config()
        self.assertEqual(cfg.branch("b").candidate_names, ["a"])

    def test_spec_decode_rejected(self):
        with self.assertRaisesRegex(ValidationError, "speculative"):
            _config({"candidate": [{"name": "speculative-algorithm", "values": ["EAGLE", "NONE"]}],
                     "baseline": {"speculative-algorithm": "NONE"}})

    def test_candidate_cap(self):
        cands = [{"name": f"a{i}", "values": [0, 1]} for i in range(11)]
        baseline = {f"a{i}": 0 for i in range(11)}
        with self.assertRaisesRegex(ValidationError, "exceeds cap"):
            _config({"candidate": cands, "baseline": baseline})

    def test_fixed_candidate_overlap_rejected(self):
        with self.assertRaisesRegex(ValidationError, "both fixed and candidate"):
            _config({"fixed": {"quantization": "modelopt_fp4", "a": 9}})

    def test_quantization_as_candidate_rejected(self):
        with self.assertRaisesRegex(ValidationError, "precision-defining"):
            _config({"fixed": {}, "candidate": [{"name": "quantization", "values": ["modelopt_fp4", "fp8"]}],
                     "baseline": {"quantization": "modelopt_fp4"}})

    def test_baseline_must_cover_candidate(self):
        with self.assertRaisesRegex(ValidationError, "baseline missing"):
            _config({"baseline": {}})

    def test_baseline_value_must_be_allowed(self):
        with self.assertRaisesRegex(ValidationError, "not one of its candidate values"):
            _config({"baseline": {"a": 99}})

    def test_baseline_must_satisfy_constraints(self):
        with self.assertRaisesRegex(ValidationError, "baseline violates constraint"):
            _config({
                "candidate": [{"name": "a", "values": [1, 2]}],
                "baseline": {"a": 1},
                "constraints": [{"name": "no-a1", "when": {}, "forbid": {"a": [1]}}],
            })

    def test_constraint_must_have_effect(self):
        with self.assertRaisesRegex(ValidationError, "must set 'forbid' or 'require'"):
            _config({"constraints": [{"name": "empty", "when": {"a": [1]}}]})

    def test_duplicate_branch_names_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate precision branch"):
            SearchConfig.model_validate(
                {"model": "m", "precision_branches": [_branch(), _branch()]}
            )

    def test_multi_branch_ok(self):
        cfg = SearchConfig.model_validate(
            {"model": "m", "precision_branches": [_branch(name="nvfp4"), _branch(name="fp8")]}
        )
        self.assertEqual([b.name for b in cfg.precision_branches], ["nvfp4", "fp8"])


if __name__ == "__main__":
    unittest.main()
