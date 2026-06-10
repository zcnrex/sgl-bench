"""Schema validation tests ([[RFC-0001:C-SCOPE]], [[RFC-0001:C-BRANCH]])."""

import unittest

from pydantic import ValidationError

from sglbench.argsearch.schema import Branch, SearchConfig


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
        {"model": "m", "branches": [_branch(**(branch_over or {}))]}
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
        with self.assertRaisesRegex(ValidationError, "weight-precision"):
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

    def test_baseline_extra_key_rejected(self):
        with self.assertRaisesRegex(ValidationError, "non-candidate args"):
            _config({"baseline": {"a": 1, "chunked-prefill-size": 16384}})

    def test_constraint_undeclared_arg_rejected(self):
        with self.assertRaisesRegex(ValidationError, "undeclared"):
            _config({"constraints": [{"name": "dead", "when": {"dp-size": [1]},
                                      "forbid": {"enable-dp-attention": [True]}}]})

    def test_constraint_must_have_effect(self):
        with self.assertRaisesRegex(ValidationError, "must set 'forbid' or 'require'"):
            _config({"constraints": [{"name": "empty", "when": {"a": [1]}}]})

    def test_duplicate_branch_names_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate branch"):
            SearchConfig.model_validate(
                {"model": "m", "branches": [_branch(), _branch()]}
            )

    def test_multi_branch_ok(self):
        cfg = SearchConfig.model_validate(
            {"model": "m", "branches": [_branch(name="b200-fp8kv"), _branch(name="b200-bf16kv")]}
        )
        self.assertEqual([b.name for b in cfg.branches], ["b200-fp8kv", "b200-bf16kv"])

    def test_precision_branches_alias_accepted(self):
        cfg = SearchConfig.model_validate(
            {"model": "m", "precision_branches": [_branch()]}
        )
        self.assertEqual([b.name for b in cfg.branches], ["b"])

    def test_branch_keys_recorded(self):
        b = Branch.model_validate(_branch(hardware="8xH100", kv_cache_precision="bf16"))
        self.assertEqual(b.branch_keys(), {"hardware": "8xH100", "kv_cache_precision": "bf16"})

    def test_kv_cache_precision_defaults_to_fixed_dtype(self):
        b = Branch.model_validate(_branch(fixed={"quantization": "modelopt_fp4", "kv-cache-dtype": "fp8_e4m3"}))
        self.assertEqual(b.kv_cache_precision_value, "fp8_e4m3")
        self.assertEqual(b.branch_keys()["kv_cache_precision"], "fp8_e4m3")

    def test_focused_grid_valid(self):
        cfg = _config({
            "candidate": [{"name": "a", "values": [1, 2]}, {"name": "b", "values": [3, 4]}],
            "baseline": {"a": 1, "b": 3},
            "focused_grid": {"args": ["a"], "rationale": "a interacts", "pins": {"b": 4}},
        })
        fg = cfg.branch("b").focused_grid
        self.assertEqual(fg.args, ["a"])
        self.assertEqual(fg.pins, {"b": 4})

    def test_focused_grid_arg_must_be_candidate(self):
        with self.assertRaisesRegex(ValidationError, "focused_grid.args are not candidates"):
            _config({"focused_grid": {"args": ["nope"], "rationale": "x"}})

    def test_focused_grid_requires_rationale(self):
        with self.assertRaises(ValidationError):
            _config({"focused_grid": {"args": ["a"], "rationale": ""}})

    def test_focused_grid_pin_must_be_candidate(self):
        with self.assertRaisesRegex(ValidationError, "pins arg 'nope' is not a candidate"):
            _config({"focused_grid": {"args": ["a"], "rationale": "x", "pins": {"nope": 1}}})

    def test_focused_grid_pin_not_gridded(self):
        with self.assertRaisesRegex(ValidationError, "is also gridded"):
            _config({"focused_grid": {"args": ["a"], "rationale": "x", "pins": {"a": 2}}})

    def test_focused_grid_pin_value_must_be_allowed(self):
        with self.assertRaisesRegex(ValidationError, "not one of its candidate values"):
            _config({
                "candidate": [{"name": "a", "values": [1, 2]}, {"name": "b", "values": [3, 4]}],
                "baseline": {"a": 1, "b": 3},
                "focused_grid": {"args": ["a"], "rationale": "x", "pins": {"b": 99}},
            })

    def test_focused_grid_requires_all_nongridded_pinned(self):
        with self.assertRaisesRegex(ValidationError, "non-gridded candidates must be pinned"):
            _config({
                "candidate": [{"name": "a", "values": [1, 2]}, {"name": "b", "values": [3, 4]}],
                "baseline": {"a": 1, "b": 3},
                "focused_grid": {"args": ["a"], "rationale": "x"},
            })

    def test_quality_gate_loads(self):
        cfg = SearchConfig.model_validate({
            "model": "m",
            "quality_gate": {"dataset": "gpqa", "tolerance": 0.02},
            "branches": [_branch()],
        })
        self.assertEqual(cfg.quality_gate.dataset, "gpqa")
        self.assertEqual(cfg.quality_gate.metric, "accuracy")
        self.assertEqual(cfg.quality_gate.direction, "higher")
        self.assertEqual(cfg.quality_gate.tolerance, 0.02)

    def test_candidate_accuracy_invariant_default_false(self):
        cfg = _config({
            "candidate": [
                {"name": "a", "values": [1, 2]},
                {"name": "b", "values": [3, 4], "accuracy_invariant": True},
            ],
            "baseline": {"a": 1, "b": 3},
        })
        cands = {c.name: c.accuracy_invariant for c in cfg.branch("b").candidate}
        self.assertEqual(cands, {"a": False, "b": True})

    def test_quality_gate_rejects_bad_direction(self):
        with self.assertRaises(ValidationError):
            SearchConfig.model_validate({
                "model": "m",
                "quality_gate": {"dataset": "d", "tolerance": 0.1, "direction": "sideways"},
                "branches": [_branch()],
            })

    def test_quality_gate_rejects_negative_tolerance(self):
        with self.assertRaises(ValidationError):
            SearchConfig.model_validate({
                "model": "m",
                "quality_gate": {"dataset": "d", "tolerance": -0.1},
                "branches": [_branch()],
            })


if __name__ == "__main__":
    unittest.main()
