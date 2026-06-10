"""Config schema for the SGLang server-arg search.

Models the versioned source of truth for restart-required server configurations
([[RFC-0001:C-CONFIG-SOURCE]]). Arguments are bucketed into fixed / candidate /
constrained ([[RFC-0001:C-SCOPE]]), and precision is a top-level branch rather than
a search axis ([[RFC-0001:C-PRECISION-BRANCH]]).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Scalar = Any  # str | int | float | bool | None

# Speculative-decode arguments are out of scope and rejected in any bucket
# ([[RFC-0001:C-SCOPE]]).
SPEC_DECODE_ARGS = {
    "speculative-algorithm",
    "speculative-num-steps",
    "speculative-eagle-topk",
    "speculative-num-draft-tokens",
    "speculative-draft-model-path",
    "speculative-draft-model",
}

# Arguments that define the precision branch; they belong in `fixed`, never `candidate`
# ([[RFC-0001:C-PRECISION-BRANCH]]).
PRECISION_DEFINING_ARGS = {"quantization"}

MAX_CANDIDATES = 10  # [[RFC-0001:C-SCOPE]]


def _key(v: Scalar):
    """Stable identity key for value comparison (keeps bool distinct from int)."""
    return (type(v).__name__, v)


class CandidateArg(BaseModel):
    name: str = Field(description="Server-arg name to vary (without the leading --). CONSUMED.")
    values: list[Scalar] = Field(min_length=1, description="Distinct values to sweep; must include the baseline value. CONSUMED.")
    accuracy_invariant: bool = Field(default=False, description="True declares the arg changes performance but not outputs, skipping its per-config accuracy gate. CONSUMED.")

    @model_validator(mode="after")
    def _unique_values(self) -> "CandidateArg":
        keys = [_key(v) for v in self.values]
        if len(keys) != len(set(keys)):
            raise ValueError(f"candidate '{self.name}' has duplicate values")
        return self


class Constraint(BaseModel):
    """An illegal-combination rule ([[RFC-0001:C-SCOPE]]).

    A config violates the constraint when every `when` clause matches AND either a
    `forbid` value is present or a `require` value is absent.
    """

    name: str = Field(description="Rule identifier for error messages. CONSUMED.")
    description: str = Field(default="", description="Human-readable note on the rule. INFORMATIONAL.")
    when: dict[str, list[Scalar]] = Field(default_factory=dict, description="Arg->allowed-values guard; the rule applies only when every clause matches. CONSUMED.")
    forbid: dict[str, list[Scalar]] = Field(default_factory=dict, description="Arg->values that are illegal when `when` matches. CONSUMED.")
    require: dict[str, list[Scalar]] = Field(default_factory=dict, description="Arg->values that must be present when `when` matches. CONSUMED.")

    @model_validator(mode="after")
    def _has_effect(self) -> "Constraint":
        if not self.forbid and not self.require:
            raise ValueError(f"constraint '{self.name}' must set 'forbid' or 'require'")
        return self

    def violated_by(self, config: dict[str, Scalar]) -> bool:
        for arg, vals in self.when.items():
            if _key(config.get(arg)) not in {_key(v) for v in vals}:
                return False
        for arg, vals in self.forbid.items():
            if _key(config.get(arg)) in {_key(v) for v in vals}:
                return True
        for arg, vals in self.require.items():
            if _key(config.get(arg)) not in {_key(v) for v in vals}:
                return True
        return False


class FocusedGrid(BaseModel):
    """Admitted focused-grid args, interaction rationale, and OFAT-best pins for the
    non-gridded candidates ([[RFC-0001:C-SEARCH-STRATEGY]])."""

    args: list[Scalar] = Field(min_length=1, description="Candidate names swept jointly in the focused grid. CONSUMED.")
    rationale: str = Field(min_length=1, description="Why these args are admitted and plausibly interact. INFORMATIONAL.")
    pins: dict[str, Scalar] = Field(default_factory=dict, description="OFAT-best value for each non-gridded candidate, held fixed during the grid. CONSUMED.")

    @model_validator(mode="after")
    def _unique_args(self) -> "FocusedGrid":
        if len(self.args) != len(set(self.args)):
            raise ValueError("focused_grid.args has duplicate names")
        return self


class PrecisionBranch(BaseModel):
    name: str = Field(description="Branch identifier, e.g. nvfp4. CONSUMED.")
    checkpoint: str | None = Field(default=None, description="Served model path for this branch; the default served model and `<model>` output slug. CONSUMED.")
    hardware: str | None = Field(default=None, description="Recorded for humans/provenance, never consumed by the tooling. INFORMATIONAL.")
    fixed: dict[str, Scalar] = Field(default_factory=dict, description="Known-best args held constant for every config in the branch. CONSUMED.")
    candidate: list[CandidateArg] = Field(default_factory=list, description="Args to vary (<=10). CONSUMED.")
    constraints: list[Constraint] = Field(default_factory=list, description="Illegal-combination rules filtering generated configs. CONSUMED.")
    baseline: dict[str, Scalar] = Field(default_factory=dict, description="One starting value per candidate; the OFAT reference point. CONSUMED.")
    focused_grid: FocusedGrid | None = Field(default=None, description="Optional second-stage joint grid spec. CONSUMED.")

    @property
    def candidate_names(self) -> list[str]:
        return [c.name for c in self.candidate]

    def merged_baseline(self) -> dict[str, Scalar]:
        cfg = dict(self.fixed)
        cfg.update(self.baseline)
        return cfg

    @model_validator(mode="after")
    def _validate(self) -> "PrecisionBranch":
        names = self.candidate_names

        if len(names) != len(set(names)):
            raise ValueError(f"branch '{self.name}': duplicate candidate names")
        if len(names) > MAX_CANDIDATES:
            raise ValueError(
                f"branch '{self.name}': {len(names)} candidates exceeds cap {MAX_CANDIDATES} (C-SCOPE)"
            )

        overlap = set(names) & set(self.fixed)
        if overlap:
            raise ValueError(
                f"branch '{self.name}': args in both fixed and candidate (C-SCOPE): {sorted(overlap)}"
            )

        spec = (set(names) | set(self.fixed)) & SPEC_DECODE_ARGS
        if spec:
            raise ValueError(
                f"branch '{self.name}': speculative-decode args out of scope (C-SCOPE): {sorted(spec)}"
            )

        precision = set(names) & PRECISION_DEFINING_ARGS
        if precision:
            raise ValueError(
                f"branch '{self.name}': precision-defining args must be fixed, not candidate "
                f"(C-PRECISION-BRANCH): {sorted(precision)}"
            )

        for c in self.candidate:
            if c.name not in self.baseline:
                raise ValueError(
                    f"branch '{self.name}': baseline missing candidate '{c.name}'"
                )
            allowed = {_key(v) for v in c.values}
            if _key(self.baseline[c.name]) not in allowed:
                raise ValueError(
                    f"branch '{self.name}': baseline '{c.name}'={self.baseline[c.name]!r} "
                    f"is not one of its candidate values"
                )

        extra = set(self.baseline) - set(names)
        if extra:
            raise ValueError(
                f"branch '{self.name}': baseline assigns non-candidate args "
                f"(put constant values in 'fixed'): {sorted(extra)}"
            )

        if self.focused_grid is not None:
            fg = self.focused_grid
            cand = set(names)
            unknown = [a for a in fg.args if a not in cand]
            if unknown:
                raise ValueError(
                    f"branch '{self.name}': focused_grid.args are not candidates "
                    f"(C-SEARCH-STRATEGY): {sorted(unknown)}"
                )
            gridded = set(fg.args)
            for arg, val in fg.pins.items():
                if arg not in cand:
                    raise ValueError(
                        f"branch '{self.name}': focused_grid.pins arg '{arg}' is not a candidate"
                    )
                if arg in gridded:
                    raise ValueError(
                        f"branch '{self.name}': focused_grid.pins arg '{arg}' is also gridded; "
                        f"a gridded arg is swept, not pinned"
                    )
                allowed = {_key(v) for c in self.candidate if c.name == arg for v in c.values}
                if _key(val) not in allowed:
                    raise ValueError(
                        f"branch '{self.name}': focused_grid.pins '{arg}'={val!r} "
                        f"is not one of its candidate values"
                    )
            unpinned = sorted(cand - gridded - set(fg.pins))
            if unpinned:
                raise ValueError(
                    f"branch '{self.name}': non-gridded candidates must be pinned to their "
                    f"OFAT-best in focused_grid.pins (C-SEARCH-STRATEGY): {unpinned}"
                )

        declared = set(names) | set(self.fixed)
        for con in self.constraints:
            refs = set(con.when) | set(con.forbid) | set(con.require)
            undeclared = refs - declared
            if undeclared:
                raise ValueError(
                    f"branch '{self.name}': constraint '{con.name}' references undeclared "
                    f"args (add to fixed or candidate): {sorted(undeclared)}"
                )

        full = self.merged_baseline()
        for con in self.constraints:
            if con.violated_by(full):
                raise ValueError(
                    f"branch '{self.name}': baseline violates constraint '{con.name}'"
                )
        return self


class SLOOverride(BaseModel):
    """Per-(isl, osl) SLO override; an unset bound inherits the global value."""

    isl: int = Field(description="Input sequence length this override applies to. CONSUMED.")
    osl: int = Field(description="Output sequence length this override applies to. CONSUMED.")
    ttft_p95_ms: float | None = Field(default=None, gt=0, description="Per-pair TTFT target (report-only); inherits the global value when unset. CONSUMED (report-only).")
    per_token_ms: float | None = Field(default=None, gt=0, description="Per-pair decode-ITL gate; inherits the global value when unset. CONSUMED.")


class SLO(BaseModel):
    """SLO for [[RFC-0001:C-OBJECTIVE]]: per-token (ITL) is the gate; TTFT is report-only."""

    per_token_ms: float = Field(gt=0, description="Global decode-ITL gate in ms; the SLO that filters the frontier. CONSUMED.")
    ttft_p95_ms: float | None = Field(default=None, gt=0, description="Global TTFT p95 target in ms; displayed but never gates. CONSUMED (report-only).")
    overrides: list[SLOOverride] = Field(default_factory=list, description="Per-(isl, osl) SLO overrides. CONSUMED.")

    def _override(self, isl: int, osl: int) -> SLOOverride | None:
        for o in self.overrides:
            if o.isl == isl and o.osl == osl:
                return o
        return None

    def gate_per_token_ms(self, isl: int, osl: int) -> float:
        o = self._override(isl, osl)
        if o is not None and o.per_token_ms is not None:
            return o.per_token_ms
        return self.per_token_ms

    def ttft_target_ms(self, isl: int, osl: int) -> float | None:
        o = self._override(isl, osl)
        if o is not None and o.ttft_p95_ms is not None:
            return o.ttft_p95_ms
        return self.ttft_p95_ms


class QualityGate(BaseModel):
    """Accuracy acceptance gate ([[RFC-0001:C-QUALITY-GATE]]): its pass/fail threshold and
    evaluation dataset are defined before the search and evaluated per precision branch."""

    dataset: str = Field(description="Evaluation dataset name passed to the accuracy harness. CONSUMED.")
    metric: str = Field(default="accuracy", description="Score key compared against the threshold. CONSUMED.")
    threshold: float = Field(description="Pass/fail bound for the metric. CONSUMED.")
    direction: Literal["higher", "lower"] = Field(default="higher", description="`higher`: score >= threshold passes; `lower`: score <= threshold. CONSUMED.")

    def passes(self, score: float | None) -> bool:
        if score is None:
            return False
        if self.direction == "higher":
            return score >= self.threshold
        return score <= self.threshold


class SearchConfig(BaseModel):
    model: str = Field(description="Default served model; the `<model>` output slug when a branch sets no checkpoint. CONSUMED.")
    workload_axes: dict[str, Any] = Field(default_factory=dict, description="Inner-loop axes (isl_osl_pairs, report_isl_osl_pairs, concurrency); never permuted into restart-required configs. CONSUMED.")
    slo: SLO | None = Field(default=None, description="Decode-first objective SLO; required for the frontier. CONSUMED.")
    quality_gate: QualityGate | None = Field(default=None, description="Optional accuracy acceptance gate. CONSUMED.")
    precision_branches: list[PrecisionBranch] = Field(min_length=1, description="One or more precision branches to search. CONSUMED.")

    @model_validator(mode="after")
    def _unique_branches(self) -> "SearchConfig":
        ns = [b.name for b in self.precision_branches]
        if len(ns) != len(set(ns)):
            raise ValueError("duplicate precision branch names")
        return self

    def branch(self, name: str) -> PrecisionBranch:
        for b in self.precision_branches:
            if b.name == name:
                return b
        raise KeyError(f"no precision branch '{name}'")
