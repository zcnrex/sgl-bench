"""Config schema for the SGLang server-arg search.

Models the versioned source of truth for restart-required server configurations
([[RFC-0001:C-CONFIG-SOURCE]]). Arguments are bucketed into fixed / candidate /
constrained ([[RFC-0001:C-SCOPE]]), and precision is a top-level branch rather than
a search axis ([[RFC-0001:C-PRECISION-BRANCH]]).
"""

from __future__ import annotations

from typing import Any

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
    name: str
    values: list[Scalar] = Field(min_length=1)

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

    name: str
    description: str = ""
    when: dict[str, list[Scalar]] = Field(default_factory=dict)
    forbid: dict[str, list[Scalar]] = Field(default_factory=dict)
    require: dict[str, list[Scalar]] = Field(default_factory=dict)

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

    args: list[Scalar] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    pins: dict[str, Scalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_args(self) -> "FocusedGrid":
        if len(self.args) != len(set(self.args)):
            raise ValueError("focused_grid.args has duplicate names")
        return self


class PrecisionBranch(BaseModel):
    name: str
    checkpoint: str | None = None
    hardware: str | None = None
    fixed: dict[str, Scalar] = Field(default_factory=dict)
    candidate: list[CandidateArg] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    baseline: dict[str, Scalar] = Field(default_factory=dict)
    focused_grid: FocusedGrid | None = None

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

    isl: int
    osl: int
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    per_token_ms: float | None = Field(default=None, gt=0)


class SLO(BaseModel):
    """SLO for [[RFC-0001:C-OBJECTIVE]]: per-token (ITL) is the gate; TTFT is report-only."""

    per_token_ms: float = Field(gt=0)
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    overrides: list[SLOOverride] = Field(default_factory=list)

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


class SearchConfig(BaseModel):
    model: str
    workload_axes: dict[str, Any] = Field(default_factory=dict)
    slo: SLO | None = None
    precision_branches: list[PrecisionBranch] = Field(min_length=1)

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
