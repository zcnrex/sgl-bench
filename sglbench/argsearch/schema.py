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


class PrecisionBranch(BaseModel):
    name: str
    checkpoint: str | None = None
    hardware: str | None = None
    fixed: dict[str, Scalar] = Field(default_factory=dict)
    candidate: list[CandidateArg] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    baseline: dict[str, Scalar] = Field(default_factory=dict)

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


class SearchConfig(BaseModel):
    model: str
    workload_axes: dict[str, Any] = Field(default_factory=dict)
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
