"""Permutation generator for the SGLang server-arg search.

Emits only constraint-valid restart-required configurations
([[RFC-0001:C-CONFIG-SOURCE]]). The two modes implement the staged search of
[[RFC-0001:C-SEARCH-STRATEGY]]: `ofat` varies one candidate at a time around the
baseline; `grid` runs a focused joint grid over a named subset of candidates. Only
restart-required server args are permuted here; workload axes are the inner loop
([[RFC-0001:C-LOOP-STRUCTURE]]) and are not expanded.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .schema import Constraint, PrecisionBranch, SearchConfig, _key


def config_hash(args: dict) -> str:
    """Content hash of a server-arg set, for result provenance ([[RFC-0001:C-MEASUREMENT]])."""
    blob = json.dumps({k: args[k] for k in sorted(args)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class ConfigPoint:
    branch: str
    mode: str
    label: str
    args: dict
    checkpoint: str | None = None
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = config_hash(self.args)

    def to_cli(self) -> str:
        return args_to_cli(self.args)


def is_valid(args: dict, constraints: list[Constraint]) -> bool:
    return not any(c.violated_by(args) for c in constraints)


def load_config(path) -> SearchConfig:
    data = yaml.safe_load(Path(path).read_text())
    return SearchConfig.model_validate(data)


def _assemble(branch: PrecisionBranch, overrides: dict) -> dict:
    cfg = dict(branch.fixed)
    cfg.update(branch.baseline)
    cfg.update(overrides)
    return cfg


def generate_ofat(branch: PrecisionBranch) -> list[ConfigPoint]:
    base = _assemble(branch, {})
    points = [ConfigPoint(branch.name, "ofat", "baseline", base, branch.checkpoint)]
    seen = {points[0].config_hash}
    for cand in branch.candidate:
        base_val = branch.baseline[cand.name]
        for v in cand.values:
            if _key(v) == _key(base_val):
                continue
            cfg = _assemble(branch, {cand.name: v})
            if not is_valid(cfg, branch.constraints):
                continue
            cp = ConfigPoint(branch.name, "ofat", f"{cand.name}={v}", cfg, branch.checkpoint)
            if cp.config_hash in seen:
                continue
            seen.add(cp.config_hash)
            points.append(cp)
    return points


def generate_grid(branch: PrecisionBranch, grid_args) -> list[ConfigPoint]:
    names = list(grid_args)
    for n in names:
        if n not in branch.candidate_names:
            raise KeyError(f"grid arg '{n}' is not a candidate in branch '{branch.name}'")
    value_lists = [next(c.values for c in branch.candidate if c.name == n) for n in names]
    points: list[ConfigPoint] = []
    seen: set[str] = set()
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(names, combo))
        cfg = _assemble(branch, overrides)
        if not is_valid(cfg, branch.constraints):
            continue
        label = ",".join(f"{n}={v}" for n, v in overrides.items())
        cp = ConfigPoint(branch.name, "grid", label, cfg, branch.checkpoint)
        if cp.config_hash in seen:
            continue
        seen.add(cp.config_hash)
        points.append(cp)
    return points


def args_to_cli(args: dict) -> str:
    parts: list[str] = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        elif v is None or v == "none":
            continue
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate SGLang server-arg search configs (RFC-0001)."
    )
    p.add_argument("--config", required=True, help="Path to the versioned search config YAML")
    p.add_argument("--branch", required=True, help="Precision branch name")
    p.add_argument("--mode", choices=["ofat", "grid"], default="ofat")
    p.add_argument("--grid-args", nargs="+", default=[], help="Candidate names for --mode grid")
    p.add_argument("--format", choices=["label", "cli", "json"], default="label")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    branch = cfg.branch(a.branch)

    if a.mode == "ofat":
        points = generate_ofat(branch)
    else:
        if not a.grid_args:
            p.error("--mode grid requires --grid-args")
        points = generate_grid(branch, a.grid_args)

    for cp in points:
        if a.format == "cli":
            print(f"# {cp.label} [{cp.config_hash}]")
            print(cp.to_cli())
        elif a.format == "json":
            print(json.dumps(asdict(cp), default=str))
        else:
            print(f"{cp.config_hash}\t{cp.branch}/{cp.mode}\t{cp.label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
