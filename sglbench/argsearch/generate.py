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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .measure import label_slug, model_slug
from .schema import Branch, Constraint, SearchConfig, _key

DEFAULT_OUT_DIR = "out"


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
    branch_keys: dict = field(default_factory=dict)
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


def _assemble(branch: Branch, overrides: dict) -> dict:
    cfg = dict(branch.fixed)
    cfg.update(branch.baseline)
    cfg.update(overrides)
    return cfg


def generate_ofat(branch: Branch) -> list[ConfigPoint]:
    base = _assemble(branch, {})
    points = [ConfigPoint(branch.name, "ofat", "baseline", base, branch.branch_keys())]
    seen = {points[0].config_hash}
    for cand in branch.candidate:
        base_val = branch.baseline[cand.name]
        for v in cand.values:
            if _key(v) == _key(base_val):
                continue
            cfg = _assemble(branch, {cand.name: v})
            if not is_valid(cfg, branch.constraints):
                continue
            cp = ConfigPoint(branch.name, "ofat", f"{cand.name}={v}", cfg, branch.branch_keys())
            if cp.config_hash in seen:
                continue
            seen.add(cp.config_hash)
            points.append(cp)
    return points


def grid_args_for(branch: Branch, grid_args=None) -> list:
    """Resolve the admitted grid-arg set: explicit `grid_args`, else the branch's
    declared focused_grid ([[RFC-0001:C-SEARCH-STRATEGY]])."""
    if grid_args is not None:
        return list(grid_args)
    if branch.focused_grid is None:
        raise ValueError(
            f"branch '{branch.name}' declares no focused_grid; pass grid_args or add a "
            f"focused_grid block (C-SEARCH-STRATEGY)"
        )
    return list(branch.focused_grid.args)


def generate_grid(branch: Branch, grid_args=None) -> list[ConfigPoint]:
    names = grid_args_for(branch, grid_args)
    for n in names:
        if n not in branch.candidate_names:
            raise KeyError(f"grid arg '{n}' is not a candidate in branch '{branch.name}'")
    pins = {} if branch.focused_grid is None else dict(branch.focused_grid.pins)
    pins = {k: v for k, v in pins.items() if k not in set(names)}
    value_lists = [next(c.values for c in branch.candidate if c.name == n) for n in names]
    points: list[ConfigPoint] = []
    seen: set[str] = set()
    for combo in itertools.product(*value_lists):
        swept = dict(zip(names, combo))
        cfg = _assemble(branch, {**pins, **swept})
        if not is_valid(cfg, branch.constraints):
            continue
        label = ",".join(f"{n}={v}" for n, v in swept.items())
        cp = ConfigPoint(branch.name, "grid", label, cfg, branch.branch_keys())
        if cp.config_hash in seen:
            continue
        seen.add(cp.config_hash)
        points.append(cp)
    return points


def focused_grid_manifest(branch: Branch, grid_args, points) -> dict:
    """The inspectable record of the focused-grid selection: the admitted argument set,
    the interaction rationale, the OFAT-best pins applied to non-gridded candidates, and
    the emitted config identities ([[RFC-0001:C-SEARCH-STRATEGY]])."""
    names = list(grid_args)
    fg = branch.focused_grid
    pins = {} if fg is None else {k: v for k, v in fg.pins.items() if k not in set(names)}
    return {
        "branch": branch.name,
        "admitted_args": names,
        "rationale": "" if fg is None else fg.rationale,
        "pins": pins,
        "config_hashes": [p.config_hash for p in points],
    }


def write_grid_manifest(manifest: dict, out_dir) -> Path:
    path = Path(out_dir) / "focused_grid.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


def varied_args(points) -> set:
    """Argument names whose value is not identical across all `points`."""
    points = list(points)
    if len(points) < 2:
        return set()
    keys: set = set()
    for cp in points:
        keys |= set(cp.args)
    return {k for k in keys if len({_key(cp.args.get(k)) for cp in points}) > 1}


def accuracy_invariant_search(branch: Branch, points) -> bool:
    """True when every candidate argument that varies across `points` is accuracy-invariant
    ([[RFC-0001:C-QUALITY-GATE]]). A single-config search varies nothing and is vacuously
    invariant. Non-invariant or unknown candidates make the search accuracy-active."""
    invariant = {c.name for c in branch.candidate if c.accuracy_invariant}
    varied_candidates = varied_args(points) & set(branch.candidate_names)
    return varied_candidates.issubset(invariant)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _extreme_choices(values: list, baseline_val) -> list:
    """Range endpoints ordered most-stressful (furthest from the baseline) first.

    Numeric values use min/max as the endpoints; other values use the first and last of
    the declared (ordered) sweep. Duplicate endpoints collapse to one choice.
    """
    if all(_is_number(v) for v in values):
        ends = [min(values), max(values)]
    else:
        ends = [values[0], values[-1]]
    uniq: list = []
    for v in ends:
        if all(_key(v) != _key(u) for u in uniq):
            uniq.append(v)
    if _is_number(baseline_val) and all(_is_number(v) for v in values):
        uniq.sort(key=lambda v: abs(v - baseline_val), reverse=True)
    else:
        uniq.sort(key=lambda v: _key(v) == _key(baseline_val))
    return uniq


def baseline_config_point(branch: Branch) -> ConfigPoint:
    """The branch baseline configuration as a "baseline"-labeled point. This is the gate
    reference ([[RFC-0001:C-QUALITY-GATE]]): its accuracy is the per-branch baseline the
    spot-check and every gated config are judged against."""
    return ConfigPoint(branch.name, "baseline", "baseline", _assemble(branch, {}), branch.branch_keys())


def spot_check_point(branch: Branch, points) -> ConfigPoint | None:
    """The accuracy-invariant spot-check configuration ([[RFC-0001:C-QUALITY-GATE]]).

    Sets every varied accuracy-invariant candidate to an extreme of its range (others held
    at baseline), choosing a constraint-valid combination of those extremes. Returns the
    baseline point when nothing accuracy-invariant varies, or None when no constraint-valid
    all-extreme assignment exists.
    """
    invariant = {c.name for c in branch.candidate if c.accuracy_invariant}
    varied = varied_args(points) & invariant
    if not varied:
        base = _assemble(branch, {})
        return ConfigPoint(branch.name, "spot-check", "spot-check", base, branch.branch_keys())
    cands = [c for c in branch.candidate if c.name in varied]
    choice_lists = [_extreme_choices(c.values, branch.baseline[c.name]) for c in cands]
    for combo in itertools.product(*choice_lists):
        overrides = {c.name: v for c, v in zip(cands, combo)}
        cfg = _assemble(branch, overrides)
        if is_valid(cfg, branch.constraints):
            label = "spot-check:" + ",".join(f"{c.name}={v}" for c, v in zip(cands, combo))
            return ConfigPoint(branch.name, "spot-check", label, cfg, branch.branch_keys())
    return None


def config_dir(base, model: str, mode: str) -> Path:
    """The per-(model, mode) configs directory under `base` ([[ADR-0007]])."""
    return Path(base) / model_slug(model) / "configs" / mode


def write_dir(points, base=DEFAULT_OUT_DIR, *, model: str | None = None, mode: str | None = None) -> Path:
    """Write generated configs under `out/<model>/configs/<mode>/` ([[ADR-0007]]).

    Each config is `<NN>-<label-slug>-<hash8>.json` plus a per-mode `index.jsonl`; the full
    config_hash stays inside each record as the provenance identity ([[RFC-0001:C-MEASUREMENT]]).
    """
    points = list(points)
    if model is None:
        model = "model"
    if mode is None:
        mode = points[0].mode if points else "ofat"
    outdir = config_dir(base, model, mode)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "index.jsonl").open("w") as idx:
        for n, cp in enumerate(points, 1):
            rec = asdict(cp)
            fname = f"{n:02d}-{label_slug(cp.label)}-{cp.config_hash[:8]}.json"
            (outdir / fname).write_text(json.dumps(rec, indent=2, default=str) + "\n")
            idx.write(json.dumps(rec, default=str) + "\n")
    return outdir


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
    p.add_argument(
        "--results",
        default=f"{DEFAULT_OUT_DIR}/results.jsonl",
        help="OFAT results JSONL; when present and a quality_gate is defined, --mode grid "
        "refuses pins whose OFAT config failed the gate (C-QUALITY-GATE)",
    )
    p.add_argument("--format", choices=["label", "cli", "json"], default="label")
    p.add_argument(
        "--save",
        action="store_true",
        help=f"Write configs to {DEFAULT_OUT_DIR}/<model>/configs/<mode>/ "
        f"(<NN>-<label>-<hash8>.json + index.jsonl) instead of printing to stdout",
    )
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="Base output dir (default: out)")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    branch = cfg.branch(a.branch)

    grid_names = None
    if a.mode == "ofat":
        points = generate_ofat(branch)
    else:
        fg = branch.focused_grid
        if fg is None:
            p.error(
                f"--mode grid requires a focused_grid block in branch '{a.branch}' so the "
                f"admitted set and rationale are recorded (C-SEARCH-STRATEGY)"
            )
        grid_names = list(fg.args)
        if a.grid_args and list(a.grid_args) != grid_names:
            p.error(
                f"--grid-args {list(a.grid_args)} does not match the branch's declared "
                f"focused_grid.args {grid_names}; edit the config to change the admitted set"
            )
        if cfg.quality_gate is not None and Path(a.results).exists():
            from .objective import gate_failed_pins, load_results

            offenders = gate_failed_pins(branch, load_results(a.results), cfg.quality_gate)
            if offenders:
                detail = "; ".join(
                    f"{o['arg']}={o['value']} ({o['config_hash']})" for o in offenders
                )
                p.error(
                    f"focused_grid pins a value whose OFAT config failed the accuracy gate "
                    f"(C-QUALITY-GATE): {detail}. Re-pin to a gate-passing OFAT-best; if none "
                    f"exists, branch '{a.branch}' has no acceptable configuration."
                )
        points = generate_grid(branch, grid_names)
        print(f"# focused grid: {grid_names}\n# rationale: {fg.rationale}")

    if a.save:
        model = cfg.model
        outdir = write_dir(points, a.out, model=model, mode=a.mode)
        msg = (f"wrote {len(points)} configs to {outdir}/ "
               f"(index: {outdir / 'index.jsonl'})")
        if a.mode == "grid":
            manifest = focused_grid_manifest(branch, grid_names, points)
            mpath = write_grid_manifest(manifest, outdir)
            msg += f"\nwrote focused-grid manifest to {mpath}"
        print(msg)
        return 0

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
