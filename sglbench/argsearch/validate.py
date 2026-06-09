"""Validate a search config and summarize it ([[RFC-0001:C-CONFIG-SOURCE]]).

Exit 0 and print a per-branch summary when the config is valid; exit 1 and print
the validation error otherwise.
"""

from __future__ import annotations

import argparse
import sys

from .generate import generate_ofat, load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Validate an SGLang arg-search config (RFC-0001).")
    p.add_argument("config", help="Path to the search config YAML")
    a = p.parse_args(argv)

    try:
        cfg = load_config(a.config)
    except Exception as e:
        print(f"INVALID {a.config}: {e}", file=sys.stderr)
        return 1

    print(f"OK {a.config}")
    print(f"model: {cfg.model}")
    for b in cfg.precision_branches:
        print(
            f"  branch {b.name}: {len(b.candidate)} candidates, "
            f"{len(b.constraints)} constraints, {len(generate_ofat(b))} OFAT configs"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
