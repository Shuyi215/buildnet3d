"""Run reconstruct jobs from an experiments/*.yaml sweep file.
Run:
```
python run_sweep.py --sweep-yaml experiments/configs/loss_mult_scan.yaml
```
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import tyro
import yaml


MODEL_OVERRIDE_KEYS = {
    "fg_loss_relat_mult",
    "rgb_uncertainty_loss_mult",
    "semantic_uncertainty_loss_mult",
    "semantic_loss_mult",
    "use_rgb_uncertainty",
    "use_semantic_uncertainty",
    "rgb_uncertainty_delay_steps",
    "semantic_uncertainty_delay_steps",
    "N_reparam_samples",
    "train_num_rays_per_batch",
}

# Short tags for experiment names: fg1000_rgb1_sem1
KEY_ABBREV = {
    "fg_loss_relat_mult": "fg",
    "rgb_uncertainty_loss_mult": "rgb",
    "semantic_uncertainty_loss_mult": "sem",
    "semantic_loss_mult": "sl",
    "use_rgb_uncertainty": "ur",
    "use_semantic_uncertainty": "us",
    "rgb_uncertainty_delay_steps": "rd",
    "semantic_uncertainty_delay_steps": "sd",
    "N_reparam_samples": "nr",
    "train_num_rays_per_batch": "bs",
}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_sweep(sweep: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [_as_list(sweep[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("/", "-").replace(" ", "")


def format_run_name(
    prefix: str,
    overrides: Mapping[str, Any],
    abbrev: Optional[Mapping[str, str]] = None,
) -> str:
    """Build a compact name like ``loss_mult_scan_fg1000_rgb1_sem1``."""
    if not overrides:
        return prefix
    tags = abbrev or KEY_ABBREV
    parts = [
        f"{tags.get(k, k)}{_format_value(v)}"
        for k, v in sorted(overrides.items())
    ]
    return f"{prefix}_{'_'.join(parts)}"


@dataclass
class SweepConfig:
    """Launch training runs defined by a YAML sweep."""

    sweep_yaml: Path
    """Path to an experiments YAML file."""
    dry_run: bool = False
    """Print commands without executing."""
    max_runs: Optional[int] = None
    """Optional cap on number of runs (for smoke tests)."""


def main(cfg: SweepConfig) -> None:
    with cfg.sweep_yaml.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    method = spec.get("method", "uncertainty-semantic-sdf")
    data = spec.get("data", "buildnet3d/data/REAL/EXP1")
    output_dir = spec.get("output_dir", "./outputs")
    max_num_iterations = spec.get("max_num_iterations", 100001)
    name_prefix = spec.get("prefix") or spec.get("name_prefix") or cfg.sweep_yaml.stem
    name_abbrev = spec.get("name_abbrev")
    defaults: Dict[str, Any] = dict(spec.get("defaults") or {})
    combos = expand_sweep(spec.get("sweep") or {})

    if cfg.max_runs is not None:
        combos = combos[: cfg.max_runs]

    reconstruct = Path(__file__).resolve()
    reconstruct = reconstruct.with_name("reconstruct.py")

    for combo in combos:
        overrides = {**defaults, **combo}
        experiment_name = format_run_name(name_prefix, combo, abbrev=name_abbrev)

        cmd = [
            sys.executable,
            str(reconstruct),
            "--model-type",
            str(method),
            "--data",
            str(data),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            experiment_name,
            "--max-num-iterations",
            str(max_num_iterations),
        ]
        for key, value in overrides.items():
            if key not in MODEL_OVERRIDE_KEYS:
                raise KeyError(
                    f"Unknown override key '{key}'. Add it to MODEL_OVERRIDE_KEYS "
                    "and reconstruct.py if needed."
                )
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Print as "flag value" pairs (skip executable + script path on first lines).
        print(f"{cmd[0]} {cmd[1]}")
        for i in range(2, len(cmd), 2):
            flag = cmd[i]
            value = cmd[i + 1] if i + 1 < len(cmd) else ""
            print(f"{flag} {value}".rstrip())
        if cfg.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
