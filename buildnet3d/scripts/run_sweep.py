"""Run reconstruct jobs from an experiments/*.yaml file.

Two YAML styles:

1) Cartesian sweep (same method, vary hyperparameters)::

    method: uncertainty-semantic-sdf
    defaults: {train_num_rays_per_batch: 1048}
    sweep:
      fg_loss_relat_mult: [1000, 500]

2) Explicit run list (different methods / flags, in order)::

    defaults: {train_num_rays_per_batch: 2048}
    runs:
      - name: neus
        method: neus
      - name: uncert_rgb
        method: uncertainty-semantic-sdf
        use_rgb_uncertainty: true
        use_semantic_uncertainty: false

Run::

    python buildnet3d/scripts/run_sweep.py --sweep-yaml experiments/configs/loss_mult_scan.yaml
    python buildnet3d/scripts/run_sweep.py --sweep-yaml experiments/configs/exp1_pipeline.yaml --dry-run
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    "eval_num_rays_per_batch",
    "eval_num_rays_per_chunk",
    "steps_per_eval_image",
    "steps_per_eval_batch",
    "steps_per_save",
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
    "eval_num_rays_per_batch": "ebs",
    "eval_num_rays_per_chunk": "ech",
    "steps_per_eval_image": "sei",
    "steps_per_eval_batch": "seb",
    "steps_per_save": "ss",
}

# Keys that describe the job itself, not reconstruct.py model overrides.
RUN_META_KEYS = {"name", "method"}


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


def build_jobs(
    spec: Mapping[str, Any],
    *,
    name_prefix: str,
    name_abbrev: Optional[Mapping[str, str]],
    default_method: str,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return list of (experiment_name, method, overrides)."""
    defaults: Dict[str, Any] = dict(spec.get("defaults") or {})
    jobs: List[Tuple[str, str, Dict[str, Any]]] = []

    if spec.get("runs") is not None:
        if spec.get("sweep"):
            raise ValueError("YAML cannot set both `runs` and `sweep`.")
        for i, raw in enumerate(spec["runs"]):
            if not isinstance(raw, Mapping):
                raise TypeError(f"runs[{i}] must be a mapping, got {type(raw)}")
            run = dict(raw)
            method = str(run.pop("method", default_method))
            name = run.pop("name", None)
            overrides = {**defaults, **run}
            if name is not None:
                experiment_name = f"{name_prefix}_{name}"
            else:
                experiment_name = format_run_name(
                    name_prefix, run, abbrev=name_abbrev
                )
            jobs.append((experiment_name, method, overrides))
        return jobs

    for combo in expand_sweep(spec.get("sweep") or {}):
        overrides = {**defaults, **combo}
        experiment_name = format_run_name(name_prefix, combo, abbrev=name_abbrev)
        jobs.append((experiment_name, default_method, overrides))
    return jobs


@dataclass
class SweepConfig:
    """Launch training runs defined by a YAML sweep / run list."""

    sweep_yaml: Path
    """Path to an experiments YAML file."""
    dry_run: bool = False
    """Print commands without executing."""
    max_runs: Optional[int] = None
    """Optional cap on number of runs (for smoke tests)."""
    start_from: int = 0
    """0-based index into the job list; skip earlier runs."""


def main(cfg: SweepConfig) -> None:
    with cfg.sweep_yaml.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    default_method = spec.get("method", "uncertainty-semantic-sdf")
    data = spec.get("data", "buildnet3d/data/REAL/EXP1")
    output_dir = spec.get("output_dir", "./outputs")
    max_num_iterations = spec.get("max_num_iterations", 100001)
    name_prefix = spec.get("prefix") or spec.get("name_prefix") or cfg.sweep_yaml.stem
    name_abbrev = spec.get("name_abbrev")

    jobs = build_jobs(
        spec,
        name_prefix=name_prefix,
        name_abbrev=name_abbrev,
        default_method=default_method,
    )
    jobs = jobs[cfg.start_from :]
    if cfg.max_runs is not None:
        jobs = jobs[: cfg.max_runs]

    reconstruct = Path(__file__).resolve().with_name("reconstruct.py")

    failed: List[str] = []
    succeeded: List[str] = []

    for idx, (experiment_name, method, overrides) in enumerate(jobs, start=1):
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
            if key in RUN_META_KEYS:
                continue
            if key not in MODEL_OVERRIDE_KEYS:
                raise KeyError(
                    f"Unknown override key '{key}'. Add it to MODEL_OVERRIDE_KEYS "
                    "and reconstruct.py if needed."
                )
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        print(f"\n=== [{idx}/{len(jobs)}] {experiment_name} ({method}) ===")
        print(f"{cmd[0]} {cmd[1]}")
        for i in range(2, len(cmd), 2):
            flag = cmd[i]
            value = cmd[i + 1] if i + 1 < len(cmd) else ""
            print(f"{flag} {value}".rstrip())
        if cfg.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            succeeded.append(experiment_name)
            print(f"[ok] {experiment_name}")
        else:
            failed.append(experiment_name)
            print(
                f"[fail] {experiment_name} (exit={result.returncode}); "
                "continuing with next job",
                file=sys.stderr,
            )

    if not cfg.dry_run:
        print(
            f"\nSweep done: {len(succeeded)} ok, {len(failed)} failed, "
            f"{len(jobs)} total"
        )
        if failed:
            print("Failed runs:")
            for name in failed:
                print(f"  - {name}")
            raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
