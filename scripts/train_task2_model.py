"""Train a source-domain diagnostic model for task 2."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tasks.task2 import run_training as run_task2_training

LOGGER = logging.getLogger("task2_training")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_training(
    config_path: Path,
    feature_table_override: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
):
    config = _load_yaml(config_path)
    return run_task2_training(
        config,
        feature_table_override=feature_table_override,
        output_dir_override=output_dir_override,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a source-domain diagnostic model (task 2).")
    parser.add_argument("--config", type=Path, default=Path("config/task2_config.yaml"), help="YAML configuration path")
    parser.add_argument("--feature-table", type=Path, default=None, help="Optional override for the feature CSV path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for the output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_training(args.config, feature_table_override=args.feature_table, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
