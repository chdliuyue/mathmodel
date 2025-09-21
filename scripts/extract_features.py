"""Command line interface to run the feature engineering pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Allow running the script directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tasks.task1 import run_feature_pipeline


LOGGER = logging.getLogger("feature_pipeline")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_pipeline(config_path: Path, output_dir: Optional[Path] = None) -> None:
    config = _load_yaml(config_path)
    run_feature_pipeline(config, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract diagnostic features from source and target domains.")
    parser.add_argument("--config", type=Path, default=Path("config/dataset_config.yaml"), help="Path to the YAML configuration file.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Optional directory override for generated feature tables."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_pipeline(args.config, args.output_dir)


if __name__ == "__main__":
    main()
