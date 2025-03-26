
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List

from config_loader import load_yaml_config, get_config_path
from configuration import Configuration, VALID_COLLECTIVES
from executor import run_executable
from output_parser import parse_output_json
from stats import summarize_results, save_summary_csv

def run_tests(config: str, executable_dir: Path) -> None:

    config_path = get_config_path(config)

    raw_config = load_yaml_config(config_path)

    output_root = Path("results") / config_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    for test_name, test_dict in raw_config.items():
        test_cfg = Configuration.from_dict(test_dict)

        test_dir = output_root / test_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, test_dir / "config.yaml")

        if test_cfg.collectives[0] == "all":
            for collective in VALID_COLLECTIVES:
                run_executable(executable_dir / (str(collective) + str("_perf")), test_cfg, test_dir)
        else:
            for collective in test_cfg.collectives:
                run_executable(executable_dir / (str(collective) + str("_perf")), test_cfg, test_dir)

        all_results = []
        for exe in test_cfg.collectives:
            out_file = test_dir / f"{Path(exe).stem}.json"
            all_results.extend(parse_output_json(out_file))

        summary_df = summarize_results(all_results)
        save_summary_csv(summary_df, test_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RCCL benchmarks from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Name of configuration file. Does not need `.yaml`")
    parser.add_argument("--executable_dir", type=Path, required=True, help="Directory of test executables")
    args = parser.parse_args()

    run_tests(args.config, args.executable_dir)