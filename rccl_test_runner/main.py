import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from config_loader import load_yaml_config, get_config_path
from configuration import Configuration, VALID_COLLECTIVES
from executor import run_executable
from output_parser import parse_output_json
from stats import summarize_results, save_summary_csv

def run_tests(config: str, executable_dir: Path) -> Path:

    config_path = get_config_path(config)
    print(f"Config path {config_path}")
    raw_config = load_yaml_config(config_path)

    output_root = Path("results") / config_path.stem
    output_root.mkdir(parents=True, exist_ok=True)


    tests_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for test_name, test_dict in raw_config.items():
        test_cfg = Configuration.from_dict(test_dict)

        test_dir = output_root / test_name / tests_start_time
        test_dir.mkdir(parents=True, exist_ok=True)


        partial_conf = { test_name: test_dict }

        with open(test_dir / "config.yaml", "w") as f:
            yaml.safe_dump(partial_conf, f)
            print(test_name)
        print(f"Raw config {raw_config}")
        if test_cfg.collectives[0] == "all":
            test_cfg.collectives = VALID_COLLECTIVES
        
        for collective in test_cfg.collectives:
            run_executable(executable_dir / (str(collective) + str("_perf")), test_cfg, test_dir)

        all_results = []
        for exe in test_cfg.collectives:
            out_file = test_dir / f"{Path(exe + str('_perf')).stem}.json"
            all_results.extend(parse_output_json(out_file))


    return output_root

def launch_dashboard(results_path: Path):
    """
    Optionally launch the Streamlit dashboard, pointing to your results folders.
    This function calls 'streamlit run' as a subprocess.
    """
    # The path to your dashboard code
    dashboard_script = Path(__file__).parent / "dashboard_json.py"

    if not dashboard_script.exists():
        print(f"[WARNING] Dashboard script not found at {dashboard_script}, skipping launch.")
        return

    print("[INFO] Launching Streamlit dashboard...")
    cmd = [
        "streamlit", "run", str(dashboard_script),
        # Optional argument: if you want to pre-populate the sidebar text input
        # e.g., -- is a delimiter for additional arguments passed to the Streamlit app
        "--",
        "--parent_dir", str(results_path)
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RCCL benchmarks from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Name of configuration file. Does not need `.yaml`")
    parser.add_argument("--executable_dir", type=Path, required=True, help="Directory of test executables")
    parser.add_argument("--show_dashboard", action="store_true", help="Launch Streamlit after tests")

    args = parser.parse_args()

    results_folder = run_tests(args.config, args.executable_dir)

    if args.show_dashboard:
        launch_dashboard(results_folder)
    else:
        print(f"\n[INFO] All tests finished. Results in: {results_folder}")
        print("[INFO] Run `streamlit run rccl_test_runner/dashboard_json.py` to visualize.\n")
