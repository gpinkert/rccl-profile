import json
import yaml
import pandas as pd
from pathlib import Path

def parse_run_folder(run_folder: Path) -> pd.DataFrame:

    config_path = run_folder / "config.yaml"
    env_str = "unknown"

    if config_path.exists():
        try:
            with config_path.open("r") as cf:
                config_data = yaml.safe_load(cf)
            first_block = next(iter(config_data.values())) if isinstance(config_data, dict) else {}
            env_list = first_block.get("ENV_VARS", [])
            env_items = []
            for var in env_list:
                if isinstance(var, dict):
                    for k, v in var.items():
                        val_str = v.get("value", "")
                        env_items.append(f"{k}={val_str}")
            if env_items:
                env_str = ", ".join(env_items)
        except Exception as ex:
            print(f"[WARNING] Could not parse config.yaml in {run_folder}: {ex}")

    all_data = []
    for json_file in run_folder.glob("*_perf.json"):
        collective_name = json_file.stem.replace("_perf", "")

        with json_file.open("r") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["collective"] = collective_name
                    all_data.append(record)
                except json.JSONDecodeError:
                    continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["env_config"] = env_str
    df["run_label"] = run_folder.name
    return df