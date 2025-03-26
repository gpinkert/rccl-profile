from pathlib import Path
from typing import Dict, Any
import yaml


def get_config_path(config: str) -> Path:
    if not ".yaml" in config:
        config += ".yaml"
    config_path = Path.cwd()/ "configs" / config

    return config_path

def load_yaml_config(config_path: Path) -> Dict[str, Any]:

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML content must be a dictionary of test entries.")

    return config

if __name__ == "__main__":
    from pprint import pprint
    pprint(load_yaml_config("all_test.yaml"))
