import os
import subprocess
from pathlib import Path
from typing import List, Optional

from configuration import Configuration

def format_size(size: str) -> str:
    return size if isinstance(size, str) and size.endswith("GB") else str(size)

def build_command(executable: Path, config: Configuration, output_path: Path) -> str:
    step_flag: str = ""
    step_value: Optional[int] = None
    if config.step_detail:
        if config.step_detail.type == "multiple":
            step_flag = "-f"
        elif config.step_detail.type == "increment":
            step_flag = "-i"
        else:
            raise KeyError(f"Invalid step type '{config.step_detail.type}'")
        step_value = config.step_detail.value

    cmd_parts = [
        f"./{str(executable)}",
        "-d", ",".join(config.datatype),
        "-b", format_size(config.start_size),
        "-e", format_size(config.end_size),
        f"{step_flag}", f"{str(step_value) if step_value else ""}",
        "-g", str(config.gpus_per_thread),
        "-n", str(config.iterations),
        "-o", ",".join(config.operation),
        "--output_file", str(output_path / f"{executable.stem}.json"),
        "--output_format", "json"
    ]

    return " ".join(cmd_parts)

def prepare_env(env_vars: List) -> dict:
    new_env = os.environ.copy()
    for var in env_vars:
        new_env[var.name] = str(var.value)
    return new_env

def run_executable(
        executable: Path,
        config: Configuration,
        output_path: Path
) -> None:
    cmd = build_command(executable, config, output_path)
    env = prepare_env(config.ENV_VARS)
    print(cmd)
    subprocess.run(cmd, env=env, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT, shell=True, check=True)