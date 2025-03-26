import json
from pathlib import Path
from typing import List, Dict

def parse_output_json(file_path: Path) -> List[Dict]:
    """Extract valid JSON lines from the output file."""
    parsed = []
    with file_path.open("r") as f:
        for line in f:
            try:
                data = json.loads(line)
                parsed.append(data)
            except json.JSONDecodeError:
                continue
    return parsed