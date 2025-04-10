# parse data with non-zero #wrong
import argparse
from pathlib import Path
import pandas as pd
import json


def human_readable_size(size_in_bytes):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = size_in_bytes
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
        
    return f"{size:.0f}{units[unit_index]}"

def get_inplace_outofplace() -> int:
    return 0

def get_dtypes() -> list:
    return ["float", "half", "bfloat16", "fp8_e4m3", "fp8_e5m2"]

def get_collective_file_names() -> list:
    return ["all_gather", "all_reduce", "alltoall", "scatter","broadcast", "gather", "reduce", "reduce_scatter", "sendrecv"]

def get_collectives() -> list:
    return ["AllGather", "AllReduce", "AlltoAll", "Scatter","Broadcast", "Gather", "Reduce", "Reduce_scatter", "Sendrecv"]

def get_env_var_values() -> list:
    return [1,2,3,4,5,6,7,8,9,10,11,12,15,16,32,64,128,1023,1024]

def get_keys() -> list:
    return ["NCCL_NCHANNELS_PER_PEER", "name", "type", "size", "wrong"]

def create_tables(root_dir:Path, output_path: Path):
    result = []
    inplace = get_inplace_outofplace()
    for var in get_env_var_values():
        dir = Path(root_dir / f"nccl_nchannels_per_peer_test_{var}")
        json_files = list(dir.rglob("*.json"))
        for f in json_files:
            with open(f, "r") as fp:
                for line in fp:
                    obj = json.loads(line)
                    obj["NCCL_NCHANNELS_PER_PEER"] = var
                    result.append(obj)
    
    df = pd.DataFrame(result)
    df = df[df["inPlace"] == inplace]
    df = df[get_keys()]
    df = df[~df["wrong"].isin(["0", "N/A"])]
    df = df[df["type"].isin(get_dtypes())]
    df.set_index(['NCCL_NCHANNELS_PER_PEER', 'name', 'type'], inplace=True)
    df["size"] = df["size"].apply(human_readable_size)
    # print(df.head(50))
    df.to_excel(output_path)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="directory to input json files", default="/rccl-workspace/rccl-profile/rccl_test_runner/results/nccl_nchannels_per_peer")
    parser.add_argument("--output_path", help="directory to output table", default="/rccl-workspace/rccl-profile/analysis/NCCL_NCHANNELS_PER_PEER/results/functionality.xlsx")
    args = parser.parse_args()
    print(args.input_dir)
    create_tables(Path(args.input_dir), Path(args.output_path))
    