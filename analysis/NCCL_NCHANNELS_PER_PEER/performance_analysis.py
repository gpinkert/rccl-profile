# find best performed env var value 
# out-of-place time, algBw, busBw are used as metric to evaluate performance
# for collective + dtype the best performed env value is selected by best value that out performs on majority of the msg sizes

import argparse
from pathlib import Path
import pandas as pd
import json
import itertools
from collections import Counter

def most_frequent_element(lst):
    count = Counter(lst)
    max_freq = max(count.values())
    most_common = [e for e, f in count.items() if f == max_freq]
    # might be ties
    return most_common[0]

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

def get_size_threshold():
    return 1024

def get_dtypes() -> list:
    return ["float", "half", "bfloat16", "fp8_e4m3", "fp8_e5m2"]

def get_collective_file_names() -> list:
    return ["alltoall", "alltoallv", "gather", "scatter", "sendrecv"]

def get_collectives() -> list:
    return ["AlltoAll", "AlltoAllv", "Gather", "Scatter", "SendRecv"]

def get_env_var_values() -> list:
    return [1,2,4,8,16,32]

def get_keys() -> list:
    return ["name", "type", "size", "NCCL_NCHANNELS_PER_PEER", "busBw"]

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
    df = df[df["inPlace"]==inplace]
    df = df[df["size"] > get_size_threshold()]
    df["size"] = df["size"].apply(human_readable_size)
    
    max_busbw_df = pd.DataFrame(columns=["collective", "dtype", "NCCL_NCHANNELS_PER_PEER max busbw across msg sizes"])
    combinations = itertools.product(get_collectives(), get_dtypes())
    for combo in combinations:
        coll = combo[0]
        dtype = combo[1]
        max_busbw_env_values = []
        df_temp = df[df["name"]==coll]
        df_temp = df_temp[df_temp["type"]==dtype]
        size_list = df_temp['size'].unique()
        for size in size_list:
            df_temp = df[df["name"]==coll]
            df_temp = df_temp[df_temp["type"]==dtype]
            df_temp = df_temp[df_temp["size"]==size]
            max_busbw_env_values.append(df_temp.loc[df_temp["busBw"].idxmax()]["NCCL_NCHANNELS_PER_PEER"])
        cur_best_env_value = most_frequent_element(max_busbw_env_values)
        print(f"for {coll} and {dtype} the max busbw NCCL_NCHANNELS_PER_PEER across msg sizes is {cur_best_env_value}")
        max_busbw_df.loc[len(max_busbw_df)] = [coll, dtype, cur_best_env_value]
    
    # group by collective 
    max_busbw_df.set_index(['collective'], inplace=True)
    print(max_busbw_df)
    max_busbw_df.to_excel(output_path) 
        
                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="directory to input json files", default="/rccl-workspace/rccl-profile/rccl_test_runner/results/nccl_nchannels_per_peer")
    parser.add_argument("--output_path", help="directory to output table", default="/rccl-workspace/rccl-profile/analysis/NCCL_NCHANNELS_PER_PEER/results/performance.xlsx")
    args = parser.parse_args()
    print(args.input_dir)
    create_tables(Path(args.input_dir), Path(args.output_path))