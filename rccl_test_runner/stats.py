import pandas as pd
from pathlib import Path
from typing import List, Dict

def summarize_results(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df

    summary = (
        df.groupby(["type", "redop", "size", "inPlace"])
        .agg(time_avg=("time", "mean"),
             algBw_avg=("algBw", "mean"),
             busBw_avg=("busBw", "mean"))
        .reset_index()
    )
    return summary

def save_summary_csv(df: pd.DataFrame, output_dir: Path) -> None:
    if not df.empty:
        df.to_csv(output_dir / "results.csv", index=False)
