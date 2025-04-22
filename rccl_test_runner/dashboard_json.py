import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import json
import jsonlines
import numpy as np
from pathlib import Path

########################################
# Utility: read a JSON file in either array or line-based format
########################################
def read_json_file_records(json_file: Path):
    """
    Attempts to parse `json_file` as either:
    1) A JSON array (or single JSON object) via json.load().
    2) Fallback: line-by-line JSON objects via jsonlines.

    Returns a list of Python dict records.
    """
    records = []
    try:
        with json_file.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            records.append(data)
        elif isinstance(data, list):
            records.extend(data)
        else:
            raise ValueError("JSON top-level not an object or list.")
    except Exception:
        with json_file.open("r") as f:
            reader = jsonlines.Reader(f)
            for obj in reader:
                records.append(obj)
    return records

########################################
# 1) Parse a single run folder
########################################
def parse_run_folder(run_folder: Path) -> pd.DataFrame:
    config_path = run_folder / "config.yaml"
    env_dict = {}
    if config_path.exists():
        try:
            with config_path.open() as cf:
                config_data = yaml.safe_load(cf)
            single_key = next(iter(config_data.keys()))
            block = config_data[single_key]
            env_list = block.get("ENV_VARS", [])
            for var_entry in env_list:
                var_id = var_entry.get("id", "")
                var_val = str(var_entry.get("value", ""))
                env_dict[var_id] = var_val
        except Exception as ex:
            st.warning(f"Could not parse config.yaml in {run_folder}: {ex}")

    all_records = []
    for json_file in run_folder.glob("*_perf.json"):
        collective_name = json_file.stem.replace("_perf", "")
        try:
            file_records = read_json_file_records(json_file)
            for rec in file_records:
                rec["collective"] = collective_name
                all_records.append(rec)
        except Exception as ex:
            st.warning(f"Error reading {json_file}: {ex}")

    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(all_records)
    df["env_dict"] = [env_dict] * len(df)
    df["run_label"] = run_folder.name
    return df

########################################
# 2) Build multi-run data
########################################
def load_runs(folder_paths):
    df_list = []
    for folder in folder_paths:
        run_df = parse_run_folder(folder)
        if not run_df.empty:
            df_list.append(run_df)
    if not df_list:
        return pd.DataFrame()
    big_df = pd.concat(df_list, ignore_index=True)
    big_df = unify_env_vars(big_df)
    return big_df

def unify_env_vars(df: pd.DataFrame) -> pd.DataFrame:
    all_env_names = set()
    for envd in df["env_dict"]:
        all_env_names.update(envd.keys())
    sorted_envs = sorted(all_env_names)
    env_bundles = []
    for _, row in df.iterrows():
        row_dict = row["env_dict"]
        parts = []
        for env_name in sorted_envs:
            val = row_dict.get(env_name, "unset")
            if val == "0":
                val = "unset"
            parts.append(f"{env_name}={val}")
        bundle = ", ".join(parts)
        env_bundles.append(bundle)
    df["env_bundle"] = env_bundles
    return df

########################################
# 3) Streamlit App
########################################
def main():
    st.title("RCCL Perf JSON Results Dashboard")
    st.write("""
    **Features**:
    - Single entry per config.yaml in each run folder.
    - JSON data can be in an array with commas **or** line-based with one record per line.
    - Environment vars are unified into 'env_bundle' (with '0' → 'unset').
    - If only 1 message size => bar chart; if multiple => line chart.
    - Slider to filter by message size.
    - Single Collective Average Explorer (computing the average time for each grouping).
    - **New:** For each message size bucket, the app finds—for every collective/reduction op/datatype/inPlace grouping—the env var configuration with the lowest average time and produces a formatted text summary for **reduce**, **reduce_scatter**, and **all_reduce**.
    """)

    parent_dir = st.sidebar.text_input("Parent results directory", "results")
    parent_path = Path(parent_dir)
    if not parent_path.is_dir():
        st.error(f"Invalid parent directory: {parent_dir}")
        st.stop()

    possible_run_folders = []
    for p in parent_path.rglob("*"):
        if p.is_dir():
            json_files = list(p.glob("*_perf.json"))
            if json_files or (p / "config.yaml").exists():
                possible_run_folders.append(p)

    if not possible_run_folders:
        st.warning("No suitable run folders found under that directory.")
        st.stop()

    chosen_folders = st.sidebar.multiselect("Select run folders", possible_run_folders)
    if not chosen_folders:
        st.info("No run folders selected.")
        st.stop()

    df = load_runs(chosen_folders)
    if df.empty:
        st.warning("No data loaded from selected folders.")
        st.stop()

    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    else:
        st.warning("No 'size' column found; using 0 as a placeholder.")
        df["size"] = 0

    # Convert the size from bytes to gigabytes.
    df["size_gb"] = df["size"] / (1024 ** 3)

    if "redop" not in df.columns:
        df["redop"] = "no_reduction"
    df["redop"] = df["redop"].astype(str).str.strip().replace({"": "no_reduction", "none": "no_reduction", " ": "no_reduction"})

    if "inPlace" not in df.columns:
        df["inPlace"] = -1

    if "type" not in df.columns:
        df["type"] = "unknown"

    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for plotting.")
        st.stop()

    # For charts, allow user to choose any numeric metric.
    y_axis_col = st.sidebar.selectbox("Y-Axis metric (for charts)", numeric_cols, index=0)

    # Use size in GB for the slider.
    min_size = float(df["size_gb"].min())
    max_size = float(df["size_gb"].max())
    if min_size == max_size:
        st.sidebar.write(f"Only one message size found: {min_size} GB")
        chosen_size_range = (min_size, max_size)
    else:
        range_span = max_size - min_size
        step_value = max(0.1, range_span / 10.0)
        chosen_size_range = st.sidebar.slider(
            "Message size range (GB)",
            min_value=min_size,
            max_value=max_size,
            value=(min_size, max_size),
            step=step_value
        )

    all_envs = sorted(df["env_bundle"].unique())
    all_ops = sorted(df["redop"].unique())
    all_inplaces = sorted(df["inPlace"].unique())
    all_types = sorted(df["type"].unique())

    chosen_envs = st.sidebar.multiselect("Env bundle(s)", all_envs, default=all_envs)
    chosen_ops = st.sidebar.multiselect("Operation(s)", all_ops, default=all_ops)
    chosen_inplaces = st.sidebar.multiselect("In-place?", all_inplaces, default=all_inplaces)
    chosen_types = st.sidebar.multiselect("Datatypes", all_types, default=all_types)

    filtered = df[
        df["env_bundle"].isin(chosen_envs) &
        df["redop"].isin(chosen_ops) &
        df["inPlace"].isin(chosen_inplaces) &
        df["type"].isin(chosen_types)
        ]
    # Filter based on the gigabyte size values.
    filtered = filtered[
        (filtered["size_gb"] >= chosen_size_range[0]) &
        (filtered["size_gb"] <= chosen_size_range[1])
        ]

    st.subheader("Filtered Data Preview (All Rows)")
    st.dataframe(filtered)

    if filtered.empty:
        st.warning("No data after filtering.")
        return

    # Generate charts per collective.
    these_collectives = sorted(filtered["collective"].unique())
    for col_name in these_collectives:
        sub = filtered[filtered["collective"] == col_name]
        if sub.empty:
            continue

        st.markdown(f"## Collective: **{col_name}**")
        unique_sizes = sub["size_gb"].dropna().unique()
        if len(unique_sizes) == 1:
            st.write(f"Only one message size => bar chart, using {y_axis_col} on Y axis")
            fig = px.bar(
                sub,
                x=["redop", "inPlace"],
                y=y_axis_col,
                color="env_bundle",
                barmode="group",
                hover_data=["run_label", "type", "size_gb"],
                title=f"{col_name}: Single-size bar chart ({y_axis_col})"
            )
            fig.update_layout(
                xaxis_title="(Operation, InPlace)",
                yaxis_title=y_axis_col
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Multiple message sizes => line chart, using {y_axis_col} on Y axis")
            sub["combo_label"] = sub["type"] + "_" + sub["redop"] + "_" + sub["env_bundle"]
            line_dash_arg = "inPlace" if len(sub["inPlace"].unique()) > 1 else None
            fig = px.line(
                sub,
                x="size_gb",
                y=y_axis_col,
                color="combo_label",
                line_dash=line_dash_arg,
                hover_data=["run_label", "redop", "type", "inPlace", "env_bundle"],
                title=f"{col_name}: Timings vs. size (y={y_axis_col})"
            )
            fig.update_layout(
                xaxis_title="Message Size (GB)",
                yaxis_title=y_axis_col
            )
            st.plotly_chart(fig, use_container_width=True)

    ########################################
    # Single Collective Average Explorer
    ########################################
    st.header("Single Collective Average Explorer")
    if not these_collectives:
        st.warning("No collectives found in the current filters.")
        return

    chosen_collective = st.selectbox("Pick a collective to analyze in detail", these_collectives)
    sub_collective = filtered[filtered["collective"] == chosen_collective]
    if sub_collective.empty:
        st.warning(f"No data for collective={chosen_collective} after filters.")
        return

    pick_mode = st.radio("Pick the mode", ["Single message size", "Average across message sizes"], index=0)
    unique_sizes_for_col = sorted(sub_collective["size_gb"].unique())
    if pick_mode == "Single message size":
        if not unique_sizes_for_col:
            st.warning("No message sizes found for this collective.")
            return
        chosen_size = st.selectbox("Pick a single message size (GB)", unique_sizes_for_col)
        working_df = sub_collective[sub_collective["size_gb"] == chosen_size]
        note_str = f"- Collective = `{chosen_collective}`, Single Size = `{chosen_size}` GB"
    else:
        working_df = sub_collective
        note_str = f"- Collective = `{chosen_collective}`, **All** message sizes in current filter"

    if working_df.empty:
        st.warning("No data for the chosen mode.")
        return

    # Allow user to choose the timing metric to average.
    judge_metric = st.selectbox("Which timing metric to average?", [c for c in numeric_cols if c in working_df.columns])

    st.markdown("### Average time for each grouping (Operation, Datatype, InPlace, Env Bundle)")
    group_cols = ["redop", "type", "inPlace", "env_bundle"]
    grouped_avg = working_df.groupby(group_cols)[judge_metric].mean().reset_index().rename(columns={judge_metric: "average_time"})
    st.dataframe(grouped_avg)

    ########################################
    # Best Env Var Configurations by Message Size Bucket (Lowest Average Time)
    ########################################
    st.header("Best Env Var Configuration by Message Size Bucket (Lowest Average Time)")
    buckets = {
        "Below 0.5gb": filtered[filtered["size_gb"] < 0.5],
        "0.5gb-1gb": filtered[(filtered["size_gb"] >= 0.5) & (filtered["size_gb"] < 1)],
        "1gb-2gb": filtered[(filtered["size_gb"] >= 1) & (filtered["size_gb"] < 2)],
        "2gb-16gb": filtered[(filtered["size_gb"] >= 2) & (filtered["size_gb"] <= 16)]
    }

    bucket_summaries = {}  # dictionary to hold each bucket's best results.
    for bucket_label, bucket_df in buckets.items():
        st.subheader(bucket_label)
        if bucket_df.empty:
            st.write("No data in this bucket.")
            bucket_summaries[bucket_label] = pd.DataFrame()
            continue

        group_cols_bucket = ["collective", "redop", "type", "inPlace", "env_bundle"]
        grouped_bucket_avg = bucket_df.groupby(group_cols_bucket)[judge_metric].mean().reset_index().rename(columns={judge_metric: "average_time"})
        # For each combination, select the env_bundle with the lowest average_time.
        best_bucket = grouped_bucket_avg.sort_values(by="average_time").groupby(
            ["collective", "redop", "type", "inPlace"], as_index=False
        ).first()
        st.dataframe(best_bucket)
        bucket_summaries[bucket_label] = best_bucket

    ########################################
    # Create a Formatted Copy/Paste Text Summary for Selected Collectives
    ########################################
    st.header("Bucket Summary (Formatted for Copy/Paste)")
    summary_lines = []
    summary_lines.append("Summary for Collectives: reduce, reduce_scatter, and all_reduce")
    summary_lines.append("=" * 50)
    for bucket_label, best_bucket in bucket_summaries.items():
        summary_lines.append(f"\nBucket: {bucket_label}")
        summary_lines.append("-" * 40)
        if best_bucket.empty:
            summary_lines.append("  No data in this bucket.\n")
        else:
            # Only include rows for the specified collectives.
            for _, row in best_bucket.iterrows():
                if row["collective"].strip().lower() in {"reduce", "reduce_scatter", "all_reduce"}:
                    summary_lines.append(f"Collective: {row['collective']}")
                    summary_lines.append(f"  Redop      : {row['redop']}")
                    summary_lines.append(f"  Type       : {row['type']}")
                    summary_lines.append(f"  InPlace    : {row['inPlace']}")
                    summary_lines.append(f"  Best Env   : {row['env_bundle']}")
                    summary_lines.append(f"  Avg. Time  : {row['average_time']:.3f}")
                    summary_lines.append("")  # extra blank line between groups
    summary_text = "\n".join(summary_lines)
    st.text_area("Copy and Paste the Following Bucket Summary", summary_text, height=400)

if __name__ == "__main__":
    st.set_page_config(
        page_title="RCCL JSON Dashboard",
        layout="wide"
    )
    main()
