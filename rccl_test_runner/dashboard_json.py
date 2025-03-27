import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List
from parse_runs import parse_run_folder  # Or inline the parse_run_folder code

def load_multiple_runs(run_folders: List[Path]) -> pd.DataFrame:
    """Load all JSON data from the given run folders into one DataFrame."""
    df_list = []
    for folder in run_folders:
        run_df = parse_run_folder(folder)
        if not run_df.empty:
            df_list.append(run_df)
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def main():
    st.title("RCCL JSON Results Dashboard")

    st.write("""
    **Instructions**:
    1. In the sidebar, pick the parent directory containing your run folders.
    2. Select the final run folders (each with config.yaml + *_perf.json files).
    3. Compare the data with interactive filtering & plotting.
    """)

    parent_dir = st.sidebar.text_input("Parent results directory", "results")
    parent_path = Path(parent_dir)
    if not parent_path.is_dir():
        st.error(f"{parent_dir} is not a valid directory.")
        st.stop()

    # Recursively find all subfolders that contain config.yaml or *_perf.json
    # Adjust to your preference
    subfolders = []
    for p in parent_path.rglob("*"):
        if p.is_dir():
            # Heuristic: it must contain at least one *_perf.json or config.yaml to be considered
            json_files = list(p.glob("*_perf.json"))
            if json_files or (p / "config.yaml").exists():
                subfolders.append(p)

    if not subfolders:
        st.warning("No suitable run folders found in the directory.")
        st.stop()

    # Let user pick multiple run folders
    chosen_folders = st.sidebar.multiselect("Run folders", subfolders, default=subfolders[:1])
    if not chosen_folders:
        st.info("Please select at least one run folder.")
        st.stop()

    # Load the data
    full_df = load_multiple_runs(chosen_folders)
    if full_df.empty:
        st.warning("No data loaded from the selected folders.")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(full_df.head(20))

    # We can guess some important columns
    # e.g. 'size', 'type', 'redop', 'time', 'env_config', 'collective'
    # Adjust for your actual JSON structure
    columns = list(full_df.columns)
    st.write("Detected columns:", columns)

    # Convert 'size' or other numeric columns if needed
    if "size" in columns:
        full_df["size"] = pd.to_numeric(full_df["size"], errors="coerce")

    # Sidebar filters for some known fields
    # If your JSON lines have 'type' or 'redop', etc., you can do:
    possible_collectives = sorted(full_df["collective"].dropna().unique()) if "collective" in columns else []
    chosen_collectives = st.sidebar.multiselect("Collective(s)", possible_collectives, default=possible_collectives)

    if "type" in columns:
        types = sorted(full_df["type"].dropna().unique())
        chosen_types = st.sidebar.multiselect("Datatype(s)", types, default=types)
        full_df = full_df[full_df["type"].isin(chosen_types)]

    if "redop" in columns:
        ops = sorted(full_df["redop"].dropna().unique())
        chosen_ops = st.sidebar.multiselect("Operation(s)", ops, default=ops)
        full_df = full_df[full_df["redop"].isin(chosen_ops)]

    if "collective" in columns and chosen_collectives:
        full_df = full_df[full_df["collective"].isin(chosen_collectives)]

    # Let user pick an environment config or run_label if present
    possible_envs = sorted(full_df["env_config"].dropna().unique()) if "env_config" in columns else []
    chosen_envs = st.sidebar.multiselect("Env Config(s)", possible_envs, default=possible_envs)
    if "env_config" in columns and chosen_envs:
        full_df = full_df[full_df["env_config"].isin(chosen_envs)]

    # If you have timing columns, e.g. 'time'
    # Possibly you have 'time' or 'time_avg'
    numeric_cols = full_df.select_dtypes(include=[float, int]).columns
    if len(numeric_cols) == 0:
        st.error("No numeric columns found to plot.")
        st.stop()

    y_col = st.selectbox("Y-axis metric", numeric_cols, index=0)

    # We'll do a line plot with size as X if it exists
    if "size" not in full_df.columns:
        st.warning("No 'size' column found. Using first numeric column for X-axis.")
        numeric_cols_list = list(numeric_cols)
        x_col = numeric_cols_list[0] if numeric_cols_list else None
    else:
        x_col = "size"

    # Combine some columns for color, e.g. collective + env_config
    color_choice = st.selectbox("Color lines by", ["collective", "env_config", "run_label", "redop", "type"])

    fig = px.line(
        full_df,
        x=x_col,
        y=y_col,
        color=color_choice,
        hover_data=["collective", "env_config", "run_label"] if "collective" in full_df.columns else None,
        title="RCCL JSON Results Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="RCCL JSON Dashboard", layout="wide")
    main()