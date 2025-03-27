import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import json
from pathlib import Path

########################################
# 1) Parse a single run folder
########################################
def parse_run_folder(run_folder: Path) -> pd.DataFrame:
    """
    Reads config.yaml (single entry) for environment variables,
    then loads each *_perf.json file to gather timing data.

    Returns a DataFrame with columns like:
    [collective, type, redop, inPlace, time, size, env_config, run_label, ...]
    """
    config_path = run_folder / "config.yaml"
    env_str = "unknown"

    # Load single-block config
    if config_path.exists():
        try:
            with config_path.open() as cf:
                config_data = yaml.safe_load(cf)
            # There's exactly one key at top-level, e.g. "all_nccl_gdr_on"
            # Extract environment variables from that block
            single_key = next(iter(config_data.keys()))
            block = config_data[single_key]
            env_list = block.get("ENV_VARS", [])
            env_items = []
            for var_entry in env_list:
                # var_entry might look like: {'id': 'NCCL_GDRCOPY_ENABLE', 'value': 1}
                var_id = var_entry.get("id", "")
                var_val = var_entry.get("value", "")
                env_items.append(f"{var_id}={var_val}")
            if env_items:
                env_str = ", ".join(env_items)
        except Exception as ex:
            st.warning(f"Could not parse config.yaml in {run_folder}: {ex}")

    # For each *_perf.json file, parse lines of JSON
    all_records = []
    for json_file in run_folder.glob("*_perf.json"):
        # Collective name from filename, e.g. "all_gather_perf.json" => "all_gather"
        collective_name = json_file.stem.replace("_perf", "")

        with json_file.open("r") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # Inject the collective name
                    rec["collective"] = collective_name
                    all_records.append(rec)
                except json.JSONDecodeError:
                    continue

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Attach environment + run label
    df["env_config"] = env_str
    df["run_label"] = run_folder.name
    return df

########################################
# 2) Build multi-run data
########################################
def load_runs(folder_paths):
    """
    Given multiple run folders, parse each.
    Concatenate into one DataFrame.
    """
    df_list = []
    for folder in folder_paths:
        run_df = parse_run_folder(folder)
        if not run_df.empty:
            df_list.append(run_df)
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

########################################
# 3) Streamlit app
########################################
def main():
    st.title("RCCL JSON Results Dashboard")

    st.write("""
    **Features**:
    - Single entry per config.yaml in each run folder
    - One chart per collective
    - In-place vs. out-of-place lines/bars
    - If only 1 message size => grouped bar chart by operation
    - If multiple message sizes => line chart
    - Filter by environment variable, operation, inPlace, etc.
    """)

    # Sidebar: pick parent directory
    parent_dir = st.sidebar.text_input("Parent results directory", "results")
    parent_path = Path(parent_dir)
    if not parent_path.is_dir():
        st.error(f"Invalid parent directory: {parent_dir}")
        st.stop()

    # Discover subfolders that might contain config.yaml + *_perf.json
    possible_run_folders = []
    for p in parent_path.rglob("*"):
        if p.is_dir():
            json_files = list(p.glob("*_perf.json"))
            if json_files or (p / "config.yaml").exists():
                possible_run_folders.append(p)

    if not possible_run_folders:
        st.warning("No suitable run folders found.")
        st.stop()

    # Let user pick one or more
    chosen_folders = st.sidebar.multiselect("Select run folders", possible_run_folders, default=possible_run_folders[:1])
    if not chosen_folders:
        st.info("No run folders selected.")
        st.stop()

    # Load data
    df = load_runs(chosen_folders)
    if df.empty:
        st.warning("No data loaded from selected folders.")
        st.stop()

    # Convert numeric columns if needed
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")

    # If there's no 'redop' for non-reduction collectives, fill with e.g. 'N/A'
    if "redop" not in df.columns:
        df["redop"] = "N/A"
    df["redop"] = df["redop"].fillna("N/A")

    # Similarly, ensure we have 'inPlace' column for in-place info
    if "inPlace" not in df.columns:
        df["inPlace"] = -1  # Means we don't have that info

    # Unique sets for filtering
    all_envs = sorted(df["env_config"].dropna().unique()) if "env_config" in df.columns else []
    all_ops = sorted(df["redop"].unique())
    all_collectives = sorted(df["collective"].unique())
    all_inplaces = sorted(df["inPlace"].unique())

    # Sidebar filters
    chosen_envs = st.sidebar.multiselect("Env config(s)", all_envs, default=all_envs)
    chosen_ops = st.sidebar.multiselect("Operation(s)", all_ops, default=all_ops)
    chosen_inplaces = st.sidebar.multiselect("In-place?", all_inplaces, default=all_inplaces)

    # Filter
    filtered = df.copy()
    if chosen_envs:
        filtered = filtered[filtered["env_config"].isin(chosen_envs)]
    if chosen_ops:
        filtered = filtered[filtered["redop"].isin(chosen_ops)]
    if chosen_inplaces:
        filtered = filtered[filtered["inPlace"].isin(chosen_inplaces)]

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered.head(30))

    if filtered.empty:
        st.warning("No data after filtering.")
        return

    # For each collective, build a separate chart
    for col_name in all_collectives:
        sub = filtered[filtered["collective"] == col_name]
        if sub.empty:
            continue

        st.markdown(f"## Collective: **{col_name}**")

        unique_sizes = sub["size"].dropna().unique()

        if len(unique_sizes) == 1:
            # => Grouped bar chart by operation, separate bars for each env/inPlace/run_label

            st.write("Only one message size found => Grouped bar chart")

            if "time" not in sub.columns:
                st.warning("No 'time' column found in data.")
                continue

            # Build a label that merges env_config, inPlace, run_label
            # so each unique combo becomes a separate bar
            sub["combo_label"] = (
                    sub["env_config"] + "_" +
                    sub["inPlace"].astype(str) + "_" +
                    sub["run_label"]
            )

            # Plot bars grouped by 'redop' on X-axis
            fig = px.bar(
                sub,
                x="redop",             # each operation group
                y="time",
                color="combo_label",   # separate bar for each combo
                barmode="group",
                hover_data=["env_config", "inPlace", "run_label", "size"],
                title=f"{col_name}: Single message size bar chart"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # => Multiple sizes => line chart
            st.write("Multiple message sizes => Line chart")

            if "time" not in sub.columns:
                st.warning("No 'time' column found for line chart.")
                continue

            # We'll set x=size, y=time, color by inPlace or env_config if you prefer
            # Example: color by "inPlace", line_dash by "env_config" to highlight differences
            fig = px.line(
                sub,
                x="size",
                y="time",
                color="inPlace" if len(sub["inPlace"].unique()) > 1 else None,
                line_dash="env_config" if len(all_envs) > 1 else None,
                hover_data=["env_config", "run_label", "redop", "inPlace"],
                title=f"{col_name}: Timings vs. size"
            )
            fig.update_layout(
                xaxis_title="Message Size",
                yaxis_title="Time"
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="RCCL JSON Dashboard", layout="wide")
    main()
