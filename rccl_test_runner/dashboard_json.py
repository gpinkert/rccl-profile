import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import json
import numpy as np   # For aggregator functions
from pathlib import Path

@st.cache_data
def parse_run_folder(run_folder: Path) -> pd.DataFrame:
    config_path = run_folder / "config.yaml"
    env_dict = {}

    if config_path.exists():
        try:
            with config_path.open() as cf:
                config_data = yaml.safe_load(cf)
            # There's exactly one key at top-level in your config
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
        with json_file.open("r") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["collective"] = collective_name
                    all_records.append(rec)
                except json.JSONDecodeError:
                    continue

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["env_dict"] = [env_dict] * len(df)
    df["run_label"] = run_folder.name
    return df


@st.cache_data
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

def main():
    st.title("RCCL Perf JSON Results Dashboard")

    st.write("""
    **Features**:
    - Single entry per config.yaml in each run folder
    - Unify environment vars, labeling 'unset' if absent or '0'
    - If only 1 message size => bar chart with X=(redop, inPlace)
    - If multiple message sizes => line chart with color=(datatype+op+env)
    - **Message size slider** for subselecting a range (doesn't work very well)
    - **Single Collective Best Explorer** for either a single message size
      or an average across multiple message sizes
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

    chosen_folders = st.sidebar.multiselect(
        "Select run folders",
        possible_run_folders
    )
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
        st.warning("No 'size' column found; skipping slider.")
        df["size"] = 0

    if "redop" not in df.columns:
        df["redop"] = "no_reduction"
    df["redop"] = df["redop"].astype(str).str.strip()
    df["redop"] = df["redop"].replace({"": "no_reduction", "none": "no_reduction", " ": "no_reduction"})

    if "inPlace" not in df.columns:
        df["inPlace"] = -1

    if "type" not in df.columns:
        df["type"] = "unknown"

    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for plotting.")
        st.stop()

    y_axis_col = st.sidebar.selectbox("Y-Axis metric (for charts)", numeric_cols, index=0)

    min_size = float(df["size"].min())
    max_size = float(df["size"].max())
    if min_size == max_size:
        st.sidebar.write(f"Only one message size found: {min_size}")
        chosen_size_range = (min_size, max_size)
    else:
        range_span = max_size - min_size
        step_value = max(1.0, range_span / 10.0)  # dynamic step
        chosen_size_range = st.sidebar.slider(
            "Message size range",
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
    filtered = filtered[
        (filtered["size"] >= chosen_size_range[0]) &
        (filtered["size"] <= chosen_size_range[1])
        ]

    st.subheader("Filtered Data Preview (All Rows)")
    st.dataframe(filtered)

    if filtered.empty:
        st.warning("No data after filtering.")
        return

    these_collectives = sorted(filtered["collective"].unique())
    for col_name in these_collectives:
        sub = filtered[filtered["collective"] == col_name]
        if sub.empty:
            continue

        st.markdown(f"## Collective: **{col_name}**")
        unique_sizes = sub["size"].dropna().unique()

        if len(unique_sizes) == 1:
            st.write(f"Only one message size => bar chart, using {y_axis_col} on Y axis")
            fig = px.bar(
                sub,
                x=["redop", "inPlace"],
                y=y_axis_col,
                color="env_bundle",
                barmode="group",
                hover_data=["run_label", "type", "size"],
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
                x="size",
                y=y_axis_col,
                color="combo_label",
                line_dash=line_dash_arg,
                hover_data=["run_label", "redop", "type", "inPlace", "env_bundle"],
                title=f"{col_name}: Timings vs. size (y={y_axis_col})"
            )
            fig.update_layout(
                xaxis_title="Message Size",
                yaxis_title=y_axis_col
            )
            st.plotly_chart(fig, use_container_width=True)

    st.header("Single Collective Best Explorer")

    if not these_collectives:
        st.warning("No collectives found in the current filters.")
        return
    chosen_collective = st.selectbox("Pick a collective to analyze in detail",
                                     these_collectives)

    sub_collective = filtered[filtered["collective"] == chosen_collective]
    if sub_collective.empty:
        st.warning(f"No data for collective={chosen_collective} after filters.")
        return

    pick_mode = st.radio("Pick the mode",
                         ["Single message size", "Average across message sizes"],
                         index=0)

    unique_sizes_for_col = sorted(sub_collective["size"].unique())
    if pick_mode == "Single message size":
        if not unique_sizes_for_col:
            st.warning("No message sizes found for this collective.")
            return
        chosen_size = st.selectbox("Pick a single message size", unique_sizes_for_col)
        working_df = sub_collective[sub_collective["size"] == chosen_size]
        note_str = f"- Collective = `{chosen_collective}`, Single Size = `{chosen_size}`"
    else:
        working_df = sub_collective
        note_str = f"- Collective = `{chosen_collective}`, **All** message sizes in the current filter"

    if working_df.empty:
        st.warning("No data for the chosen mode.")
        return

    judge_metric = st.selectbox("Which metric to pick best environment?", numeric_cols)

    direction = st.selectbox("Interpretation of the chosen metric:",
                             ["Lower is better", "Higher is better"])
    ascending = (direction == "Lower is better")

    aggregator_choice = st.selectbox("Aggregator for duplicates or multiple runs:",
                                     ["mean", "median", "min", "max"])
    aggregator_map = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max
    }
    agg_func = aggregator_map[aggregator_choice]

    group_cols = ["redop", "type", "inPlace", "env_bundle"]
    grouped = (working_df
               .groupby(group_cols)[judge_metric]
               .agg(agg_func)
               .reset_index())

    grouped_sorted = grouped.sort_values(by=judge_metric, ascending=ascending)
    best_rows = grouped_sorted.groupby(["redop", "type", "inPlace"], as_index=False).first()

    st.markdown("### Best environment combos for each (Operation, Datatype, inPlace)")
    st.markdown(note_str)
    st.markdown(f"- Judging by `{judge_metric}` with aggregator = `{aggregator_choice}`")
    st.markdown(f"- **{'Lower' if ascending else 'Higher'}** is better")

    st.dataframe(best_rows)

    global_best = grouped_sorted.iloc[0]
    st.subheader("Overall Best Single Combination for that selection")
    st.write(f"- Operation (redop): `{global_best['redop']}`")
    st.write(f"- Datatype: `{global_best['type']}`")
    st.write(f"- inPlace: `{global_best['inPlace']}`")
    st.write(f"- Env Bundle: `{global_best['env_bundle']}`")
    st.write(f"- Aggregated `{judge_metric}`: **{global_best[judge_metric]}**")

if __name__ == "__main__":
    st.set_page_config(
        page_title="RCCL JSON Dashboard",
        layout="wide"
    )
    main()