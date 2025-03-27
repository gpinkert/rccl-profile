import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import yaml

def load_run_data(run_folder: Path, csv_name: str = "results.csv") -> pd.DataFrame:
    """
    Load the CSV file from a run folder, parse config.yaml to read environment variables,
    and attach them as new columns to the DataFrame (e.g., env_config).

    Args:
        run_folder (Path): Path to the folder containing config.yaml and CSV
        csv_name (str): Name of the CSV file, defaults to 'results.csv'.

    Returns:
        pd.DataFrame: Data from CSV with an added 'env_config' and/or 'run_label' column
    """
    csv_path = run_folder / csv_name
    config_path = run_folder / "config.yaml"

    if not csv_path.exists():
        st.warning(f"No CSV found at {csv_path}, skipping...")
        return pd.DataFrame()

    # Load CSV
    df = pd.read_csv(csv_path)
    # If 'size' is numeric, convert
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")

    # Attempt to parse config.yaml
    env_str = "unknown"
    if config_path.exists():
        try:
            with config_path.open("r") as f:
                config_dict = yaml.safe_load(f)
            # Optionally parse environment variables from top-level or sub-level
            # For simplicity, let's gather them all into a single string
            # e.g., "SOME_VAR=on, ANOTHER_VAR=some_value"
            # This assumes just ONE test block in config.yaml. Adjust logic if multiple test blocks
            # or if you want more structured approach.
            #
            # If multiple test blocks are present, you'd have to decide which to pick.
            # Here we pick the first block from the dict, or build a set of all envs.
            first_block = next(iter(config_dict.values())) if isinstance(config_dict, dict) else {}
            env_list = first_block.get("ENV_VARS", [])
            # env_list is typically a list of dicts like: [{'SOME_VAR': {'value': 'on'}}]
            env_items = []
            for var in env_list:
                if isinstance(var, dict):
                    for k, v in var.items():
                        val_str = v.get("value", "")
                        env_items.append(f"{k}={val_str}")
            if env_items:
                env_str = ", ".join(env_items)
        except Exception as ex:
            st.warning(f"Could not parse environment from {config_path}: {ex}")

    # Add columns to identify this run
    df["env_config"] = env_str
    df["run_label"] = run_folder.name  # e.g. 'run_with_envA'

    return df

def main():
    st.title("RCCL Benchmark Multi-Run Comparison")

    st.write("""
    **Instructions**:
    1. In the sidebar, pick one or more folders that contain `results.csv` and `config.yaml`.
    2. Optionally, rename the `results.csv` if your file name differs.
    3. Filter the combined data by datatype, operation, message size, etc.
    4. Compare how different environment variables (from each run's config) affect performance.
    """)

    # Let the user pick a parent directory that contains run subfolders
    parent_dir = st.sidebar.text_input("Parent directory of run folders", "my_experiments")

    # List subfolders
    parent_path = Path(parent_dir)
    if not parent_path.is_dir():
        st.error("Invalid parent directory.")
        st.stop()

    subfolders = [f for f in parent_path.iterdir() if f.is_dir()]
    if not subfolders:
        st.warning(f"No subfolders found in {parent_dir}.")
        st.stop()

    # Let user pick one or more run folders
    selected_folders = st.sidebar.multiselect(
        "Select run folders",
        subfolders,
        default=subfolders[:1]  # pick the first by default if available
    )

    csv_name = st.sidebar.text_input("Name of CSV file", "results.csv")

    # Load data from each folder and combine
    df_list = []
    for folder in selected_folders:
        run_df = load_run_data(folder, csv_name=csv_name)
        if not run_df.empty:
            df_list.append(run_df)
    if not df_list:
        st.warning("No data loaded from the selected folders.")
        st.stop()

    full_df = pd.concat(df_list, ignore_index=True)

    st.subheader("Combined Data Preview")
    st.write(full_df.head(30))

    # Now let's do some filters
    columns = full_df.columns
    required_cols = {"type", "redop", "size"}
    if not required_cols.issubset(set(columns)):
        st.error(f"Missing expected columns {required_cols} in combined data.")
        st.stop()

    # Unique values
    types = sorted(full_df["type"].dropna().unique())
    ops = sorted(full_df["redop"].dropna().unique())
    sizes = sorted(full_df["size"].dropna().unique())
    envs = sorted(full_df["env_config"].dropna().unique()) if "env_config" in full_df.columns else []

    # Filter selectors in sidebar
    chosen_types = st.sidebar.multiselect("Datatypes", types, default=types)
    chosen_ops = st.sidebar.multiselect("Operations", ops, default=ops)
    chosen_sizes = st.sidebar.multiselect("Message sizes", sizes, default=sizes)
    chosen_envs = envs
    if "env_config" in full_df.columns:
        chosen_envs = st.sidebar.multiselect("Env Config(s)", envs, default=envs)

    filtered_df = full_df[
        full_df["type"].isin(chosen_types) &
        full_df["redop"].isin(chosen_ops) &
        full_df["size"].isin(chosen_sizes)
        ]
    if "env_config" in full_df.columns:
        filtered_df = filtered_df[filtered_df["env_config"].isin(chosen_envs)]

    if filtered_df.empty:
        st.warning("No data matches your filters.")
        return

    st.subheader("Filtered Data")
    st.write(filtered_df)

    # Let user pick which metric to graph on Y-axis
    possible_metrics = [c for c in filtered_df.columns if c.endswith("_avg") or c in ["time", "time_avg"]]
    if not possible_metrics:
        possible_metrics = list(filtered_df.select_dtypes(include=[float,int]).columns)
    if not possible_metrics:
        st.warning("No numeric columns found for plotting.")
        return

    y_axis_choice = st.selectbox("Select metric for Y-axis", possible_metrics)

    # Combine type + op for color
    filtered_df["type_op"] = filtered_df["type"] + "_" + filtered_df["redop"]
    # Convert size to numeric for x-axis
    # (should already be numeric from earlier, but just in case)
    filtered_df["size_numeric"] = pd.to_numeric(filtered_df["size"], errors="coerce")

    # If we have multiple runs with different environment variables,
    # we can use 'env_config' or 'run_label' as a facet or color dimension.
    color_dim = st.selectbox("Choose color dimension", ["type_op", "run_label", "env_config", "redop", "type"])

    # We can do a facet by env_config or run_label if you want
    # Let's do a checkbox for faceting
    facet_options = ["No facet", "Facet by env_config", "Facet by run_label"]
    facet_choice = st.selectbox("Facet option", facet_options)
    facet_col = None
    if facet_choice == "Facet by env_config" and "env_config" in filtered_df.columns:
        facet_col = "env_config"
    elif facet_choice == "Facet by run_label" and "run_label" in filtered_df.columns:
        facet_col = "run_label"

    # Now build Plotly figure
    fig = px.line(
        filtered_df,
        x="size_numeric",
        y=y_axis_choice,
        color=color_dim,
        facet_col=facet_col,
        facet_col_wrap=2 if facet_col else 0,
        hover_data=["type", "redop", "env_config", "run_label"]
    )
    fig.update_layout(
        xaxis_title="Message Size",
        yaxis_title=y_axis_choice,
        title="RCCL Benchmark Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="RCCL Multi-Run Dashboard", layout="wide")
    main()
