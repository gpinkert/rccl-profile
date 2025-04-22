import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import plotly.express as px
from models import Base
from ingest import ingest_zip

# --------------------------------------
# Database configuration
# --------------------------------------
# Use environment variable or default to MySQL on host network
# Assumes a MySQL container running on localhost:3306 with credentials dashuser/dashpass
default_db = os.getenv('DB_URL', 'mysql+pymysql://dashuser:dashpass@127.0.0.1/perf_dashboard')
engine = create_engine(default_db, echo=False)
Session = sessionmaker(bind=engine)
# Create tables if they don't exist
Base.metadata.create_all(engine)

# --------------------------------------
# Streamlit App Configuration
# --------------------------------------
st.set_page_config(page_title="RCCL & Coverage Dashboard", layout="wide")
st.title("RCCL & Coverage Dashboard")

# --------------------------------------
# Ingestion Section
# --------------------------------------
st.header("Upload .zip Runs")
uploaded = st.file_uploader("Upload .zip files", type='zip', accept_multiple_files=True)
if uploaded:
    session = Session()
    for f in uploaded:
        ingest_zip(f, session)
    session.close()

with st.expander("Ingest from server filesystem"):
    dirpath = st.text_input("Directory path", "/")
    if os.path.isdir(dirpath):
        items = [f for f in os.listdir(dirpath) if f.endswith('.zip')]
        to_ingest = st.multiselect("Select .zip files", items)
        if st.button("Ingest selected"):
            session = Session()
            for fn in to_ingest:
                ingest_zip(Path(dirpath) / fn, session)
            session.close()
    else:
        st.error("Invalid path provided")

# --------------------------------------
# Analysis Tabs
# --------------------------------------
tab1, tab2 = st.tabs(["Benchmarking", "Coverage"])

# Benchmarking Explorer
with tab1:
    st.header("Benchmarking Explorer")
    session = Session()
    rows = session.execute(text(
        "SELECT id, run_label FROM runs WHERE measurement_type='benchmark' ORDER BY timestamp"
    )).fetchall()
    session.close()

    if not rows:
        st.info("No benchmark runs available.")
    else:
        labels = [r.run_label for r in rows]
        chosen = st.multiselect("Select runs to view", labels, default=labels)
        if chosen:
            ids = [r.id for r in rows if r.run_label in chosen]
            placeholder = ','.join(['?'] * len(ids))
            df = pd.read_sql(
                f"SELECT * FROM benchmark_records WHERE run_id IN ({placeholder})", engine, params=ids
            )
            metrics_df = df['metrics'].apply(pd.Series)
            df = pd.concat([df.drop(columns=['metrics']), metrics_df], axis=1)
            df['size_gb'] = df['size_gb'].astype(float)

            # Sidebar filters
            st.sidebar.subheader("Filters")
            y_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['id','run_id']]
            y_axis = st.sidebar.selectbox("Y-axis metric", y_cols)
            envs = st.sidebar.multiselect("Env bundles", df['env_bundle'].unique(), default=list(df['env_bundle'].unique()))
            ops = st.sidebar.multiselect("Reduction ops", df['redop'].unique(), default=list(df['redop'].unique()))
            types = st.sidebar.multiselect("Datatypes", df['datatype'].unique(), default=list(df['datatype'].unique()))
            inplace_vals = st.sidebar.multiselect("In-place flags", df['in_place'].unique(), default=list(df['in_place'].unique()))
            size_min, size_max = df['size_gb'].min(), df['size_gb'].max()
            size_range = st.sidebar.slider("Message size (GB)", size_min, size_max, (size_min, size_max))

            fdf = df[
                df['env_bundle'].isin(envs) &
                df['redop'].isin(ops) &
                df['datatype'].isin(types) &
                df['in_place'].isin(inplace_vals) &
                df['size_gb'].between(*size_range)
                ]
            st.subheader("Filtered Benchmark Data")
            st.dataframe(fdf)

            for collective in fdf['collective'].unique():
                sub = fdf[fdf['collective'] == collective]
                st.markdown(f"### Collective: **{collective}**")
                if sub['size_gb'].nunique() == 1:
                    fig = px.bar(sub, x='redop', y=y_axis, color='env_bundle', barmode='group')
                else:
                    sub = sub.copy()
                    sub['combo'] = sub['datatype'] + '_' + sub['redop'] + '_' + sub['env_bundle']
                    fig = px.line(sub, x='size_gb', y=y_axis, color='combo', markers=True)
                    fig.update_layout(xaxis_title='Size (GB)', yaxis_title=y_axis)
                st.plotly_chart(fig, use_container_width=True)

# Coverage Explorer
with tab2:
    st.header("Coverage Over Time")
    session = Session()
    rows = session.execute(text(
        "SELECT id, run_label FROM runs WHERE measurement_type='coverage' ORDER BY timestamp"
    )).fetchall()
    session.close()

    if not rows:
        st.info("No coverage runs available.")
    else:
        labels = [r.run_label for r in rows]
        chosen = st.multiselect("Select coverage runs", labels, default=labels)
        if chosen:
            ids = [r.id for r in rows if r.run_label in chosen]
            placeholder = ','.join(['?'] * len(ids))
            dfc = pd.read_sql(
                f"SELECT * FROM coverage_records WHERE run_id IN ({placeholder})", engine, params=ids
            )
            dfc['run_label'] = pd.Categorical(
                dfc['run_id'].map({r.id: r.run_label for r in rows}),
                categories=chosen, ordered=True
            )
            metrics = ['function_cov', 'line_cov', 'region_cov', 'branch_cov']
            fig = px.line(dfc, x='run_label', y=metrics, markers=True)
            st.plotly_chart(fig, use_container_width=True)
