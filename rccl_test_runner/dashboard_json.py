# dashboard_json.py

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import plotly.express as px

from models import Base
from ingest import ingest_zip, ingest_coverage_json

# --------------------------------------
# Database setup
# --------------------------------------
default_db = os.getenv(
    'DB_URL',
    'mysql+pymysql://dashuser:dashpass@127.0.0.1/perf_dashboard'
)
engine = create_engine(default_db, echo=True)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

st.set_page_config(page_title='RCCL & Coverage Dashboard', layout='wide')
st.title('RCCL & Coverage Dashboard')
debug = st.sidebar.checkbox('Show debug info', value=False)

# --------------------------------------
# Unified ingest uploader
# --------------------------------------
st.header('Upload Runs (.zip for benchmarks, .json for coverage)')
uploads = st.file_uploader(
    'Select .zip and/or .json files',
    type=['zip', 'json'],
    accept_multiple_files=True
)
coverage_label = None
if uploads:
    zip_files = [f for f in uploads if f.name.endswith('.zip')]
    json_files = [f for f in uploads if f.name.endswith('.json')]
    if json_files:
        default_label = Path(json_files[0].name).stem
        coverage_label = st.text_input(
            'Label for coverage run(s)', value=default_label
        )
    if st.button('Ingest Selected Runs'):
        sess = Session()
        # ingest benchmarks
        for z in zip_files:
            ingest_zip(z, sess)
        # ingest coverage
        if json_files:
            if not coverage_label or not coverage_label.strip():
                st.error('Please enter a label for coverage run(s)')
            else:
                for j in json_files:
                    ingest_coverage_json(j, coverage_label.strip(), sess)
        sess.close()
        st.success('Ingested selected runs')

# --------------------------------------
# Tabs
# --------------------------------------
tab1, tab2 = st.tabs(['Benchmarking', 'Coverage'])

# Benchmarking Explorer placeholder
with tab1:
    st.header('Benchmarking Explorer')

    sess = Session()
    runs = sess.execute(text(
        "SELECT id, run_label FROM runs "
        "WHERE measurement_type='benchmark' ORDER BY timestamp"
    )).fetchall()
    sess.close()

    if not runs:
        st.info('No benchmark runs available.')
        st.stop()

    labels = [r.run_label for r in runs]
    chosen = st.multiselect('Select runs', labels, default=labels)
    if not chosen:
        st.stop()

    run_ids = [r.id for r in runs if r.run_label in chosen]
    q = ','.join(str(i) for i in run_ids)

    with engine.connect() as conn:
        data = conn.execute(text(
            f"SELECT * FROM benchmark_records WHERE run_id IN ({q})"
        )).mappings().all()
    df = pd.DataFrame(data)
    if df.empty:
        st.warning('No data for selected runs.')
        st.stop()

    # merge run labels
    with engine.connect() as conn:
        runs_df = pd.DataFrame(
            conn.execute(text(
                f"SELECT id AS run_id, run_label FROM runs WHERE id IN ({q})"
            )).mappings().all()
        )
    df = df.merge(runs_df, on='run_id', how='left')

    if debug:
        st.subheader('🐛 DEBUG: Raw data')
        st.write(df.head())

    df['size_gb'] = df['size'] / 1024**3

    # Sidebar filters
    st.sidebar.header('Filters')
    y_axis = st.sidebar.selectbox(
        'Y-axis metric',
        [c for c in ['time','bus_bw','alg_bw'] if c in df.columns]
    )
    envs = st.sidebar.multiselect(
        'Env bundles',
        df['env_bundle'].unique(),
        default=df['env_bundle'].unique()
    )
    ops = st.sidebar.multiselect(
        'Reduction ops',
        df['redop'].unique(),
        default=df['redop'].unique()
    )
    types = st.sidebar.multiselect(
        'Datatypes',
        df['datatype'].unique(),
        default=df['datatype'].unique()
    )
    inplace = st.sidebar.multiselect(
        'In-place',
        df['in_place'].unique(),
        default=df['in_place'].unique()
    )
    mn, mx = float(df['size_gb'].min()), float(df['size_gb'].max())
    size_range = st.sidebar.slider('Size GB', mn, mx, (mn, mx))

    fdf = df[
        df['env_bundle'].isin(envs) &
        df['redop'].isin(ops) &
        df['datatype'].isin(types) &
        df['in_place'].isin(inplace) &
        df['size_gb'].between(size_range[0], size_range[1])
        ]

    st.subheader('Filtered Benchmark Data')
    st.dataframe(fdf)

    for coll in fdf['collective'].unique():
        sub = fdf[fdf['collective'] == coll].copy()
        st.markdown(f"### Collective: **{coll}**")
        sub['combo'] = (
                sub['datatype'].astype(str) + '_' +
                sub['redop'].astype(str) + '_' +
                sub['env_bundle'].astype(str) + '_' +
                sub['in_place'].astype(str)
        )
        if sub['size_gb'].nunique() == 1:
            fig = px.bar(sub, x='combo', y=y_axis)
        else:
            fig = px.line(
                sub,
                x='size_gb',
                y=y_axis,
                color='combo',
                markers=True
            )
            fig.update_layout(xaxis_title='Size (GB)', yaxis_title=y_axis)
        st.plotly_chart(fig, use_container_width=True)

# Coverage Explorer
with tab2:
    st.header('Coverage Comparison')

    # Fetch existing coverage runs
    sess = Session()
    runs_cov = sess.execute(text(
        "SELECT id, run_label FROM runs WHERE measurement_type='coverage' ORDER BY timestamp"
    )).fetchall()
    sess.close()

    if not runs_cov:
        st.info('No coverage runs available. Ingest some using the uploader above.')
        st.stop()

    labels_cov = [r.run_label for r in runs_cov]
    chosen_cov = st.multiselect(
        'Select coverage runs to compare', labels_cov, default=labels_cov[:2]
    )
    if not chosen_cov:
        st.stop()

    run_ids = [r.id for r in runs_cov if r.run_label in chosen_cov]
    ids_str = ','.join(map(str, run_ids))

    # Load per-file records
    with engine.connect() as conn:
        cov_rows = conn.execute(text(
            f"SELECT * FROM coverage_records WHERE run_id IN ({ids_str})"
        )).mappings().all()
    dfc = pd.DataFrame(cov_rows)
    if dfc.empty:
        st.warning('No coverage data for selected runs.')
        st.stop()
    dfc['run_label'] = dfc['run_id'].map({r.id: r.run_label for r in runs_cov})

    if debug:
        st.subheader('🐛 DEBUG: Raw coverage_records')
        st.dataframe(dfc)

    # Per-file details
    st.subheader('Per-File Coverage Details')
    per_file = dfc[dfc['file'] != '__summary__'].copy()
    fmt = lambda pct, cov, tot: f"{pct:.1f}% ({cov}/{tot})" if tot else ''
    per_file['Function'] = per_file.apply(lambda r: fmt(r.function_cov, r.function_covered, r.function_count), axis=1)
    per_file['Line']     = per_file.apply(lambda r: fmt(r.line_cov,     r.line_covered,     r.line_count),     axis=1)
    per_file['Region']   = per_file.apply(lambda r: fmt(r.region_cov,   r.region_covered,   r.region_count),   axis=1)
    per_file['Branch']   = per_file.apply(lambda r: fmt(r.branch_cov,   r.branch_covered,   r.branch_count),   axis=1)
    st.dataframe(
        per_file[['run_label','file','Function','Line','Region','Branch']]
        .rename(columns={'run_label':'Run','file':'File'})
    )

    # Run-level summary from stored totals
    st.subheader('Run-Level Coverage Summary (JSON totals)')
    sql = f"""
        SELECT run_label,
               total_function_percent,
               total_line_percent,
               total_region_percent,
               total_branch_percent
          FROM runs
         WHERE measurement_type='coverage'
           AND id IN ({ids_str})
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    df_runs = pd.DataFrame(rows)

    # reorder & rename columns
    df_runs = df_runs[['run_label', 'total_function_percent', 'total_line_percent',
                       'total_region_percent', 'total_branch_percent']]
    df_runs.columns = ['Run', 'Function', 'Line', 'Region', 'Branch']
    st.dataframe(df_runs)

    # Bar charts with distinct colors and no legend
    st.subheader('Coverage Comparison Bar Charts')
    color_seq = px.colors.qualitative.Plotly
    for col in ['Function','Line','Region','Branch']:
        fig = px.bar(
            df_runs,
            x='Run',
            y=col,
            color='Run',
            color_discrete_sequence=color_seq,
            title=f'{col}',
            labels={col:f'{col} (%)'}
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0,100])
        if len(chosen_cov) == 1:
            fig.update_traces(width=0.3)
        st.plotly_chart(fig, use_container_width=True)
