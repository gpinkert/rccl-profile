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
# Ingest .zip Runs
# --------------------------------------
st.header('Upload .zip Runs')
ufs = st.file_uploader('Upload .zip files', type='zip', accept_multiple_files=True)
if ufs:
    sess = Session()
    for f in ufs:
        ingest_zip(f, sess)
    sess.close()

with st.expander('Ingest from server filesystem'):
    dp = st.text_input('Directory path', '/')
    if os.path.isdir(dp):
        zips = [f for f in os.listdir(dp) if f.endswith('.zip')]
        sel = st.multiselect('Select .zip files', zips)
        if st.button('Ingest selected'):
            sess = Session()
            for z in sel:
                ingest_zip(Path(dp) / z, sess)
            sess.close()
    else:
        st.error('Invalid path')

# Tabs
tab1, tab2 = st.tabs(['Benchmarking', 'Coverage'])

# Benchmarking placeholder
with tab1:
    st.header('Benchmarking Explorer')
    st.info('… your existing benchmarking UI …')

# Coverage Explorer
with tab2:
    st.header('Coverage Ingestion & Comparison')

    # 1) Upload & label new coverage JSON(s)
    uploaded = st.file_uploader(
        'Upload one or more llvm-cov JSON files', type='json', accept_multiple_files=True
    )
    if uploaded:
        default_label = Path(uploaded[0].name).stem
        label = st.text_input('Label for this coverage run', value=default_label)
        if st.button('Ingest Coverage Run'):
            if not label.strip():
                st.error('You must enter a non-empty label.')
            else:
                sess = Session()
                for f in uploaded:
                    ingest_coverage_json(f, label.strip(), sess)
                sess.close()
                st.success(f'Ingested coverage run "{label.strip()}"')

    st.markdown('---')

    # 2) Select existing coverage runs
    sess = Session()
    runs_cov = sess.execute(text(
        "SELECT id, run_label FROM runs WHERE measurement_type='coverage' ORDER BY timestamp"
    )).fetchall()
    sess.close()

    if not runs_cov:
        st.info('No coverage runs ingested yet.')
        st.stop()

    labels_cov = [r.run_label for r in runs_cov]
    chosen_cov = st.multiselect('Select coverage runs to compare', labels_cov, default=labels_cov[:2])
    if not chosen_cov:
        st.stop()

    run_ids = [r.id for r in runs_cov if r.run_label in chosen_cov]

    # 3) Fetch per-file coverage records
    ids_str = ','.join(str(i) for i in run_ids)
    with engine.connect() as conn:
        cov_rows = conn.execute(text(
            f"SELECT * FROM coverage_records WHERE run_id IN ({ids_str})"
        )).mappings().all()

    dfc = pd.DataFrame(cov_rows)
    if dfc.empty:
        st.warning('No coverage data for those runs.')
        st.stop()

    dfc['run_label'] = dfc['run_id'].map({r.id: r.run_label for r in runs_cov})

    if debug:
        st.subheader('🐛 DEBUG: Raw coverage_records')
        st.dataframe(dfc)

    # 4) Per-file Coverage Details
    st.subheader('Per-File Coverage Details')
    per_file = dfc[dfc['file'] != '__summary__'].copy()
    def fmt(pct, cov, tot): return f"{pct:.1f}% ({cov}/{tot})" if tot else ''
    per_file['Function'] = per_file.apply(lambda r: fmt(r.function_cov, r.function_covered, r.function_count), axis=1)
    per_file['Line']     = per_file.apply(lambda r: fmt(r.line_cov,     r.line_covered,     r.line_count),     axis=1)
    per_file['Region']   = per_file.apply(lambda r: fmt(r.region_cov,   r.region_covered,   r.region_count),   axis=1)
    per_file['Branch']   = per_file.apply(lambda r: fmt(r.branch_cov,   r.branch_covered,   r.branch_count),   axis=1)
    st.dataframe(
        per_file[['run_label','file','Function','Line','Region','Branch']]
        .rename(columns={'run_label':'Run','file':'File'})
    )

    # 5) Run-level summary from stored totals
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
    # reorder columns
    df_runs = df_runs[['run_label', 'total_function_percent', 'total_line_percent', 'total_region_percent', 'total_branch_percent']]
    df_runs.columns = ['Run', 'Function', 'Line', 'Region', 'Branch']
    st.dataframe(df_runs)

    # 6) Bar charts for stored totals
    st.subheader('Coverage Comparison Bar Charts')
    for col in ['Function','Line','Region','Branch']:
        fig = px.bar(df_runs, x='Run', y=col, title=f'{col}', labels={col:f'{col} (%)'})
        fig.update_yaxes(range=[0,100])
        if len(chosen_cov)==1:
            fig.update_traces(width=0.3)
        st.plotly_chart(fig, use_container_width=True)
