import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import plotly.express as px

from models import Base
from ingest import ingest_zip

# DB config
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

# Ingest UI
st.header('Upload .zip Runs')
ufs = st.file_uploader('Upload .zip files', type='zip', accept_multiple_files=True)
if ufs:
    s = Session()
    for f in ufs:
        ingest_zip(f, s)
    s.close()
with st.expander('Ingest from server filesystem'):
    dp = st.text_input('Directory path', '/')
    if os.path.isdir(dp):
        zips = [f for f in os.listdir(dp) if f.endswith('.zip')]
        sel = st.multiselect('Select .zip files', zips)
        if st.button('Ingest selected'):
            s = Session()
            for z in sel:
                ingest_zip(Path(dp)/z, s)
            s.close()
    else:
        st.error('Invalid path')

tab1, tab2 = st.tabs(['Benchmarking', 'Coverage'])

# Benchmarking Explorer
with tab1:
    st.header('Benchmarking Explorer')
    s = Session()
    runs = s.execute(text(
        "SELECT id, run_label FROM runs "
        "WHERE measurement_type='benchmark' ORDER BY timestamp"
    )).fetchall()
    s.close()
    if not runs:
        st.info('No benchmark runs available.')
        st.stop()
    labels = [r.run_label for r in runs]
    chosen = st.multiselect('Select runs', labels, default=labels)
    if not chosen:
        st.stop()
    ids = [r.id for r in runs if r.run_label in chosen]
    q = ','.join(str(i) for i in ids)

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

    # Filters
    st.sidebar.header('Filters')
    y_axis = st.sidebar.selectbox('Y-axis metric', [c for c in ['time','bus_bw','alg_bw'] if c in df.columns])
    envs = st.sidebar.multiselect('Env bundles', df['env_bundle'].unique(), default=df['env_bundle'].unique())
    ops = st.sidebar.multiselect('Reduction ops', df['redop'].unique(), default=df['redop'].unique())
    types = st.sidebar.multiselect('Datatypes', df['datatype'].unique(), default=df['datatype'].unique())
    inplace = st.sidebar.multiselect('In-place', df['in_place'].unique(), default=df['in_place'].unique())
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
        sub = fdf[fdf['collective']==coll].copy()
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
            fig = px.line(sub, x='size_gb', y=y_axis, color='combo', markers=True)
            fig.update_layout(xaxis_title='Size (GB)', yaxis_title=y_axis)
        st.plotly_chart(fig, use_container_width=True)

# Coverage Explorer
with tab2:
    st.header('Coverage Over Time')
    s = Session()
    runs_cov = s.execute(text(
        "SELECT id, run_label FROM runs WHERE measurement_type='coverage' ORDER BY timestamp"
    )).fetchall()
    s.close()
    if not runs_cov:
        st.info('No coverage runs.')
        st.stop()
    labels_cov = [r.run_label for r in runs_cov]
    chosen_cov = st.multiselect('Coverage runs', labels_cov, default=labels_cov)
    if not chosen_cov:
        st.stop()
    ids_cov = [r.id for r in runs_cov if r.run_label in chosen_cov]
    q2 = ','.join(str(i) for i in ids_cov)

    with engine.connect() as conn:
        cov_data = conn.execute(text(
            f"SELECT * FROM coverage_records WHERE run_id IN ({q2})"
        )).mappings().all()
    dfc = pd.DataFrame(cov_data)
    if dfc.empty:
        st.warning('No coverage data.')
        st.stop()

    # env_bundle already in table
    dfc['run_label'] = pd.Categorical(
        dfc['run_id'].map({r.id: r.run_label for r in runs_cov}),
        categories=chosen_cov, ordered=True
    )
    metrics = ['function_cov','line_cov','region_cov','branch_cov']
    figc = px.line(
        dfc,
        x='run_label',
        y=metrics,
        color='env_bundle',
        markers=True
    )
    figc.update_layout(xaxis_title='Run', yaxis_title='Coverage (%)')
    st.plotly_chart(figc, use_container_width=True)