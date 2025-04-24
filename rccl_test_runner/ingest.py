import zipfile
import tempfile
import io
import json
from pathlib import Path
from json import JSONDecodeError

import yaml
import jsonlines
from sqlalchemy.orm import Session

from models import Run, BenchmarkRecord, CoverageRecord, EnvVar

def ingest_zip(zip_input, session: Session):
    """
    Ingests a .zip (benchmark or coverage) into the DB,
    extracts bundle name from second-level directory,
    logs each record with its env_bundle.
    """
    tmp = tempfile.TemporaryDirectory()
    extract_dir = tmp.name

    # unzip
    if isinstance(zip_input, (str, Path)):
        with zipfile.ZipFile(Path(zip_input), 'r') as zf:
            zf.extractall(extract_dir)
        label = Path(zip_input).stem
    else:
        data_bytes = zip_input.read()
        with zipfile.ZipFile(io.BytesIO(data_bytes), 'r') as zf:
            zf.extractall(extract_dir)
        label = getattr(zip_input, "name", "uploaded_run")

    # detect benchmark vs coverage
    perf_paths = list(Path(extract_dir).rglob("*_perf.json"))
    is_bench = bool(perf_paths)

    # create Run entry
    run = Run(
        run_label=label,
        measurement_type='benchmark' if is_bench else 'coverage'
    )
    session.add(run)
    session.flush()

    # ingest ENV_VARS from YAML if needed
    for yaml_file in Path(extract_dir).rglob("*.yaml"):
        try:
            cfg = yaml.safe_load(yaml_file.read_text())
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        for bundle_cfg in cfg.values():
            for v in bundle_cfg.get('ENV_VARS', []):
                name = v.get('id')
                val = v.get('value')
                if name and val is not None:
                    session.add(EnvVar(
                        run_id=run.id,
                        name=name,
                        value=str(val)
                    ))

    # ingest benchmark JSON files
    if is_bench:
        for pf in perf_paths:
            # bundle is second-level dir: extract_dir/.../<bundle>/<timestamp>/file.json
            bundle = pf.parent.parent.name
            with open(pf, 'r') as f:
                try:
                    data = json.load(f)
                    recs = data if isinstance(data, list) else [data]
                except JSONDecodeError:
                    f.seek(0)
                    recs = [obj for obj in jsonlines.Reader(f)]
            for rec in recs:
                session.add(BenchmarkRecord(
                    run_id=run.id,
                    collective=rec.get('name'),
                    nodes=rec.get('nodes'),
                    ranks=rec.get('ranks'),
                    ranks_per_node=rec.get('ranksPerNode'),
                    gpus_per_rank=rec.get('gpusPerRank'),
                    size=rec.get('size'),
                    datatype=rec.get('type'),
                    redop=rec.get('redop'),
                    in_place=bool(rec.get('inPlace')),
                    time=rec.get('time'),
                    alg_bw=rec.get('algBw'),
                    bus_bw=rec.get('busBw'),
                    wrong=str(rec.get('wrong', '0')),
                    env_bundle=bundle
                ))
    else:
        # ingest coverage JSON files
        for cf in Path(extract_dir).rglob("*.json"):
            # skip perf.json if accidentally picked
            if cf.name.endswith('_perf.json'):
                continue
            bundle = cf.parent.parent.name
            cov = json.loads(cf.read_text())
            for fcov in cov.get('data', [])[0].get('files', []):
                session.add(CoverageRecord(
                    run_id=run.id,
                    file=fcov.get('fileName'),
                    function_cov=fcov.get('summary', {}).get('functions', {}).get('percent'),
                    line_cov=fcov.get('summary', {}).get('lines', {}).get('percent'),
                    region_cov=fcov.get('summary', {}).get('regions', {}).get('percent'),
                    branch_cov=fcov.get('summary', {}).get('branches', {}).get('percent'),
                    env_bundle=bundle
                ))

    session.commit()
    tmp.cleanup()
