import os, zipfile, io
from pathlib import Path
import json
from sqlalchemy.orm import Session
from models import Run, BenchmarkRecord, CoverageRecord
from utils import read_json_records


def ingest_zip(source, session: Session):
    """
    Ingest a .zip of benchmark or coverage JSON into the database.
    """
    # Determine buffer and filename
    if hasattr(source, 'read'):
        data = source.read()
        buf = io.BytesIO(data)
        filename = source.name
    else:
        buf = None
        filename = os.path.basename(source)

    with zipfile.ZipFile(buf or source) as zf:
        names = zf.namelist()
        is_benchmark = any(n.endswith('_perf.json') for n in names)
        is_coverage = any(n.lower().endswith('.json') and 'cov' in n.lower() for n in names)

        run = Run(run_label=Path(filename).stem,
                  measurement_type='benchmark' if is_benchmark else 'coverage')
        session.add(run)
        session.commit()

        for member in names:
            raw = zf.read(member)
            if is_benchmark and member.endswith('_perf.json'):
                records = read_json_records(raw)
                collective = Path(member).stem.replace('_perf', '')
                for rec in records:
                    size = float(rec.get('size', 0))
                    br = BenchmarkRecord(
                        run_id=run.id,
                        collective=collective,
                        redop=rec.get('redop') or 'no_reduction',
                        datatype=rec.get('type') or 'unknown',
                        in_place=int(rec.get('inPlace', -1)),
                        size=size,
                        size_gb=size / 1024**3,
                        env_bundle=rec.get('env_bundle', ''),
                        metrics={k: v for k, v in rec.items() if isinstance(v, (int, float))}
                    )
                    session.add(br)
            elif is_coverage and member.lower().endswith('.json'):
                cov = json.loads(raw)
                summary = cov.get('data', [{}])[0].get('totals', {})
                cr = CoverageRecord(
                    run_id=run.id,
                    function_cov=summary.get('functions', {}).get('percent'),
                    line_cov=summary.get('lines', {}).get('percent'),
                    region_cov=summary.get('regions', {}).get('percent'),
                    branch_cov=summary.get('branches', {}).get('percent'),
                    extra_metrics={k: v for k, v in summary.items() if k not in {'functions', 'lines', 'regions', 'branches'}}
                )
                session.add(cr)
        session.commit()
