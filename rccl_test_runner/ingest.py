# ingest.py

import zipfile
import tempfile
import io
import json
from pathlib import Path
from json import JSONDecodeError

import yaml
import jsonlines
from sqlalchemy.orm import Session

from models import (
    Run, BenchmarkRecord,
    CoverageRecord, EnvVar
)

def ingest_zip(zip_input, session: Session):
    tmp = tempfile.TemporaryDirectory()
    extract_dir = tmp.name

    # --- unzip ---
    if isinstance(zip_input, (str, Path)):
        with zipfile.ZipFile(Path(zip_input), 'r') as zf:
            zf.extractall(extract_dir)
        label = Path(zip_input).stem
    else:
        data_bytes = zip_input.read()
        with zipfile.ZipFile(io.BytesIO(data_bytes), 'r') as zf:
            zf.extractall(extract_dir)
        label = getattr(zip_input, "name", "uploaded_run")

    # --- determine type & create run record ---
    perf_paths = list(Path(extract_dir).rglob("*_perf.json"))
    is_bench   = bool(perf_paths)

    run = Run(
        run_label=label,
        measurement_type='benchmark' if is_bench else 'coverage'
        # after you add the fields below to Run:
        # , total_function_count=None,
        #   total_function_covered=None,
        #   total_function_percent=None,
        #   total_line_count=None,
        #   total_line_covered=None,
        #   total_line_percent=None,
        #   total_region_count=None,
        #   total_region_covered=None,
        #   total_region_percent=None,
        #   total_branch_count=None,
        #   total_branch_covered=None,
        #   total_branch_percent=None,
    )
    session.add(run)
    session.flush()  # so run.id is populated

    # --- ingest ENV_VARS ---
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
                val  = v.get('value')
                if name and val is not None:
                    session.add(EnvVar(
                        run_id=run.id,
                        name=name,
                        value=str(val)
                    ))

    if is_bench:
        # --- benchmark ingestion unchanged ---
        for pf in perf_paths:
            bundle = pf.parent.parent.name
            with open(pf, 'r') as f:
                try:
                    data = json.load(f)
                    recs = data if isinstance(data, list) else [data]
                except JSONDecodeError:
                    f.seek(0)
                    recs = list(jsonlines.Reader(f))
            for rec in recs:
                session.add(BenchmarkRecord(
                    run_id=run.id,
                    collective=rec.get("name"),
                    nodes=rec.get("nodes"),
                    ranks=rec.get("ranks"),
                    ranks_per_node=rec.get("ranksPerNode"),
                    gpus_per_rank=rec.get("gpusPerRank"),
                    size=rec.get("size"),
                    datatype=rec.get("type"),
                    redop=rec.get("redop"),
                    in_place=bool(rec.get("inPlace")),
                    time=rec.get("time"),
                    alg_bw=rec.get("algBw"),
                    bus_bw=rec.get("busBw"),
                    wrong=str(rec.get("wrong", "0")),
                    env_bundle=bundle
                ))
    else:
        # --- coverage ingestion, per-file plus run-level totals ---
        for cf in Path(extract_dir).rglob("*.json"):
            if cf.name.endswith("_perf.json"):
                continue
            bundle = cf.parent.parent.name
            cov = json.loads(cf.read_text())
            data = cov.get("data", [])
            if not data:
                raise ValueError("No 'data' array in coverage JSON")
            item = data[0]

            # ingest each file’s coverage
            for fcov in item.get("files", []):
                s     = fcov.get("summary", {}) or {}
                funcs = s.get("functions", {}) or {}
                lines = s.get("lines", {})     or {}
                regs  = s.get("regions", {})   or {}
                brs   = s.get("branches", {})  or {}

                filename = fcov.get("filename") or fcov.get("fileName") or "<unknown>"

                session.add(CoverageRecord(
                    run_id=run.id,
                    file=filename,
                    function_count   = funcs.get("count", 0),
                    function_covered = funcs.get("covered", 0),
                    function_cov     = funcs.get("percent", 0),
                    line_count       = lines.get("count", 0),
                    line_covered     = lines.get("covered", 0),
                    line_cov         = lines.get("percent", 0),
                    region_count     = regs.get("count", 0),
                    region_covered   = regs.get("covered", 0),
                    region_cov       = regs.get("percent", 0),
                    branch_count     = brs.get("count", 0),
                    branch_covered   = brs.get("covered", 0),
                    branch_cov       = brs.get("percent", 0),
                    env_bundle=bundle
                ))

            # pull the run-level totals
            totals = item.get("totals")
            if totals is None:
                raise ValueError("No 'totals' block in coverage JSON")

            # assign them to the Run record
            funcs = totals.get("functions", {}) or {}
            lines = totals.get("lines", {})     or {}
            regs  = totals.get("regions", {})   or {}
            brs   = totals.get("branches", {})  or {}

            run.total_function_count   = funcs.get("count", 0)
            run.total_function_covered = funcs.get("covered", 0)
            run.total_function_percent = funcs.get("percent", 0)

            run.total_line_count       = lines.get("count", 0)
            run.total_line_covered     = lines.get("covered", 0)
            run.total_line_percent     = lines.get("percent", 0)

            run.total_region_count     = regs.get("count", 0)
            run.total_region_covered   = regs.get("covered", 0)
            run.total_region_percent   = regs.get("percent", 0)

            run.total_branch_count     = brs.get("count", 0)
            run.total_branch_covered   = brs.get("covered", 0)
            run.total_branch_percent   = brs.get("percent", 0)

    session.commit()
    tmp.cleanup()


def ingest_coverage_json(json_input, label: str, session: Session):
    # identical coverage-only path
    run = Run(run_label=label, measurement_type='coverage')
    session.add(run)
    session.flush()

    if hasattr(json_input, "read"):
        cov = json.load(json_input)
    else:
        cov = json.loads(Path(json_input).read_text())

    data = cov.get("data", [])
    if not data:
        raise ValueError("No 'data' array in coverage JSON")
    item = data[0]

    for fcov in item.get("files", []):
        s     = fcov.get("summary", {}) or {}
        funcs = s.get("functions", {}) or {}
        lines = s.get("lines", {})     or {}
        regs  = s.get("regions", {})   or {}
        brs   = s.get("branches", {})  or {}

        filename = fcov.get("filename") or fcov.get("fileName") or "<unknown>"

        session.add(CoverageRecord(
            run_id=run.id,
            file=filename,
            function_count   = funcs.get("count", 0),
            function_covered = funcs.get("covered", 0),
            function_cov     = funcs.get("percent", 0),
            line_count       = lines.get("count", 0),
            line_covered     = lines.get("covered", 0),
            line_cov         = lines.get("percent", 0),
            region_count     = regs.get("count", 0),
            region_covered   = regs.get("covered", 0),
            region_cov       = regs.get("percent", 0),
            branch_count     = brs.get("count", 0),
            branch_covered   = brs.get("covered", 0),
            branch_cov       = brs.get("percent", 0),
            env_bundle=label
        ))

    totals = item.get("totals")
    if totals is None:
        raise ValueError("No 'totals' block in coverage JSON")

    funcs = totals.get("functions", {}) or {}
    lines = totals.get("lines", {})     or {}
    regs  = totals.get("regions", {})   or {}
    brs   = totals.get("branches", {})  or {}

    run.total_function_count   = funcs.get("count", 0)
    run.total_function_covered = funcs.get("covered", 0)
    run.total_function_percent = funcs.get("percent", 0)

    run.total_line_count       = lines.get("count", 0)
    run.total_line_covered     = lines.get("covered", 0)
    run.total_line_percent     = lines.get("percent", 0)

    run.total_region_count     = regs.get("count", 0)
    run.total_region_covered   = regs.get("covered", 0)
    run.total_region_percent   = regs.get("percent", 0)

    run.total_branch_count     = brs.get("count", 0)
    run.total_branch_covered   = brs.get("covered", 0)
    run.total_branch_percent   = brs.get("percent", 0)

    session.commit()
