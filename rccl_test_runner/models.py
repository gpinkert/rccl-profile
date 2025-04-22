from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Run(Base):
    __tablename__ = 'runs'
    id = Column(Integer, primary_key=True)
    run_label = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    measurement_type = Column(String(50), index=True)  # 'benchmark' or 'coverage'

class BenchmarkRecord(Base):
    __tablename__ = 'benchmark_records'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, index=True)
    collective = Column(String(100))
    redop = Column(String(100))
    datatype = Column(String(100))
    in_place = Column(Integer)
    size = Column(Float)
    size_gb = Column(Float)
    env_bundle = Column(String(500))
    metrics = Column(JSON)  # store all numeric metrics in a JSON column

class CoverageRecord(Base):
    __tablename__ = 'coverage_records'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, index=True)
    function_cov = Column(Float)
    line_cov = Column(Float)
    region_cov = Column(Float)
    branch_cov = Column(Float)
    extra_metrics = Column(JSON)  # additional coverage-related fields
