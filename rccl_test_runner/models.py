import datetime
from sqlalchemy import (
    Column, Integer, Float, BigInteger,
    String, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Run(Base):
    __tablename__ = "runs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_label        = Column(String(255), nullable=False, unique=True)
    measurement_type = Column(String(50),  nullable=False)
    timestamp        = Column(DateTime, default=datetime.datetime.utcnow)

    # ─── Stored total coverage fields ──────────────────────────────────────────
    total_function_count   = Column(Integer)
    total_function_covered = Column(Integer)
    total_function_percent = Column(Float)
    total_line_count       = Column(Integer)
    total_line_covered     = Column(Integer)
    total_line_percent     = Column(Float)
    total_region_count     = Column(Integer)
    total_region_covered   = Column(Integer)
    total_region_percent   = Column(Float)
    total_branch_count     = Column(Integer)
    total_branch_covered   = Column(Integer)
    total_branch_percent   = Column(Float)

    benchmarks = relationship(
        "BenchmarkRecord",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    coverages = relationship(
        "CoverageRecord",
        back_populates="run",
        cascade="all, delete-orphan",
        foreign_keys="[CoverageRecord.run_id]"
    )
    env_vars = relationship(
        "EnvVar",
        back_populates="run",
        cascade="all, delete-orphan"
    )

class BenchmarkRecord(Base):
    __tablename__ = "benchmark_records"

    id             = Column(Integer,    primary_key=True, autoincrement=True)
    run_id         = Column(Integer,    ForeignKey("runs.id"), nullable=False)
    collective     = Column(String(100), nullable=False)
    nodes          = Column(Integer)
    ranks          = Column(Integer)
    ranks_per_node = Column(Integer)
    gpus_per_rank  = Column(Integer)
    size           = Column(BigInteger)
    datatype       = Column(String(50))
    redop          = Column(String(50))
    in_place       = Column(Boolean)
    time           = Column(Float)
    alg_bw         = Column(Float)
    bus_bw         = Column(Float)
    wrong          = Column(String(50))
    env_bundle     = Column(String(255))

    run = relationship("Run", back_populates="benchmarks")

class CoverageRecord(Base):
    __tablename__ = "coverage_records"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_id           = Column(Integer, ForeignKey("runs.id"), nullable=False)
    file             = Column(String(500))

    function_count   = Column(Integer)
    function_covered = Column(Integer)
    function_cov     = Column(Float)

    line_count       = Column(Integer)
    line_covered     = Column(Integer)
    line_cov         = Column(Float)

    region_count     = Column(Integer)
    region_covered   = Column(Integer)
    region_cov       = Column(Float)

    branch_count     = Column(Integer)
    branch_covered   = Column(Integer)
    branch_cov       = Column(Float)

    env_bundle       = Column(String(255))

    run = relationship(
        "Run",
        back_populates="coverages",
        foreign_keys=[run_id]
    )

class EnvVar(Base):
    __tablename__ = "env_vars"

    id     = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    name   = Column(String(255), nullable=False)
    value  = Column(String(255), nullable=True)

    run    = relationship("Run", back_populates="env_vars")