# reset_schema.py

from sqlalchemy import create_engine
from models import Base

# Change this URL if you’re not on localhost:
DB_URL = "mysql+pymysql://dashuser:dashpass@127.0.0.1/perf_dashboard"
# Or for SQLite: DB_URL = "sqlite:///runs.db"

engine = create_engine(DB_URL)
Base.metadata.create_all(engine)
print("Schema created.")
