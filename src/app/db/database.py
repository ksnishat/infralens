from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Connection String (using the credentials from docker-compose)
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@db/infralens"

# Create the Engine (The connection pool)
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create the SessionLocal (Each request gets its own session)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# The Base class for our models
Base = declarative_base()

# Helper function to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()