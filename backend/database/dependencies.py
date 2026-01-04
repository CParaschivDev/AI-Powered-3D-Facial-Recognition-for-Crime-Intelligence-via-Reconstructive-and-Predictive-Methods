from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from backend.core.config import settings

# Create engine and session maker
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency to get a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()