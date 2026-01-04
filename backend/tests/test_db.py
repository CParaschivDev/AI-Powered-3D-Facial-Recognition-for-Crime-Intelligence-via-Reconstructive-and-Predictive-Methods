import pytest
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base
from backend.database.db_utils import store_identity_embedding, search_identities

TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="module")
def test_db():
    """Fixture to set up and tear down a test database."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    # Monkeypatch the db_utils to use the test DB
    import backend.database.db_utils
    backend.database.db_utils.engine = engine
    backend.database.db_utils.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield

    # Tear down test DB: dispose engine and remove file if possible
    try:
        engine.dispose()
    except Exception:
        pass

    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass
    except PermissionError:
        # If the file is locked by the OS, try again after closing any lingering connections
        try:
            # best effort: call garbage collector and sleep briefly
            import gc, time
            gc.collect()
            time.sleep(0.1)
            os.remove("test.db")
        except Exception:
            # Give up silently; test runner will clean up later
            pass

def test_store_and_search(test_db):
    """Test storing an embedding and then searching for it."""
    identity_id = "test_person_01"
    embedding = np.random.rand(512).astype(np.float64)

    # Store the embedding
    store_identity_embedding(identity_id, embedding)

    # Search for the same embedding
    results = search_identities(embedding, top_n=1)

    assert len(results) == 1
    assert results[0]["identity_id"] == identity_id
    # Confidence should be very close to 1.0
    assert results[0]["confidence"] > 0.999
