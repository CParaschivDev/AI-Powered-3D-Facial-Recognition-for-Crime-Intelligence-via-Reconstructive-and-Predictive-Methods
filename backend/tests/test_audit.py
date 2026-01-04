import pytest
import uuid
import json
import hashlib
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.audit import write_event
from backend.database.models import Base, User, AuditLog

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Fixture to create a new database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def test_user(db_session):
    """Fixture to create a test user."""
    user = User(id=1, username="test_investigator", hashed_password="...", role="investigator", disabled=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

def verify_chain(db_session, case_id: str) -> bool:
    """
    Verifies the integrity of the audit chain for a given case.
    """
    logs = db_session.query(AuditLog).filter(AuditLog.case_id == case_id).order_by(AuditLog.created_at.asc()).all()
    
    if not logs:
        return True # An empty chain is valid

    # Check genesis block
    if logs[0].prev_hash != "genesis":
        print(f"Genesis block has invalid prev_hash: {logs[0].prev_hash}")
        return False

    # Check subsequent blocks
    for i in range(len(logs)):
        entry = logs[i]
        prev_hash = logs[i-1].hash if i > 0 else "genesis"

        # Verify prev_hash link
        if entry.prev_hash != prev_hash:
            print(f"Chain broken at entry ID {entry.id}: prev_hash mismatch.")
            return False

        # Verify content hash
        hash_content = (
            f"{entry.prev_hash}{entry.payload_json}{entry.file_hash or ''}{entry.created_at.isoformat()}"
        ).encode("utf-8")
        recalculated_hash = hashlib.sha256(hash_content).hexdigest()

        if entry.hash != recalculated_hash:
            print(f"Chain broken at entry ID {entry.id}: content hash mismatch.")
            return False
            
    return True


def test_audit_chain_integrity(db_session, test_user):
    """
    Tests that a correctly generated audit chain is valid.
    """
    case_id = str(uuid.uuid4())
    
    write_event(db_session, case_id, test_user, "UPLOAD", {"file": "image1.jpg"}, "hash1")
    write_event(db_session, case_id, test_user, "RECONSTRUCTION", {"result": "3dmodel1"}, "hash1")
    write_event(db_session, case_id, test_user, "RECOGNITION", {"match": "person_A"}, "hash1")

    assert verify_chain(db_session, case_id) is True

def test_audit_chain_tampering_invalidates_chain(db_session, test_user):
    """
    Tests that tampering with a record in the audit chain invalidates it.
    """
    case_id = str(uuid.uuid4())
    
    write_event(db_session, case_id, test_user, "UPLOAD", {"file": "image1.jpg"}, "hash1")
    log_to_tamper = write_event(db_session, case_id, test_user, "RECONSTRUCTION", {"result": "3dmodel1"}, "hash1")
    write_event(db_session, case_id, test_user, "RECOGNITION", {"match": "person_A"}, "hash1")

    assert verify_chain(db_session, case_id) is True

    log_to_tamper.payload_json = json.dumps({"result": "tampered_model"})
    db_session.commit()

    assert verify_chain(db_session, case_id) is False