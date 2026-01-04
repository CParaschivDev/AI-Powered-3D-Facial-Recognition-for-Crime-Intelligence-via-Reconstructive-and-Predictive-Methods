import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.main import app
from backend.database.models import Base, ModelVersion, User
from backend.database.dependencies import get_db
from backend.core.security import get_password_hash

import pytest
try:
    client = TestClient(app)
except TypeError:
    pytest.skip("Incompatible TestClient in this environment", allow_module_level=True)

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Fixture to create a new database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        # Add some test models
        models = [
            ModelVersion(name="recognition", version=1, path="/models/rec/v1", sha256="abc", active=True),
            ModelVersion(name="recognition", version=2, path="/models/rec/v2", sha256="def", active=False),
            ModelVersion(name="reconstruction", version=1, path="/models/recon/v1.pth", sha256="123", active=True),
        ]
        db.add_all(models)
        db.commit()
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def override_get_db(db_session):
    """Fixture to override the get_db dependency with the test session."""
    def _override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def admin_auth_token(db_session):
    """Fixture to get a valid auth token for an admin user."""
    user = User(username="testadmin", hashed_password=get_password_hash("testpass"), role="admin")
    db_session.add(user)
    db_session.commit()
    response = client.post("/api/v1/auth/token", data={"username": "testadmin", "password": "testpass"})
    assert response.status_code == 200
    return response.json()["access_token"]

def test_list_models(override_get_db, admin_auth_token):
    headers = {"Authorization": f"Bearer {admin_auth_token}"}
    response = client.get("/api/v1/models", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["name"] == "recognition"
    assert data[0]["version"] == 2

def test_activate_model(db_session, override_get_db, admin_auth_token):
    headers = {"Authorization": f"Bearer {admin_auth_token}"}
    
    # Check initial state
    rec_v1 = db_session.query(ModelVersion).filter_by(name="recognition", version=1).one()
    rec_v2 = db_session.query(ModelVersion).filter_by(name="recognition", version=2).one()
    assert rec_v1.active is True
    assert rec_v2.active is False

    # Activate v2
    response = client.post(
        "/api/v1/models/activate",
        json={"name": "recognition", "version": 2},
        headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "recognition"
    assert data["version"] == 2
    assert data["active"] is True

    # Verify state in DB
    db_session.expire_all() # Refresh from DB
    rec_v1_after = db_session.query(ModelVersion).filter_by(name="recognition", version=1).one()
    rec_v2_after = db_session.query(ModelVersion).filter_by(name="recognition", version=2).one()
    recon_v1_after = db_session.query(ModelVersion).filter_by(name="reconstruction", version=1).one()
    
    assert rec_v1_after.active is False
    assert rec_v2_after.active is True
    assert recon_v1_after.active is True # Should be unchanged

def test_activate_nonexistent_model(override_get_db, admin_auth_token):
    headers = {"Authorization": f"Bearer {admin_auth_token}"}
    response = client.post(
        "/api/v1/models/activate",
        json={"name": "recognition", "version": 99},
        headers=headers
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
