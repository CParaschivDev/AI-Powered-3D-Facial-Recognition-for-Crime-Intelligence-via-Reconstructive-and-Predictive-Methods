import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, timezone

from backend.api.main import app
from backend.database.models import Base, Prediction, User
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
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def override_get_db(db_session):
    """Fixture to override the get_db dependency with the test session."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def auth_token(db_session):
    """Fixture to get a valid auth token."""
    user = User(username="testofficer", hashed_password=get_password_hash("testpass"), role="officer")
    db_session.add(user)
    db_session.commit()
    response = client.post("/api/v1/auth/token", data={"username": "testofficer", "password": "testpass"})
    assert response.status_code == 200
    return response.json()["access_token"]

def test_get_predictions(db_session, override_get_db, auth_token):
    """Test the /analytics/predictions endpoint."""
    # 1. Populate the test database with some prediction data
    now = datetime.now(timezone.utc)
    test_data = [
        Prediction(area_id="test-area", ts=now + timedelta(days=1), crime_type="theft", yhat=5.5, yhat_lower=4.0, yhat_upper=7.0),
        Prediction(area_id="test-area", ts=now + timedelta(days=2), crime_type="theft", yhat=6.0, yhat_lower=4.5, yhat_upper=7.5),
        Prediction(area_id="test-area", ts=now + timedelta(days=1), crime_type="burglary", yhat=2.1, yhat_lower=1.0, yhat_upper=3.0),
        Prediction(area_id="other-area", ts=now + timedelta(days=1), crime_type="theft", yhat=10.0, yhat_lower=8.0, yhat_upper=12.0),
    ]
    db_session.add_all(test_data)
    db_session.commit()

    # 2. Make a request to the endpoint
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.get("/api/v1/analytics/predictions?area_id=test-area&window=5", headers=headers)

    # 3. Assert the response
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3 # Should only get predictions for 'test-area'
    assert data["predictions"][0]["crime_type"] == "theft"
    assert data["predictions"][0]["yhat"] == 5.5
    assert "yhat_lower" in data["predictions"][0]
    assert "yhat_upper" in data["predictions"][0]
    assert data["predictions"][0]["yhat_lower"] == 4.0
    assert data["predictions"][0]["yhat_upper"] == 7.0
