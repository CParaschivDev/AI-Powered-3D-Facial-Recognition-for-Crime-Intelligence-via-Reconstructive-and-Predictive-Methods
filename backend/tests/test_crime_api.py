import pytest
from fastapi.testclient import TestClient
import polars as pl
from unittest.mock import patch

from backend.api.main import app
from backend.core.dependencies import get_current_user
from backend.api.models import schemas


@pytest.fixture
def test_client():
    # Clear overrides before yielding the client
    app.dependency_overrides.clear()
    # Some environments have an httpx/TestClient mismatch; skip tests if creation fails
    try:
        client = TestClient(app)
    except TypeError as e:
        import pytest as _pytest
        _pytest.skip(f"Incompatible TestClient/httpx in environment: {e}")
        return
    yield client
    # Clean up after the test
    app.dependency_overrides.clear()


@pytest.fixture
def mock_crime_df():
    """Provides a mock Polars DataFrame for crime data."""
    data = {
        "Month": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01", "2023-02-01"],
        "Falls within": ["Force A", "Force B", "Force A", "Force A", "Force B"],
        "LSOA name": ["LSOA_A1", "LSOA_B1", "LSOA_A1", "LSOA_A2", "LSOA_B1"],
        "Crime type": ["Theft", "Burglary", "Theft", "Assault", "Theft"],
        "Latitude": [51.5, 52.5, 51.5, 51.6, 52.5],
        "Longitude": [-0.1, -0.2, -0.1, -0.15, -0.2],
    }
    df = pl.DataFrame(data).with_columns(pl.col("Month").str.to_date("%Y-%m-%d"))
    return df


def mock_officer():
    return schemas.User(username="test_officer", role="officer")


@patch("backend.services.crime_service.get_crime_dataframe")
def test_get_monthly_trends(mock_get_df, test_client, mock_crime_df):
    app.dependency_overrides[get_current_user] = mock_officer
    mock_get_df.return_value = mock_crime_df

    response = test_client.get("/api/v1/crime/forces/monthly?from=2023-01&to=2023-02")

    assert response.status_code == 200
    data = response.json()

    # Expected: Force A (Jan, Feb), Force B (Jan, Feb)
    assert len(data) == 4

    force_a_jan = next(item for item in data if item["Falls within"] == "Force A" and item["Month"] == "2023-01-01")
    assert force_a_jan["crime_count"] == 1

    force_a_feb = next(item for item in data if item["Falls within"] == "Force A" and item["Month"] == "2023-02-01")
    assert force_a_feb["crime_count"] == 2


@patch("backend.services.crime_service.get_crime_dataframe")
def test_get_latest_hotspots(mock_get_df, test_client, mock_crime_df):
    app.dependency_overrides[get_current_user] = mock_officer
    mock_get_df.return_value = mock_crime_df

    response = test_client.get("/api/v1/crime/hotspots/latest?force=Force A")

    assert response.status_code == 200
    data = response.json()

    # Latest month is Feb. Force A has two LSOAs with 1 crime each.
    assert len(data) == 2
    assert {d["LSOA name"] for d in data} == {"LSOA_A1", "LSOA_A2"}
    assert all(d["crime_count"] == 1 for d in data)


@patch("backend.services.crime_service.get_crime_dataframe")
def test_get_lsoa_series(mock_get_df, test_client, mock_crime_df):
    app.dependency_overrides[get_current_user] = mock_officer
    mock_get_df.return_value = mock_crime_df

    response = test_client.get("/api/v1/crime/lsoa/series?lsoa=LSOA_B1")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    jan_data = next(item for item in data if item["Month"] == "2023-01-01")
    assert jan_data["Crime type"] == "Burglary"
    assert jan_data["crime_count"] == 1

    feb_data = next(item for item in data if item["Month"] == "2023-02-01")
    assert feb_data["Crime type"] == "Theft"
    assert feb_data["crime_count"] == 1


def test_unauthorized_access(test_client):
    # No dependency override, so no user is authenticated
    response = test_client.get("/api/v1/crime/forces/monthly?from=2023-01&to=2023-02")
    assert response.status_code == 401


def test_bad_date_format(test_client):
    app.dependency_overrides[get_current_user] = mock_officer
    response = test_client.get("/api/v1/crime/forces/monthly?from=2023/01&to=2023-02")
    assert response.status_code == 400
    assert "Invalid date format" in response.json()["detail"]