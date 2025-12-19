import pytest
from fastapi.testclient import TestClient
from main import app
from database import Base, engine
from sqlalchemy.orm import sessionmaker

# Create test database
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def client():
    """Create test client"""
    Base.metadata.create_all(bind=engine)
    return TestClient(app)


@pytest.fixture
def db_session():
    """Create database session for testing"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_models_empty(client):
    """Test getting models when none exist"""
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == []


def test_train_endpoint_missing_file(client):
    """Test train endpoint with missing file"""
    response = client.post(
        "/api/train",
        data={
            "model_type": "logistic_regression",
            "model_name": "test_model",
            "target_column": "target"
        }
    )
    assert response.status_code == 422  # Validation error


def test_train_endpoint_invalid_model_type(client):
    """Test train endpoint with invalid model type"""
    files = {"file": ("test.csv", b"col1,col2,target\n1,2,0\n3,4,1")}
    data = {
        "model_type": "invalid_type",
        "model_name": "test_model",
        "target_column": "target"
    }
    response = client.post("/api/train", files=files, data=data)
    assert response.status_code == 400


def test_predict_endpoint_invalid_model(client):
    """Test predict endpoint with non-existent model"""
    response = client.post(
        "/api/predict",
        json={
            "model_id": 999,
            "input_data": [1, 2, 3]
        }
    )
    assert response.status_code == 404

