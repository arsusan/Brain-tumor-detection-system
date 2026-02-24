import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from backend.database import Base, get_db
from backend.main import app

# Use StaticPool to keep the in-memory database alive across multiple connections
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool, 
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # Create the tables once
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
            
    app.dependency_overrides[get_db] = override_get_db
    yield
    # Tables are dropped after all tests in the session finish
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    """Provides a clean session for each test and handles rollbacks."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()