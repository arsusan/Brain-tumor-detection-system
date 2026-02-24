import os
import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from dotenv import load_dotenv

Base = declarative_base()
# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()

# 2. THE CONNECTION URL
# Defaults to your local Postgres, but allows GitHub Actions to inject 'sqlite:///:memory:'
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5433/brain_tumor_db")

# FIX: SQLAlchemy requires 'postgresql://' but some providers still use the old 'postgres://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3. CREATE THE ENGINE WITH SMART SWITCHING
# We check the URL type to avoid passing Postgres-only arguments to SQLite
# database.py
if "sqlite" in DATABASE_URL:
    # SQLite does NOT support pooling (pool_size/max_overflow)
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL DOES support/need pooling
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

# 4. SESSION AND BASE
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 5. DEFINE THE USER TABLE
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to link scans to this user
    scans = relationship("ScanResult", back_populates="owner")

# 6. DEFINE THE SCAN RESULT TABLE
class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) 
    
    filename = Column(String)
    
    # AI Results
    prediction = Column(String) # 'glioma', 'meningioma', etc.
    confidence = Column(Float)
    
    # Probability distribution
    prob_glioma = Column(Float)
    prob_meningioma = Column(Float)
    prob_pituitary = Column(Float)
    prob_notumor = Column(Float)
    
    # Image Paths / URLs (Cloudinary)
    heatmap_url = Column(String) 
    original_image_url = Column(String) 
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship back to the user
    owner = relationship("User", back_populates="scans")

# 7. INITIALIZATION HELPER
def init_db():
    # Only creates tables that do not already exist
    Base.metadata.create_all(bind=engine)

# 8. DEPENDENCY HELPER
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()