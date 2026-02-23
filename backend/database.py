import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES
# Locally, it reads from .env. On Render/Production, it reads from Dashboard Settings.
load_dotenv()

# 2. THE CONNECTION URL
# Fetches from .env. Fallback to your local DB only if DATABASE_URL is missing.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5433/brain_tumor_db")

# FIX: SQLAlchemy requires 'postgresql://' but Supabase often provides 'postgres://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3. CREATE THE ENGINE WITH PRODUCTION POOLING
# pool_pre_ping: vital for cloud DBs to prevent 'connection closed' errors.
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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

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
    
    # Image Paths / URLs
    heatmap_url = Column(String) 
    original_image_url = Column(String) 
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship back to the user
    owner = relationship("User", back_populates="scans")

# 7. INITIALIZATION HELPER
def init_db():
    # Only creates tables that do not already exist in the cloud DB
    Base.metadata.create_all(bind=engine)

# 8. DEPENDENCY HELPER
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()