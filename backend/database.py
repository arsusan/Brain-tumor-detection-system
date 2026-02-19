from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# 1. THE CONNECTION URL
DATABASE_URL = "postgresql://postgres:root@localhost:5433/brain_tumor_db"

# 2. CREATE THE ENGINE
engine = create_engine(DATABASE_URL)

# 3. CREATE A SESSION
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. BASE CLASS
Base = declarative_base()

# 5. DEFINE THE USER TABLE (To match User entity in DFD)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False) # From user input
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship to link scans to this user
    scans = relationship("ScanResult", back_populates="owner")

# 6. DEFINE THE SCAN RESULT TABLE (Updated with Foreign Key)
class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    # Link to the User table
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) 
    
    filename = Column(String)
    
    # AI Results (Matches prob distribution from image_6050d9.png)
    prediction = Column(String) # 'glioma', 'meningioma', etc.
    confidence = Column(Float)
    
    # Probability distribution across all 4 classes
    prob_glioma = Column(Float)
    prob_meningioma = Column(Float)
    prob_pituitary = Column(Float)
    prob_notumor = Column(Float)
    
    # Image Paths
    heatmap_url = Column(String) # Path to Grad-CAM heatmap
    original_image_url = Column(String) # Path to original MRI
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship back to the user
    owner = relationship("User", back_populates="scans")

# 7. INITIALIZATION HELPER
def init_db():
    Base.metadata.create_all(bind=engine)

# 8. DEPENDENCY HELPER
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()