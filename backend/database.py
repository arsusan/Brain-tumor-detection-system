from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. THE CONNECTION URL
DATABASE_URL = "postgresql://postgres:root@localhost:5433/brain_tumor_db"

# 2. CREATE THE ENGINE
engine = create_engine(DATABASE_URL)

# 3. CREATE A SESSION
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. BASE CLASS
Base = declarative_base()

# 5. DEFINE THE UPDATED SCAN TABLE
class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    
    # Stores: 'glioma', 'meningioma', 'pituitary', or 'notumor'
    prediction = Column(String) 
    confidence = Column(Float)
    
    # Probability distribution across all 4 classes
    prob_glioma = Column(Float)
    prob_meningioma = Column(Float)
    prob_pituitary = Column(Float)
    prob_notumor = Column(Float)
    
    # Path to the unique Grad-CAM image (e.g., /plots/scan_123_heatmap.png)
    heatmap_url = Column(String)
    
    # Path to the original MRI scan
    original_image_url = Column(String)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 6. INITIALIZATION HELPER
def init_db():
    Base.metadata.create_all(bind=engine)

# 7. DEPENDENCY HELPER
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()