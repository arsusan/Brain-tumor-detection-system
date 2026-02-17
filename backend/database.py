from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. THE CONNECTION URL
# Keep your port 5433 and password 'root' as per your setup
DATABASE_URL = "postgresql://postgres:root@localhost:5433/brain_tumor_db"

# 2. CREATE THE ENGINE
engine = create_engine(DATABASE_URL)

# 3. CREATE A SESSION
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. BASE CLASS
Base = declarative_base()

# 5. DEFINE THE UPDATED SCAN TABLE (Matching the 4-Class Categorization)
class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    
    # This will now store: 'glioma', 'meningioma', 'pituitary', or 'notumor'
    prediction = Column(String) 
    
    # The highest probability score
    confidence = Column(Float)
    
    # NEW: Individual probabilities for the 4-class distribution
    # This matches your project synopsis requirements for categorization
    prob_glioma = Column(Float)
    prob_meningioma = Column(Float)
    prob_pituitary = Column(Float)
    prob_notumor = Column(Float)
    
    heatmap_url = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 6. INITIALIZATION HELPER
def init_db():
    # This will create the table with the new columns
    Base.metadata.create_all(bind=engine)

# 7. DEPENDENCY HELPER
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()