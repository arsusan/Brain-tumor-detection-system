from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. THE CONNECTION URL
# Format: postgresql://[USER]:[PASSWORD]@[HOST]:[PORT]/[DATABASE_NAME]
# CHANGE 'yourpassword' to the password you set during Postgres installation!
DATABASE_URL = "postgresql://postgres:root@localhost:5433/brain_tumor_db"

# 2. CREATE THE ENGINE (The actual connection tool)
engine = create_engine(DATABASE_URL)

# 3. CREATE A SESSION (The factory for individual database tasks)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. BASE CLASS (All our tables will inherit from this)
Base = declarative_base()

# 5. DEFINE THE SCAN TABLE (This is the "Excel" sheet structure)
class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True) # Unique ID for every row
    filename = Column(String)                         # Name of MRI file
    prediction = Column(String)                       # "Tumor" or "No Tumor"
    confidence = Column(Float)                        # e.g., 0.98
    heatmap_url = Column(String)                      # Path to the saved heatmap image
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 6. INITIALIZATION HELPER
def init_db():
    # This automatically creates the "scans" table in Postgres if it doesn't exist
    Base.metadata.create_all(bind=engine)

# 7. DEPENDENCY HELPER
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()