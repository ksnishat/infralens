from sqlalchemy.orm import Session
from src.app.db import models
from src.app import schemas

# 1. READ: Get a tenant by ID
def get_tenant(db: Session, tenant_id: int):
    return db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()

# 2. READ: Get a tenant by Name (to check duplicates)
def get_tenant_by_name(db: Session, name: str):
    return db.query(models.Tenant).filter(models.Tenant.name == name).first()

# 3. CREATE: Make a new tenant
def create_tenant(db: Session, tenant: schemas.TenantCreate):
    # Step A: Create the Database Model
    db_tenant = models.Tenant(name=tenant.name)
    
    # Step B: Add to the "Stage"
    db.add(db_tenant)
    
    # Step C: Commit (Save to Disk)
    db.commit()
    
    # Step D: Refresh (Get the ID and Date back from DB)
    db.refresh(db_tenant)
    
    return db_tenant