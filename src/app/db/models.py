from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Tenant(Base):
    """Represents a Company (e.g., 'Avacon Region North')"""
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    api_key = Column(String, unique=True)  # Simple auth for now
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    users = relationship("User", back_populates="tenant")
    assets = relationship("Asset", back_populates="tenant")

class User(Base):
    """A Technician or Manager belonging to a Tenant"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Foreign Key (Links User to a specific Tenant)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    tenant = relationship("Tenant", back_populates="users")

class Asset(Base):
    """An Infrastructure Object (e.g., Power Pole #123)"""
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)  # e.g., "GPS Coordinates"
    image_url = Column(String, nullable=True) # Where we store the uploaded photo
    
    # Foreign Key (Assets belong to a Tenant too!)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    tenant = relationship("Tenant", back_populates="assets")