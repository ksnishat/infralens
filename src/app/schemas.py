from pydantic import BaseModel
from datetime import datetime

# 1. Base Schema (Shared properties)
# This is the "Bun and Meat" that every burger has.
class TenantBase(BaseModel):
    name: str  # <--- Changed "string" to "str" (Python type)

# 2. Create Schema (Input)
# This is what the Customer sees on the menu.
# We explicitly define this class so other files can find it!
class TenantCreate(TenantBase):
    pass 

# 3. Response Schema (Output)
# This is the "Cooked Burger" with the wrapper (ID) and receipt (Date).
class Tenant(TenantBase):
    id: int
    created_at: datetime

    # This tells Pydantic to read data even if it comes from a Database model
    class Config:
        from_attributes = True