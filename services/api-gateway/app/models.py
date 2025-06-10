from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"

class User(BaseModel):
    id: str
    email: EmailStr
    username: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.USER

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class TokenData(BaseModel):
    user_id: str
    role: UserRole

class SimulationRequest(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    strategy: str
    num_agents: int = 5

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    created_at: datetime
    results: Optional[Dict] = None
    error: Optional[str] = None

class ModelTrainingRequest(BaseModel):
    model_type: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, any]

class ModelTrainingResponse(BaseModel):
    training_id: str
    status: str
    created_at: datetime
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class DataCollectionRequest(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    force_download: bool = False

class DataCollectionResponse(BaseModel):
    collection_id: str
    status: str
    created_at: datetime
    data: Optional[Dict] = None
    error: Optional[str] = None 