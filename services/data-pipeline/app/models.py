from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class StockData(BaseModel):
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class DataCollectionRequest(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    force_download: bool = False

class DataCollectionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, List[StockData]]] = None
    errors: Optional[Dict[str, str]] = None

class DataValidationRequest(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime

class DataValidationResponse(BaseModel):
    valid: bool
    missing_data: Dict[str, List[datetime]]
    invalid_data: Dict[str, List[datetime]]
    message: str

class DataCleanupRequest(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    remove_outliers: bool = True
    fill_missing: bool = True

class DataCleanupResponse(BaseModel):
    success: bool
    message: str
    cleaned_data: Optional[Dict[str, List[StockData]]] = None
    statistics: Optional[Dict[str, Dict[str, float]]] = None 