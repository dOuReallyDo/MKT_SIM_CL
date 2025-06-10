from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"

class TrainingConfig(BaseModel):
    model_type: ModelType
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 0.001
    sequence_length: int = 60
    features: List[str] = ["open", "high", "low", "close", "volume"]

class TrainingResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    training_time: Optional[float] = None

class PredictionRequest(BaseModel):
    model_id: str
    symbol: str
    data: List[Dict[str, float]]
    sequence_length: int = 60

class PredictionResponse(BaseModel):
    success: bool
    message: str
    predictions: Optional[List[float]] = None
    confidence: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelMetadata(BaseModel):
    model_id: str
    model_type: ModelType
    symbol: str
    created_at: datetime
    metrics: Dict[str, float]
    parameters: Dict[str, Union[str, int, float, bool]]
    version: str = "1.0.0"

class ModelEvaluation(BaseModel):
    model_id: str
    symbol: str
    evaluation_date: datetime
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None 