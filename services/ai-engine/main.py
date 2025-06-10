import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uvicorn
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
import joblib
import uuid
import time
from pathlib import Path

from app.models import (
    ModelType, TrainingConfig, TrainingResponse,
    PredictionRequest, PredictionResponse,
    ModelMetadata, ModelEvaluation
)

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/ai_engine_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AIEngine')

app = FastAPI(title="AI Engine API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurazione InfluxDB
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "mkt_sim")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "market_data")

influxdb_client = InfluxDBClient(
    url=INFLUXDB_URL,
    token=INFLUXDB_TOKEN,
    org=INFLUXDB_ORG
)

# Directory per i modelli
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Cache dei modelli
model_cache: Dict[str, any] = {}

@app.post("/train", response_model=TrainingResponse)
async def train_model(config: TrainingConfig):
    """Addestra un nuovo modello di machine learning."""
    try:
        start_time = time.time()
        model_id = str(uuid.uuid4())
        
        # Recupera i dati di training da InfluxDB
        data = {}
        for symbol in config.symbols:
            query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {config.start_date.isoformat()}, stop: {config.end_date.isoformat()})
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
            '''
            
            result = influxdb_client.query_api().query(query)
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Dati non trovati per {symbol}")
            
            # Converti i dati in DataFrame
            records = []
            for table in result:
                for record in table.records:
                    records.append({
                        'date': record.get_time(),
                        'open': record.values.get('open'),
                        'high': record.values.get('high'),
                        'low': record.values.get('low'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume')
                    })
            
            data[symbol] = pd.DataFrame(records)
        
        # TODO: Implementare la logica di training specifica per ogni tipo di modello
        # Per ora restituiamo una risposta di esempio
        training_time = time.time() - start_time
        
        return TrainingResponse(
            success=True,
            message="Training completato con successo",
            model_id=model_id,
            metrics={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            },
            training_time=training_time
        )
    
    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Effettua predizioni usando un modello addestrato."""
    try:
        # Carica il modello se non è in cache
        if request.model_id not in model_cache:
            model_path = MODELS_DIR / f"{request.model_id}.joblib"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Modello non trovato")
            
            model_cache[request.model_id] = joblib.load(model_path)
        
        model = model_cache[request.model_id]
        
        # Prepara i dati per la predizione
        df = pd.DataFrame(request.data)
        
        # TODO: Implementare la logica di predizione specifica per ogni tipo di modello
        # Per ora restituiamo una risposta di esempio
        return PredictionResponse(
            success=True,
            message="Predizione completata con successo",
            predictions=[0.75, 0.82, 0.68],
            confidence=[0.85, 0.92, 0.78],
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Errore durante la predizione: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}", response_model=ModelMetadata)
async def get_model_metadata(model_id: str):
    """Recupera i metadati di un modello."""
    try:
        model_path = MODELS_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Modello non trovato")
        
        # TODO: Implementare il recupero dei metadati del modello
        # Per ora restituiamo una risposta di esempio
        return ModelMetadata(
            model_id=model_id,
            model_type=ModelType.LSTM,
            symbol="AAPL",
            created_at=datetime.now(),
            metrics={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            },
            parameters={
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "sequence_length": 60
            }
        )
    
    except Exception as e:
        logger.error(f"Errore nel recupero dei metadati del modello: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/{model_id}", response_model=ModelEvaluation)
async def evaluate_model(model_id: str):
    """Valuta le performance di un modello."""
    try:
        model_path = MODELS_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Modello non trovato")
        
        # TODO: Implementare la logica di valutazione del modello
        # Per ora restituiamo una risposta di esempio
        return ModelEvaluation(
            model_id=model_id,
            symbol="AAPL",
            evaluation_date=datetime.now(),
            metrics={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            },
            confusion_matrix=[[100, 20], [15, 85]],
            feature_importance={
                "open": 0.25,
                "high": 0.20,
                "low": 0.20,
                "close": 0.25,
                "volume": 0.10
            }
        )
    
    except Exception as e:
        logger.error(f"Errore nella valutazione del modello: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True) 