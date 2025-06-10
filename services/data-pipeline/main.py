import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from app.models import (
    StockData, DataCollectionRequest, DataCollectionResponse,
    DataValidationRequest, DataValidationResponse,
    DataCleanupRequest, DataCleanupResponse
)

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/data_pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataPipeline')

app = FastAPI(title="Data Pipeline API")

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

write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)

@app.post("/data/collect", response_model=DataCollectionResponse)
async def collect_data(request: DataCollectionRequest):
    """Raccoglie i dati storici per i simboli specificati."""
    try:
        data = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                # Scarica i dati usando yfinance
                stock = yf.Ticker(symbol)
                df = stock.history(
                    start=request.start_date,
                    end=request.end_date,
                    auto_adjust=True
                )
                
                if df.empty:
                    errors[symbol] = "Nessun dato disponibile"
                    continue
                
                # Converti i dati nel formato richiesto
                stock_data = []
                for index, row in df.iterrows():
                    stock_data.append(StockData(
                        symbol=symbol,
                        date=index,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']),
                        adjusted_close=float(row['Close'])  # yfinance già fornisce i prezzi aggiustati
                    ))
                
                data[symbol] = stock_data
                
                # Salva i dati in InfluxDB
                for point in stock_data:
                    p = Point("stock_data") \
                        .tag("symbol", symbol) \
                        .field("open", point.open) \
                        .field("high", point.high) \
                        .field("low", point.low) \
                        .field("close", point.close) \
                        .field("volume", point.volume) \
                        .field("adjusted_close", point.adjusted_close) \
                        .time(point.date)
                    write_api.write(bucket=INFLUXDB_BUCKET, record=p)
                
            except Exception as e:
                errors[symbol] = str(e)
                logger.error(f"Errore nel download dei dati per {symbol}: {e}")
        
        return DataCollectionResponse(
            success=len(errors) == 0,
            message="Raccolta dati completata",
            data=data if data else None,
            errors=errors if errors else None
        )
    
    except Exception as e:
        logger.error(f"Errore nella raccolta dei dati: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/validate", response_model=DataValidationResponse)
async def validate_data(request: DataValidationRequest):
    """Valida i dati per i simboli specificati."""
    try:
        missing_data = {}
        invalid_data = {}
        
        for symbol in request.symbols:
            try:
                # Recupera i dati da InfluxDB
                query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                    |> range(start: {request.start_date.isoformat()}, stop: {request.end_date.isoformat()})
                    |> filter(fn: (r) => r["symbol"] == "{symbol}")
                '''
                
                result = influxdb_client.query_api().query(query)
                
                if not result:
                    missing_data[symbol] = [request.start_date, request.end_date]
                    continue
                
                # Verifica la validità dei dati
                dates = set()
                invalid_dates = []
                
                for table in result:
                    for record in table.records:
                        date = record.get_time()
                        dates.add(date)
                        
                        # Verifica la validità dei valori
                        if any(v is None or np.isnan(v) for v in [
                            record.get_value(),
                            record.values.get('open'),
                            record.values.get('high'),
                            record.values.get('low'),
                            record.values.get('close'),
                            record.values.get('volume')
                        ]):
                            invalid_dates.append(date)
                
                # Verifica le date mancanti
                expected_dates = pd.date_range(
                    start=request.start_date,
                    end=request.end_date,
                    freq='B'
                )
                missing_dates = [d for d in expected_dates if d not in dates]
                
                if missing_dates:
                    missing_data[symbol] = missing_dates
                if invalid_dates:
                    invalid_data[symbol] = invalid_dates
                
            except Exception as e:
                logger.error(f"Errore nella validazione dei dati per {symbol}: {e}")
                missing_data[symbol] = [request.start_date, request.end_date]
        
        return DataValidationResponse(
            valid=len(missing_data) == 0 and len(invalid_data) == 0,
            missing_data=missing_data,
            invalid_data=invalid_data,
            message="Validazione completata"
        )
    
    except Exception as e:
        logger.error(f"Errore nella validazione dei dati: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/cleanup", response_model=DataCleanupResponse)
async def cleanup_data(request: DataCleanupRequest):
    """Pulisce i dati per i simboli specificati."""
    try:
        cleaned_data = {}
        statistics = {}
        
        for symbol in request.symbols:
            try:
                # Recupera i dati da InfluxDB
                query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                    |> range(start: {request.start_date.isoformat()}, stop: {request.end_date.isoformat()})
                    |> filter(fn: (r) => r["symbol"] == "{symbol}")
                '''
                
                result = influxdb_client.query_api().query(query)
                
                if not result:
                    continue
                
                # Converti i dati in DataFrame
                data = []
                for table in result:
                    for record in table.records:
                        data.append({
                            'date': record.get_time(),
                            'open': record.values.get('open'),
                            'high': record.values.get('high'),
                            'low': record.values.get('low'),
                            'close': record.values.get('close'),
                            'volume': record.values.get('volume'),
                            'adjusted_close': record.values.get('adjusted_close')
                        })
                
                df = pd.DataFrame(data)
                
                # Rimuovi gli outlier se richiesto
                if request.remove_outliers:
                    for col in ['open', 'high', 'low', 'close']:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
                
                # Riempie i valori mancanti se richiesto
                if request.fill_missing:
                    df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Calcola le statistiche
                stats = {
                    'mean': df['close'].mean(),
                    'std': df['close'].std(),
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'missing_values': df.isnull().sum().to_dict()
                }
                
                # Converti i dati nel formato richiesto
                stock_data = []
                for _, row in df.iterrows():
                    stock_data.append(StockData(
                        symbol=symbol,
                        date=row['date'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']),
                        adjusted_close=float(row['adjusted_close'])
                    ))
                
                cleaned_data[symbol] = stock_data
                statistics[symbol] = stats
                
                # Aggiorna i dati in InfluxDB
                for point in stock_data:
                    p = Point("stock_data") \
                        .tag("symbol", symbol) \
                        .field("open", point.open) \
                        .field("high", point.high) \
                        .field("low", point.low) \
                        .field("close", point.close) \
                        .field("volume", point.volume) \
                        .field("adjusted_close", point.adjusted_close) \
                        .time(point.date)
                    write_api.write(bucket=INFLUXDB_BUCKET, record=p)
                
            except Exception as e:
                logger.error(f"Errore nella pulizia dei dati per {symbol}: {e}")
                continue
        
        return DataCleanupResponse(
            success=len(cleaned_data) > 0,
            message="Pulizia dati completata",
            cleaned_data=cleaned_data if cleaned_data else None,
            statistics=statistics if statistics else None
        )
    
    except Exception as e:
        logger.error(f"Errore nella pulizia dei dati: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True) 