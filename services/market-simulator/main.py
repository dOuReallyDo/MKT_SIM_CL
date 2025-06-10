import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn

from app.models import SimulationConfig, SimulationResult, MarketData, Transaction, AgentPerformance
from app.market_environment import MarketEnvironment

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/market_simulator_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MarketSimulator')

app = FastAPI(title="Market Simulator API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stato globale
market_environment = None
current_simulation = None

@app.post("/simulation/start", response_model=SimulationResult)
async def start_simulation(config: SimulationConfig):
    """Avvia una nuova simulazione di mercato."""
    global market_environment, current_simulation
    
    try:
        # Inizializza l'ambiente di mercato
        market_environment = MarketEnvironment(
            data={},  # I dati verranno caricati dal data-pipeline
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        # TODO: Implementare la logica di simulazione
        # Per ora restituiamo un risultato di esempio
        return SimulationResult(
            transactions=[],
            agent_performances=[],
            market_data={},
            summary={
                "total_transactions": 0,
                "total_value": 0.0,
                "success_rate": 0.0
            }
        )
    except Exception as e:
        logger.error(f"Errore nell'avvio della simulazione: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/data/{date}", response_model=Dict[str, MarketData])
async def get_market_data(date: str):
    """Recupera i dati di mercato per una data specifica."""
    if not market_environment:
        raise HTTPException(status_code=400, detail="Simulazione non inizializzata")
    
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        return market_environment.get_market_data(date_obj)
    except Exception as e:
        logger.error(f"Errore nel recupero dei dati di mercato: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/price/{symbol}", response_model=float)
async def get_current_price(symbol: str):
    """Recupera il prezzo corrente per un simbolo."""
    if not market_environment:
        raise HTTPException(status_code=400, detail="Simulazione non inizializzata")
    
    try:
        price = market_environment.get_current_price(symbol)
        if price is None:
            raise HTTPException(status_code=404, detail=f"Prezzo non trovato per {symbol}")
        return price
    except Exception as e:
        logger.error(f"Errore nel recupero del prezzo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transaction", response_model=Transaction)
async def execute_transaction(
    agent_id: str,
    symbol: str,
    action: str,
    quantity: int,
    price: float
):
    """Esegue una transazione di trading."""
    if not market_environment:
        raise HTTPException(status_code=400, detail="Simulazione non inizializzata")
    
    try:
        transaction = market_environment.execute_transaction(
            agent_id=agent_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price
        )
        if transaction is None:
            raise HTTPException(status_code=400, detail="Transazione fallita")
        return transaction
    except Exception as e:
        logger.error(f"Errore nell'esecuzione della transazione: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Recupera le metriche di performance del sistema."""
    if not market_environment:
        raise HTTPException(status_code=400, detail="Simulazione non inizializzata")
    
    try:
        return market_environment.get_performance_metrics()
    except Exception as e:
        logger.error(f"Errore nel recupero delle metriche di performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 