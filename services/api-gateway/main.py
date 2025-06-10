import os
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uvicorn
import httpx
import jwt
from passlib.context import CryptContext
import redis
from prometheus_client import Counter, Histogram
import time

from app.models import (
    User, UserCreate, UserLogin, Token, TokenData,
    SimulationRequest, SimulationResponse,
    ModelTrainingRequest, ModelTrainingResponse,
    DataCollectionRequest, DataCollectionResponse
)

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/api_gateway_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('APIGateway')

app = FastAPI(title="API Gateway")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurazione dei servizi
MARKET_SIMULATOR_URL = os.getenv("MARKET_SIMULATOR_URL", "http://market-simulator:8001")
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://ai-engine:8002")
DATA_PIPELINE_URL = os.getenv("DATA_PIPELINE_URL", "http://data-pipeline:8003")

# Configurazione JWT
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configurazione Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Metriche Prometheus
REQUEST_COUNT = Counter('api_requests_total', 'Numero totale di richieste API', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Latenza delle richieste API', ['endpoint'])

# Configurazione password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Client HTTP
http_client = httpx.AsyncClient()

# Funzioni di utilità
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token non valido"
            )
        token_data = TokenData(**payload)
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token non valido"
        )
    
    # TODO: Recupera l'utente dal database
    # Per ora restituiamo un utente di esempio
    return User(
        id=token_data.user_id,
        email="user@example.com",
        username="user",
        role=token_data.role,
        created_at=datetime.now()
    )

# Endpoint di autenticazione
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # TODO: Implementare la verifica delle credenziali
    # Per ora restituiamo un token di esempio
    access_token = create_access_token(
        data={"sub": "user123", "role": "user"}
    )
    return Token(
        access_token=access_token,
        expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

# Endpoint di simulazione
@app.post("/simulation", response_model=SimulationResponse)
async def create_simulation(
    request: SimulationRequest,
    current_user: User = Depends(get_current_user)
):
    """Crea una nuova simulazione di mercato."""
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/simulation", method="POST").inc()
    
    try:
        async with http_client as client:
            response = await client.post(
                f"{MARKET_SIMULATOR_URL}/simulation/start",
                json=request.dict()
            )
            response.raise_for_status()
            result = response.json()
            
            REQUEST_LATENCY.labels(endpoint="/simulation").observe(time.time() - start_time)
            
            return SimulationResponse(
                simulation_id=result["simulation_id"],
                status="created",
                created_at=datetime.now(),
                results=result.get("results")
            )
    
    except Exception as e:
        logger.error(f"Errore nella creazione della simulazione: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint di training
@app.post("/training", response_model=ModelTrainingResponse)
async def start_training(
    request: ModelTrainingRequest,
    current_user: User = Depends(get_current_user)
):
    """Avvia il training di un nuovo modello."""
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/training", method="POST").inc()
    
    try:
        async with http_client as client:
            response = await client.post(
                f"{AI_ENGINE_URL}/train",
                json=request.dict()
            )
            response.raise_for_status()
            result = response.json()
            
            REQUEST_LATENCY.labels(endpoint="/training").observe(time.time() - start_time)
            
            return ModelTrainingResponse(
                training_id=result["model_id"],
                status="created",
                created_at=datetime.now(),
                model_id=result.get("model_id"),
                metrics=result.get("metrics")
            )
    
    except Exception as e:
        logger.error(f"Errore nell'avvio del training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint di raccolta dati
@app.post("/data/collect", response_model=DataCollectionResponse)
async def collect_data(
    request: DataCollectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Avvia la raccolta dei dati di mercato."""
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/data/collect", method="POST").inc()
    
    try:
        async with http_client as client:
            response = await client.post(
                f"{DATA_PIPELINE_URL}/data/collect",
                json=request.dict()
            )
            response.raise_for_status()
            result = response.json()
            
            REQUEST_LATENCY.labels(endpoint="/data/collect").observe(time.time() - start_time)
            
            return DataCollectionResponse(
                collection_id=result["collection_id"],
                status="created",
                created_at=datetime.now(),
                data=result.get("data")
            )
    
    except Exception as e:
        logger.error(f"Errore nella raccolta dei dati: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint di health check
@app.get("/health")
async def health_check():
    """Verifica lo stato di salute dei servizi."""
    services = {
        "market-simulator": MARKET_SIMULATOR_URL,
        "ai-engine": AI_ENGINE_URL,
        "data-pipeline": DATA_PIPELINE_URL
    }
    
    health_status = {}
    for service, url in services.items():
        try:
            async with http_client as client:
                response = await client.get(f"{url}/health")
                health_status[service] = response.status_code == 200
        except Exception:
            health_status[service] = False
    
    return {
        "status": "healthy" if all(health_status.values()) else "unhealthy",
        "services": health_status,
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 