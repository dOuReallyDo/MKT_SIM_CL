# MKT_SIM_CL - Simulatore di Mercato

Sistema di simulazione del mercato finanziario basato su microservizi.

## Architettura

Il sistema è composto dai seguenti microservizi:

- **API Gateway** (porta 8000): Gestisce le richieste in ingresso e l'autenticazione
- **Market Simulator** (porta 8001): Simula il comportamento del mercato
- **AI Engine** (porta 8002): Gestisce i modelli di machine learning
- **Data Pipeline** (porta 8003): Gestisce la raccolta e l'elaborazione dei dati

Servizi di supporto:
- **Redis**: Cache distribuita
- **InfluxDB**: Database per serie temporali
- **Prometheus**: Monitoraggio
- **Grafana**: Visualizzazione dei dati

## Prerequisiti

- Docker e Docker Compose
- Python 3.9+
- Git

## Configurazione

1. Clona il repository:
```bash
git clone https://github.com/your-username/MKT_SIM_cl.git
cd MKT_SIM_cl
```

2. Crea un file `.env` nella root del progetto con le seguenti variabili:
```env
# API Gateway
SECRET_KEY=your-secret-key-here
MARKET_SIMULATOR_URL=http://market-simulator:8001
AI_ENGINE_URL=http://ai-engine:8002
DATA_PIPELINE_URL=http://data-pipeline:8003

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# InfluxDB
INFLUXDB_USERNAME=admin
INFLUXDB_PASSWORD=admin123
INFLUXDB_ORG=mkt_sim
INFLUXDB_BUCKET=market-data
INFLUXDB_TOKEN=your-influxdb-token-here
INFLUXDB_URL=http://influxdb:8086

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin123

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Market Simulator
SIMULATION_INTERVAL=60
MAX_SIMULATIONS=100
CACHE_TTL=3600

# AI Engine
MODEL_SAVE_PATH=/app/models
TRAINING_BATCH_SIZE=32
PREDICTION_THRESHOLD=0.7

# Data Pipeline
DATA_COLLECTION_INTERVAL=300
MAX_RETRIES=3
DATA_CLEANUP_THRESHOLD=0.95

# Monitoring
PROMETHEUS_METRICS_PORT=9090
GRAFANA_PORT=3000
```

## Avvio del Sistema

1. Avvia tutti i servizi:
```bash
docker-compose up -d
```

2. Verifica lo stato dei servizi:
```bash
docker-compose ps
```

## Accesso ai Servizi

- API Gateway: http://localhost:8000
- Market Simulator: http://localhost:8001
- AI Engine: http://localhost:8002
- Data Pipeline: http://localhost:8003
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## API Endpoints

### API Gateway

- `POST /token`: Autenticazione
- `POST /simulation`: Avvia una nuova simulazione
- `POST /training`: Avvia il training di un modello
- `POST /data/collect`: Avvia la raccolta dei dati
- `GET /health`: Verifica lo stato dei servizi

### Market Simulator

- `POST /simulation/start`: Avvia una simulazione
- `GET /market/data/{date}`: Ottiene i dati di mercato
- `GET /market/price/{symbol}`: Ottiene il prezzo di un simbolo
- `POST /transaction`: Esegue una transazione

### AI Engine

- `POST /train`: Addestra un nuovo modello
- `POST /predict`: Effettua predizioni
- `GET /models/{model_id}`: Ottiene i metadati del modello
- `POST /evaluate/{model_id}`: Valuta le performance del modello

### Data Pipeline

- `POST /data/collect`: Raccoglie i dati di mercato
- `POST /data/validate`: Valida i dati raccolti
- `POST /data/cleanup`: Pulisce i dati

## Monitoraggio

### Prometheus

- Accesso: http://localhost:9090
- Metriche disponibili:
  - Numero di richieste API
  - Latenza delle richieste
  - Utilizzo delle risorse
  - Performance dei modelli

### Grafana

- Accesso: http://localhost:3000
- Dashboard predefinite:
  - Overview del sistema
  - Performance dei servizi
  - Metriche di business

## Sviluppo

### Struttura del Progetto

```
MKT_SIM_cl/
├── services/
│   ├── api-gateway/
│   ├── market-simulator/
│   ├── ai-engine/
│   └── data-pipeline/
├── prometheus/
├── logs/
├── data/
├── models/
├── docker-compose.yml
└── README.md
```

### Aggiungere un Nuovo Servizio

1. Crea una nuova directory in `services/`
2. Aggiungi i file necessari (Dockerfile, requirements.txt, etc.)
3. Aggiorna `docker-compose.yml`
4. Aggiungi le configurazioni in Prometheus

## Troubleshooting

### Logs

I log sono disponibili in:
- Directory `logs/` per i log dell'applicazione
- `docker-compose logs [service]` per i log dei container

### Problemi Comuni

1. **Servizi non raggiungibili**:
   - Verifica che tutti i container siano in esecuzione
   - Controlla i log per errori
   - Verifica le configurazioni di rete

2. **Errori di autenticazione**:
   - Verifica le credenziali in `.env`
   - Controlla i token JWT

3. **Problemi di performance**:
   - Monitora le risorse con Grafana
   - Verifica le configurazioni di cache
   - Controlla i log per bottleneck

## Contribuire

1. Fork il repository
2. Crea un branch per la feature
3. Commit le modifiche
4. Push al branch
5. Crea una Pull Request

## Licenza

Questo progetto è sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.
