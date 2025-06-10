#!/bin/bash

# Colori per i messaggi
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funzione per stampare i messaggi
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica dei prerequisiti
check_prerequisites() {
    print_message "Verifica dei prerequisiti..."

    # Verifica Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker non è installato. Per favore installa Docker."
        exit 1
    fi

    # Verifica Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose non è installato. Per favore installa Docker Compose."
        exit 1
    fi

    # Verifica Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 non è installato. Per favore installa Python 3."
        exit 1
    fi

    print_success "Tutti i prerequisiti sono soddisfatti"
}

# Creazione delle directory necessarie
create_directories() {
    print_message "Creazione delle directory necessarie..."

    directories=(
        "logs"
        "data"
        "models"
        "prometheus"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Creata directory: $dir"
        else
            print_message "Directory già esistente: $dir"
        fi
    done
}

# Configurazione dell'ambiente
setup_environment() {
    print_message "Configurazione dell'ambiente..."

    # Creazione del file .env se non esiste
    if [ ! -f .env ]; then
        print_message "Creazione del file .env..."
        cat > .env << EOL
# API Gateway
SECRET_KEY=$(openssl rand -hex 32)
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
INFLUXDB_TOKEN=$(openssl rand -hex 32)
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
EOL
        print_success "File .env creato con successo"
    else
        print_message "File .env già esistente"
    fi
}

# Inizializzazione di InfluxDB
init_influxdb() {
    print_message "Inizializzazione di InfluxDB..."

    # Attendi che InfluxDB sia pronto
    until docker-compose exec influxdb influx ping &> /dev/null; do
        print_message "In attesa che InfluxDB sia pronto..."
        sleep 5
    done

    # Crea l'organizzazione e il bucket
    docker-compose exec influxdb influx setup \
        --username $INFLUXDB_USERNAME \
        --password $INFLUXDB_PASSWORD \
        --org $INFLUXDB_ORG \
        --bucket $INFLUXDB_BUCKET \
        --force

    print_success "InfluxDB inizializzato con successo"
}

# Inizializzazione di Grafana
init_grafana() {
    print_message "Inizializzazione di Grafana..."

    # Attendi che Grafana sia pronto
    until curl -s http://localhost:3000/api/health &> /dev/null; do
        print_message "In attesa che Grafana sia pronto..."
        sleep 5
    done

    # Configura la fonte dati Prometheus
    curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"name":"Prometheus","type":"prometheus","url":"http://prometheus:9090","access":"proxy"}' \
        http://admin:admin123@localhost:3000/api/datasources

    print_success "Grafana inizializzato con successo"
}

# Funzione principale
main() {
    print_message "Inizializzazione del progetto MKT_SIM_CL..."

    # Verifica dei prerequisiti
    check_prerequisites

    # Creazione delle directory
    create_directories

    # Configurazione dell'ambiente
    setup_environment

    # Build e avvio dei servizi
    print_message "Build e avvio dei servizi..."
    docker-compose build
    docker-compose up -d

    # Inizializzazione dei servizi
    init_influxdb
    init_grafana

    print_success "Inizializzazione completata con successo!"
    print_message "Puoi accedere ai servizi su:"
    echo "- API Gateway: http://localhost:8000"
    echo "- Market Simulator: http://localhost:8001"
    echo "- AI Engine: http://localhost:8002"
    echo "- Data Pipeline: http://localhost:8003"
    echo "- Grafana: http://localhost:3000"
    echo "- Prometheus: http://localhost:9090"
}

# Esecuzione dello script
main 