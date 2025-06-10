#!/bin/bash

# Crea le cartelle necessarie
mkdir -p services/{api-gateway,market-simulator,ai-engine,data-pipeline,dashboard}/{app,}
mkdir -p infrastructure/{databases/{postgres,influxdb},monitoring/{prometheus,grafana},nginx}
mkdir -p scripts

# Copia il file .env.example se non esiste
if [ ! -f .env ]; then
    cp .env.example .env
    echo "File .env creato da .env.example. Modifica le variabili d'ambiente secondo necessità."
fi

# Rendi eseguibili gli script
chmod +x scripts/*.sh

# Installa le dipendenze di sviluppo
make setup-dev

echo "Setup completato con successo!" 