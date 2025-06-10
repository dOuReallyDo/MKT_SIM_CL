#!/bin/bash

# Verifica se il file .env esiste
if [ ! -f .env ]; then
    echo "File .env non trovato. Esegui prima ./scripts/setup.sh"
    exit 1
fi

# Avvia i servizi in modalità sviluppo
make dev 