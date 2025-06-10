#!/bin/bash

# Verifica se il file .env esiste
if [ ! -f .env ]; then
    echo "File .env non trovato. Esegui prima ./scripts/setup.sh"
    exit 1
fi

# Ferma i servizi esistenti
make down

# Pulisce l'ambiente
make clean

# Costruisce le nuove immagini
make build

# Avvia i servizi in produzione
make up

echo "Deployment completato con successo!" 