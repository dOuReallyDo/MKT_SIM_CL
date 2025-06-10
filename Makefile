.PHONY: build up down restart logs clean test lint help

# Variabili
DOCKER_COMPOSE = docker-compose
PYTHON = python3
VENV = venv

# Colori per i messaggi
BLUE = \033[0;34m
GREEN = \033[0;32m
RED = \033[0;31m
NC = \033[0m # No Color

help:
	@echo "${BLUE}Comandi disponibili:${NC}"
	@echo "${GREEN}build${NC}    - Costruisce le immagini Docker"
	@echo "${GREEN}up${NC}      - Avvia tutti i servizi"
	@echo "${GREEN}down${NC}    - Ferma tutti i servizi"
	@echo "${GREEN}restart${NC} - Riavvia tutti i servizi"
	@echo "${GREEN}logs${NC}    - Mostra i log dei servizi"
	@echo "${GREEN}clean${NC}   - Pulisce i container e le immagini non utilizzate"
	@echo "${GREEN}test${NC}    - Esegue i test"
	@echo "${GREEN}lint${NC}    - Esegue il linting del codice"
	@echo "${GREEN}setup${NC}   - Configura l'ambiente di sviluppo"
	@echo "${GREEN}monitor${NC} - Avvia il monitoraggio dei servizi"

build:
	@echo "${BLUE}Costruzione delle immagini Docker...${NC}"
	$(DOCKER_COMPOSE) build

up:
	@echo "${BLUE}Avvio dei servizi...${NC}"
	$(DOCKER_COMPOSE) up -d

down:
	@echo "${BLUE}Arresto dei servizi...${NC}"
	$(DOCKER_COMPOSE) down

restart: down up
	@echo "${GREEN}Servizi riavviati con successo${NC}"

logs:
	@echo "${BLUE}Visualizzazione dei log...${NC}"
	$(DOCKER_COMPOSE) logs -f

clean:
	@echo "${BLUE}Pulizia dei container e delle immagini non utilizzate...${NC}"
	$(DOCKER_COMPOSE) down --rmi all --volumes --remove-orphans
	docker system prune -f

test:
	@echo "${BLUE}Esecuzione dei test...${NC}"
	$(DOCKER_COMPOSE) run --rm api-gateway pytest
	$(DOCKER_COMPOSE) run --rm market-simulator pytest
	$(DOCKER_COMPOSE) run --rm ai-engine pytest
	$(DOCKER_COMPOSE) run --rm data-pipeline pytest

lint:
	@echo "${BLUE}Esecuzione del linting...${NC}"
	$(DOCKER_COMPOSE) run --rm api-gateway flake8
	$(DOCKER_COMPOSE) run --rm market-simulator flake8
	$(DOCKER_COMPOSE) run --rm ai-engine flake8
	$(DOCKER_COMPOSE) run --rm data-pipeline flake8

setup:
	@echo "${BLUE}Configurazione dell'ambiente di sviluppo...${NC}"
	@if [ ! -f .env ]; then \
		echo "${RED}File .env non trovato. Creazione del file .env.example...${NC}"; \
		cp .env.example .env; \
	fi
	@echo "${GREEN}Ambiente configurato con successo${NC}"

monitor:
	@echo "${BLUE}Avvio del monitoraggio...${NC}"
	@echo "${GREEN}Prometheus: http://localhost:9090${NC}"
	@echo "${GREEN}Grafana: http://localhost:3000${NC}"
	@echo "${GREEN}Premi Ctrl+C per terminare${NC}"
	@$(DOCKER_COMPOSE) logs -f prometheus grafana

# Comandi specifici per i servizi
api-logs:
	@echo "${BLUE}Log dell'API Gateway...${NC}"
	$(DOCKER_COMPOSE) logs -f api-gateway

market-logs:
	@echo "${BLUE}Log del Market Simulator...${NC}"
	$(DOCKER_COMPOSE) logs -f market-simulator

ai-logs:
	@echo "${BLUE}Log dell'AI Engine...${NC}"
	$(DOCKER_COMPOSE) logs -f ai-engine

data-logs:
	@echo "${BLUE}Log del Data Pipeline...${NC}"
	$(DOCKER_COMPOSE) logs -f data-pipeline

# Comandi per il database
db-backup:
	@echo "${BLUE}Backup del database...${NC}"
	$(DOCKER_COMPOSE) exec influxdb influx backup /backup/$(shell date +%Y%m%d_%H%M%S)

db-restore:
	@echo "${BLUE}Ripristino del database...${NC}"
	@read -p "Inserisci il nome del file di backup: " backup_file; \
	$(DOCKER_COMPOSE) exec influxdb influx restore /backup/$$backup_file

# Comandi per i modelli
models-backup:
	@echo "${BLUE}Backup dei modelli...${NC}"
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/

models-restore:
	@echo "${BLUE}Ripristino dei modelli...${NC}"
	@read -p "Inserisci il nome del file di backup: " backup_file; \
	tar -xzf $$backup_file

# Comandi per i log
logs-clean:
	@echo "${BLUE}Pulizia dei log...${NC}"
	find logs/ -type f -name "*.log" -mtime +7 -delete

logs-backup:
	@echo "${BLUE}Backup dei log...${NC}"
	tar -czf logs_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz logs/

# Comandi per il monitoraggio
metrics:
	@echo "${BLUE}Metriche del sistema...${NC}"
	curl -s http://localhost:9090/api/v1/query?query=up | jq .

health:
	@echo "${BLUE}Stato di salute dei servizi...${NC}"
	curl -s http://localhost:8000/health | jq . 