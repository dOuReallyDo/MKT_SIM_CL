# MKT_SIM_CL - Sistema di Simulazione del Mercato Azionario

Sistema di simulazione del mercato azionario con trading algoritmico basato su Intelligenza Artificiale.

## Panoramica del sistema

MKT_SIM_CL è un sistema completo che integra simulazione del mercato azionario, strategie di trading algoritmico, reti neurali e visualizzazione interattiva tramite dashboard. Il sistema è composto da diversi moduli:

1. **Data Collection** - Raccolta e preprocessing di dati storici di mercato
2. **Market Simulation** - Simulazione del mercato e gestione degli agenti di trading
3. **Trading Strategies** - Implementazione di diverse strategie di trading algoritmico
4. **Neural Network** - Addestramento e inferenza di modelli di deep learning per la previsione dei prezzi
5. **Dashboard** - Interfaccia grafica per la visualizzazione e analisi dei risultati

## Requisiti

- Python 3.8 o superiore
- Dipendenze elencate in `requirements.txt`

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/yourusername/mkt_sim_cl.git
cd mkt_sim_cl
```

2. Crea un ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

### Esecuzione rapida dell'intero sistema

Per eseguire l'intero sistema (simulazione + dashboard) con le impostazioni predefinite:

```bash
python run_system.py
```

Questo comando:
1. Prepara i dati di mercato
2. Esegue una simulazione con strategie di trading
3. Avvia la dashboard per la visualizzazione dei risultati

### Modalità di esecuzione specifiche

#### Solo simulazione

Per eseguire solo la simulazione di mercato:

```bash
python run_system.py --mode simulation --strategy random --num-agents 5
```

Parametri disponibili:
- `--symbols` - Simboli da utilizzare (es. `--symbols AAPL MSFT GOOGL`)
- `--days` - Numero di giorni per la simulazione (es. `--days 30`)
- `--strategy` - Strategia di trading da utilizzare (`random`, `mean_reversion`, `trend_following`, `value_investing`, `neural_network`)
- `--num-agents` - Numero di agenti per la simulazione
- `--initial-capital` - Capitale iniziale per gli agenti

#### Solo dashboard

Per avviare solo la dashboard:

```bash
python run_system.py --mode dashboard --dashboard-port 8080
```

Parametri disponibili:
- `--dashboard-port` - Porta per il server web della dashboard
- `--no-browser` - Non aprire automaticamente il browser

### Script di test

Per verificare il corretto funzionamento del sistema, è possibile eseguire il test di integrazione:

```bash
python test_integration.py
```

## Moduli del sistema

### 1. Data Collection (`data/collector.py`)

Questo modulo si occupa di:
- Scaricare dati storici da fonti come Yahoo Finance
- Pulire e normalizzare i dati
- Salvare i dati in cache per riutilizzo futuro
- Generare feature per l'addestramento dei modelli

Per scaricare dati manualmente:

```bash
python -c "from data.collector import DataCollector; collector = DataCollector(); collector.get_stock_data('AAPL', '2023-01-01', '2023-12-31', force_download=True)"
```

### 2. Market Simulation (`market_simulator/`)

Il simulatore di mercato include:
- Un ambiente di mercato che simula lo scambio di azioni
- Agenti di trading che eseguono strategie
- Gestione delle transazioni e del portafoglio
- Calcolo delle performance

### 3. Trading Strategies (`trading_strategy/`)

Il sistema supporta diverse strategie di trading:
- **Random** - Strategie casuali per testing
- **Mean Reversion** - Compra quando il prezzo è sotto la media, vende quando è sopra
- **Trend Following** - Compra in trend rialzista, vende in trend ribassista
- **Value Investing** - Basata sui fondamentali delle aziende
- **Neural Network** - Utilizza reti neurali per previsioni

### 4. Neural Network (`neural_network/`)

Il modulo di reti neurali include:
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Networks)
- Transformer networks

Per addestrare un modello:

```bash
python main.py --mode training
```

### 5. Dashboard (`dashboard/`)

La dashboard interattiva permette di:
- Visualizzare i risultati delle simulazioni
- Analizzare le performance degli agenti
- Confrontare diverse strategie
- Monitorare l'andamento dei titoli

Per avviare la dashboard separatamente:

```bash
python run_dashboard.py
```

## Struttura del progetto

```
MKT_SIM_CL/
├── data/                 # Dati di mercato e modulo di raccolta
│   └── collector.py      # Script per la raccolta dati
├── market_simulator/     # Simulatore di mercato
│   ├── environment.py    # Ambiente di simulazione
│   ├── agents.py         # Agenti di trading
│   └── simulation.py     # Gestore della simulazione
├── trading_strategy/     # Strategie di trading
│   └── strategies.py     # Implementazioni delle strategie
├── neural_network/       # Moduli per reti neurali
│   └── model_trainer.py  # Addestramento modelli
├── dashboard/            # Dashboard interattiva
│   ├── app.py            # Applicazione Flask
│   ├── static/           # File statici (CSS, JS)
│   └── templates/        # Template HTML
├── reports/              # Report delle simulazioni
├── models/               # Modelli addestrati
└── logs/                 # Log del sistema
```

## Configurazione

Il sistema legge la configurazione da diversi file:
- `config_user.json` - Configurazione personalizzata (principali parametri)
- `config_updated.json` - Configurazione base del sistema
- `dashboard_state.json` - Stato della dashboard

## Licenza

Questo progetto è sotto licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.

## Contatti

Mario Curcio - mario.curcio@example.com

Link del progetto: [https://github.com/yourusername/mkt_sim_cl](https://github.com/yourusername/mkt_sim_cl) 