# Guida Utente MKT_SIM_CL

## Indice
1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Configurazione](#configurazione)
4. [Utilizzo Base](#utilizzo-base)
5. [Strategie di Trading](#strategie-di-trading)
6. [Dashboard](#dashboard)
7. [Analisi dei Risultati](#analisi-dei-risultati)
8. [Risoluzione Problemi](#risoluzione-problemi)

## Introduzione

MKT_SIM_CL è un sistema di simulazione del mercato azionario che permette di testare strategie di trading algoritmico in un ambiente virtuale. Il sistema supporta multiple strategie di trading e integra modelli di deep learning per le previsioni dei prezzi.

### Caratteristiche Principali

- Simulazione realistica del mercato azionario
- Supporto per multiple strategie di trading
- Integrazione con reti neurali per le previsioni
- Dashboard interattiva per l'analisi
- Gestione sicura delle credenziali
- Sistema di caching per ottimizzare le performance

## Installazione

### Requisiti di Sistema

- Python 3.8 o superiore
- 8GB RAM (consigliato)
- Almeno 2GB di spazio su disco

### Procedura di Installazione

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

4. Configura le variabili d'ambiente:
```bash
cp .env.example .env
# Modifica .env con le tue credenziali
```

## Configurazione

### File di Configurazione

Il sistema utilizza un file di configurazione JSON per gestire i parametri principali:

```json
{
    "market": {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "timeframes": ["1d", "1h"],
        "default_timeframe": "1d"
    },
    "trading": {
        "initial_capital": 100000,
        "order_types": ["market", "limit", "stop"],
        "position_sizing": {
            "default_quantity": 10,
            "max_position_size": 0.2
        },
        "risk_management": {
            "use_stop_loss": true,
            "stop_loss_percentage": 2.0,
            "use_take_profit": true,
            "take_profit_percentage": 5.0
        }
    }
}
```

### Parametri Principali

- **symbols**: Lista dei simboli da simulare
- **start_date**: Data di inizio della simulazione
- **end_date**: Data di fine della simulazione
- **initial_capital**: Capitale iniziale per il trading
- **position_sizing**: Parametri per la gestione delle dimensioni delle posizioni
- **risk_management**: Parametri per la gestione del rischio

## Utilizzo Base

### Avvio della Simulazione

1. Avvia la simulazione:
```bash
python main.py
```

2. Accedi alla dashboard:
```bash
python dashboard/app.py
```

### Interfaccia a Linea di Comando

Il sistema fornisce un'interfaccia a linea di comando per le operazioni principali:

```bash
# Avvia una simulazione con parametri personalizzati
python main.py --symbols AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31

# Genera un report dettagliato
python main.py --generate-report

# Esegui backtesting di una strategia
python main.py --strategy mean_reversion --backtest
```

## Strategie di Trading

### Strategie Disponibili

1. **Random Strategy**
   - Genera segnali casuali di trading
   - Utile per benchmark e test iniziali

2. **Mean Reversion Strategy**
   - Basata sul principio di regressione alla media
   - Parametri configurabili:
     - window: dimensione della finestra
     - threshold: soglia per i segnali

3. **Trend Following Strategy**
   - Segue i trend di mercato
   - Parametri configurabili:
     - short_window: finestra breve
     - long_window: finestra lunga

4. **Value Investing Strategy**
   - Basata su metriche fondamentali
   - Parametri configurabili:
     - pe_ratio_threshold: soglia P/E
     - market_cap_threshold: soglia capitalizzazione

5. **Neural Network Strategy**
   - Utilizza modelli di deep learning
   - Parametri configurabili:
     - model_type: tipo di modello
     - sequence_length: lunghezza sequenza

### Configurazione delle Strategie

```json
{
    "strategies": {
        "active_strategy": "mean_reversion",
        "strategy_params": {
            "mean_reversion": {
                "window": 20,
                "threshold": 2.0
            }
        }
    }
}
```

## Dashboard

### Funzionalità Principali

1. **Monitoraggio in Tempo Reale**
   - Visualizzazione prezzi
   - Stato delle posizioni
   - Performance degli agenti

2. **Analisi Tecnica**
   - Grafici dei prezzi
   - Indicatori tecnici
   - Segnali di trading

3. **Gestione del Portafoglio**
   - Valore totale
   - Distribuzione asset
   - Performance storica

4. **Report e Statistiche**
   - Metriche di performance
   - Analisi del rischio
   - Confronto strategie

### Accesso alla Dashboard

1. Avvia il server:
```bash
python dashboard/app.py
```

2. Apri il browser:
```
http://localhost:5000
```

## Analisi dei Risultati

### Metriche di Performance

1. **Rendimento**
   - Return totale
   - Return giornaliero
   - Sharpe ratio

2. **Rischio**
   - Volatilità
   - Drawdown massimo
   - Value at Risk (VaR)

3. **Efficienza**
   - Win rate
   - Profit factor
   - Recovery factor

### Generazione Report

```bash
# Genera report completo
python main.py --generate-report

# Genera report specifico
python main.py --report-type performance
```

## Risoluzione Problemi

### Problemi Comuni

1. **Errore di Connessione**
   - Verifica le credenziali API
   - Controlla la connessione internet
   - Verifica i proxy

2. **Errore di Memoria**
   - Riduci la dimensione del dataset
   - Attiva il garbage collection
   - Ottimizza i parametri di caching

3. **Errore di Performance**
   - Verifica le risorse di sistema
   - Ottimizza i parametri di simulazione
   - Utilizza il caching

### Log e Debug

I log sono disponibili nella directory `logs/`:
- `market_simulator.log`: Log principale
- `error.log`: Log degli errori
- `performance.log`: Log delle performance

### Supporto

Per supporto tecnico:
- Email: support@example.com
- GitHub Issues: [Repository Issues](https://github.com/yourusername/mkt_sim_cl/issues) 