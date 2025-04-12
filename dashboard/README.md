# MKT_SIM_CL Dashboard

La dashboard di MKT_SIM_CL è un'interfaccia web interattiva per la gestione e il monitoraggio del sistema di simulazione di mercato e trading algoritmico.

## Caratteristiche

- Interfaccia utente moderna e reattiva
- Visualizzazioni in tempo reale dei dati
- Gestione dello stato persistente
- Integrazione con WebSocket per aggiornamenti live
- Grafici interattivi per l'analisi dei dati
- Monitoraggio delle performance in tempo reale

## Requisiti

- Python 3.8+
- Flask
- Dash
- Flask-SocketIO
- Plotly
- Pandas
- NumPy

## Installazione

1. Assicurati di avere Python 3.8 o superiore installato
2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

### Avvio del Server

Per avviare il server della dashboard:

```bash
python run_server.py
```

Il server sarà accessibile all'indirizzo `http://localhost:5000`

### Arresto del Server

Per arrestare il server:

```bash
python stop_server.py
```

## Struttura del Progetto

```
dashboard/
├── app.py                 # Applicazione principale
├── state_manager.py       # Gestore dello stato
├── websocket_manager.py   # Gestore delle comunicazioni WebSocket
├── visualization_manager.py # Gestore delle visualizzazioni
├── templates/            # Template HTML
│   └── index.html       # Template principale
├── static/              # File statici
├── run_server.py        # Script di avvio
└── stop_server.py       # Script di arresto
```

## Moduli

### Data Collection
- Download di dati storici
- Visualizzazione dello stato dei download
- Riepilogo dei dati disponibili

### Market Simulation
- Avvio e controllo delle simulazioni
- Visualizzazione dei dati di mercato in tempo reale
- Monitoraggio del bookkeeping

### Neural Network
- Configurazione e avvio del training
- Visualizzazione delle metriche di training
- Monitoraggio delle performance del modello

### Self-Play
- Configurazione degli agenti
- Monitoraggio delle performance degli agenti
- Visualizzazione dei portafogli

### Predictions
- Generazione di previsioni
- Confronto con i dati reali
- Analisi delle performance predittive

## Logging

I log vengono salvati nella directory `logs/` con i seguenti file:
- `dashboard.log`: Log principale dell'applicazione
- `state_manager.log`: Log del gestore dello stato
- `websocket_manager.log`: Log delle comunicazioni WebSocket
- `visualization_manager.log`: Log delle visualizzazioni

## Contribuire

1. Fai un fork del repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Committa le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Pusha al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Licenza

Questo progetto è sotto licenza MIT. Vedi il file `LICENSE` per i dettagli. 