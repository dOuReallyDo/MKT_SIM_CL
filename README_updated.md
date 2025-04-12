# MKT_SIM_CL - Sistema di Simulazione del Mercato Azionario

Sistema avanzato di simulazione del mercato azionario con trading algoritmico basato su IA, completamente ristrutturato per maggiore coerenza e facilità d'uso.

## Introduzione

MKT_SIM_CL è un sistema di simulazione avanzato per il mercato azionario che permette di testare diverse strategie di trading in un ambiente controllato. Il sistema include supporto per reti neurali, strategie di trading classiche, e funzionalità di backtesting.

## Funzionalità principali

- Simulazione di mercato con agenti multipli
- Supporto per diverse strategie di trading
- Addestramento di reti neurali
- Self-play per l'evoluzione delle strategie
- Dashboard integrata per la visualizzazione dei risultati
- Wizard interattivo per la configurazione del sistema

## Caratteristiche

- **Simulazione realistica del mercato azionario**
  - Ambiente di mercato basato su dati storici reali
  - Gestione di prezzi, volumi e orari di trading
  - Sistema di cache distribuito per ottimizzare le performance

- **Multiple strategie di trading**
  - Strategie casuali (baseline)
  - Mean Reversion
  - Trend Following
  - Value Investing
  - Strategie basate su reti neurali

- **Reti neurali per previsioni**
  - Supporto per modelli LSTM, CNN e Transformer
  - Addestramento su dati storici
  - Previsione di prezzi futuri
  - Auto-apprendimento tramite self-play

- **Interfacce utente**
  - Dashboard web interattiva
  - Wizard guidato per utenti non tecnici
  - Visualizzazione avanzata dei risultati

- **Funzionalità avanzate**
  - Monitoraggio completo delle performance
  - Gestione sicura delle credenziali
  - Configurazione flessibile
  - Sistema di logging avanzato

## Nuova struttura del progetto

```
mkt_sim_cl/
├── config/                  # Configurazione unificata
│   ├── __init__.py
│   ├── base_config.py       # Configurazione di base
│   ├── monitoring_config.py # Configurazione del monitoraggio
│   └── user_config.py       # Configurazione dell'utente
├── market_simulator/        # Moduli del simulatore
│   ├── __init__.py
│   ├── environment.py       # Ambiente di mercato
│   ├── agents.py            # Agenti di trading
│   └── simulation.py        # Gestore della simulazione
├── data/                    # Dati di mercato
│   ├── __init__.py
│   └── collector.py         # Collector di dati
├── dashboard/               # Dashboard web
│   ├── app.py               # Applicazione Flask
│   ├── templates/           # Template HTML
│   └── static/              # Risorse statiche
├── neural_network/          # Reti neurali
│   ├── __init__.py
│   └── model_trainer.py     # Trainer dei modelli
├── trading_strategy/        # Strategie di trading
│   ├── __init__.py
│   └── strategies.py        # Implementazione strategie
├── utils/                   # Utility condivise
│   ├── __init__.py
│   ├── security.py          # Gestione sicurezza
│   └── cache.py             # Sistema di caching
├── interface/               # Interfaccia per utenti non tecnici
│   ├── __init__.py
│   └── wizard.py            # Wizard guidato
├── run_dashboard.py         # Script per avviare la dashboard
└── main.py                  # Script principale
```

## Requisiti

- Python 3.8 o superiore
- Dipendenze elencate in `requirements.txt`

## Installazione

1. **Clona il repository:**
   ```
   git clone https://github.com/tuonome/MKT_SIM_CL.git
   cd MKT_SIM_CL
   ```

2. **Installa le dipendenze:**
   ```
   pip install -r requirements.txt
   ```

3. **Configurazione iniziale:**
   Copia il file `.env.example` in `.env` e modifica i parametri secondo le tue esigenze.
   ```
   cp .env.example .env
   ```

## Utilizzo della Dashboard con Wizard Integrato

Il modo più semplice per configurare e utilizzare il sistema è attraverso la dashboard web con il wizard integrato:

1. **Avvia la dashboard**:
   ```bash
   python run_dashboard.py
   ```

2. **Accedi al wizard**:
   - Nel menu principale, clicca su "Wizard" e seleziona "Wizard Completo"
   - In alternativa, puoi accedere direttamente alle singole fasi del wizard dal menu a tendina

3. **Procedura guidata**:
   La procedura guidata ti accompagnerà attraverso i seguenti passaggi:
   - Configurazione dei simboli di trading
   - Impostazione del periodo di tempo
   - Configurazione del capitale iniziale
   - Selezione e configurazione della strategia di trading
   - Impostazione del numero di agenti
   - Salvataggio della configurazione
   - Raccolta dati
   - Esecuzione della simulazione

4. **Funzionalità avanzate**:
   Dalla dashboard puoi anche accedere a:
   - Addestramento di reti neurali
   - Self-play per l'evoluzione delle strategie
   - Visualizzazione dei report
   - Analisi dei dati disponibili

Questo approccio ti permette di utilizzare tutte le funzionalità del sistema senza dover usare il terminale.

## Funzionalità di self-play

Il sistema supporta il self-play per l'auto-apprendimento delle reti neurali:

1. Configurazione del self-play:
```bash
python -m interface.wizard
```
Seleziona la strategia "neural_network" e configura i parametri di self-play.

2. Esecuzione del self-play:
```bash
python main.py --mode self_play
```

Il sistema creerà una popolazione di modelli, li farà competere tra loro e selezionerà i migliori per la generazione successiva.

## Documentazione

La documentazione completa è disponibile nella directory `docs/`:
- [API Documentation](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Development Guide](docs/development.md)

## Contribuire

1. Fork il repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Licenza

Questo progetto è sotto licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.

## Autori

- Mario Curcio - Sviluppatore Principale

## Riconoscimenti

Un ringraziamento speciale a tutti i contributori che hanno aiutato a migliorare questo progetto. 