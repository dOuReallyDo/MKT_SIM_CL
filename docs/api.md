# Documentazione API MKT_SIM_CL

## Indice
1. [MarketEnvironment](#marketenvironment)
2. [TradingAgent](#tradingagent)
3. [SimulationManager](#simulationmanager)
4. [CredentialManager](#credentialmanager)

## MarketEnvironment

Classe che gestisce l'ambiente di simulazione del mercato.

### Metodi Principali

#### `__init__(stocks_data, trading_days, opening_time="09:30", closing_time="16:00")`
Inizializza l'ambiente di mercato.

**Parametri:**
- `stocks_data`: Dizionario di DataFrame con i dati storici per ogni simbolo
- `trading_days`: Lista di date di trading
- `opening_time`: Orario di apertura del mercato (default: "09:30")
- `closing_time`: Orario di chiusura del mercato (default: "16:00")

#### `get_current_price(symbol)`
Recupera il prezzo corrente per un simbolo.

**Parametri:**
- `symbol`: Simbolo del titolo

**Ritorna:**
- `float`: Prezzo corrente del titolo

#### `get_market_data(date)`
Recupera i dati di mercato per una data specifica.

**Parametri:**
- `date`: Data per cui recuperare i dati

**Ritorna:**
- `dict`: Dizionario con i dati di mercato

#### `execute_transaction(agent, symbol, action, price)`
Esegue una transazione di trading.

**Parametri:**
- `agent`: Oggetto TradingAgent
- `symbol`: Simbolo del titolo
- `action`: Tipo di azione ('buy' o 'sell')
- `price`: Prezzo della transazione

**Ritorna:**
- `dict`: Dettagli della transazione se eseguita con successo
- `None`: Se la transazione non può essere eseguita

## TradingAgent

Classe che rappresenta un agente di trading.

### Metodi Principali

#### `__init__(id, initial_capital, strategy)`
Inizializza un agente di trading.

**Parametri:**
- `id`: Identificatore univoco dell'agente
- `initial_capital`: Capitale iniziale
- `strategy`: Oggetto strategia di trading

#### `generate_signal(market_data)`
Genera un segnale di trading basato sulla strategia.

**Parametri:**
- `market_data`: Dizionario con i dati di mercato

**Ritorna:**
- `dict`: Segnale di trading
- `None`: Se non ci sono segnali

#### `get_portfolio_value(market_data)`
Calcola il valore totale del portafoglio.

**Parametri:**
- `market_data`: Dizionario con i dati di mercato

**Ritorna:**
- `float`: Valore totale del portafoglio

## SimulationManager

Classe che gestisce l'intera simulazione.

### Metodi Principali

#### `__init__(config)`
Inizializza il gestore della simulazione.

**Parametri:**
- `config`: Dizionario di configurazione

#### `initialize_simulation()`
Inizializza l'ambiente di simulazione.

**Ritorna:**
- `bool`: True se l'inizializzazione ha successo

#### `create_agents(num_agents=5)`
Crea gli agenti di trading.

**Parametri:**
- `num_agents`: Numero di agenti da creare

**Ritorna:**
- `bool`: True se la creazione ha successo

#### `run_simulation()`
Esegue la simulazione.

**Ritorna:**
- `list`: Lista delle transazioni eseguite

## CredentialManager

Classe che gestisce le credenziali in modo sicuro.

### Metodi Principali

#### `__init__(encryption_key=None)`
Inizializza il gestore delle credenziali.

**Parametri:**
- `encryption_key`: Chiave di crittografia (opzionale)

#### `encrypt_credential(credential)`
Cripta una credenziale.

**Parametri:**
- `credential`: Credenziale da criptare

**Ritorna:**
- `str`: Credenziale criptata

#### `decrypt_credential(encrypted_credential)`
Decripta una credenziale.

**Parametri:**
- `encrypted_credential`: Credenziale criptata

**Ritorna:**
- `str`: Credenziale decriptata

## Esempi di Utilizzo

### Inizializzazione di una Simulazione

```python
from market_simulator import SimulationManager
from config import load_config

# Carica la configurazione
config = load_config()

# Crea il gestore della simulazione
sim_manager = SimulationManager(config)

# Inizializza la simulazione
sim_manager.initialize_simulation()

# Crea gli agenti
sim_manager.create_agents(num_agents=3)

# Esegui la simulazione
results = sim_manager.run_simulation()
```

### Gestione delle Credenziali

```python
from config import credential_manager

# Cripta una credenziale
encrypted = credential_manager.encrypt_credential("my_secret_key")

# Decripta una credenziale
decrypted = credential_manager.decrypt_credential(encrypted)

# Salva le credenziali
credentials = {
    "api_key": "my_api_key",
    "secret_key": "my_secret_key"
}
credential_manager.save_credentials(credentials)

# Carica le credenziali
loaded_credentials = credential_manager.load_credentials()
``` 