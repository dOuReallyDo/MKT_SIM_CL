"""
Base Configuration Module.

Questo modulo contiene la configurazione di base del sistema.
"""

import os
from datetime import datetime
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione generale
BASE_CONFIG = {
    # Dati di mercato
    'market': {
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],  # Simboli da tradare
        'start_date': '2023-01-01',  # Data di inizio simulazione
        'end_date': '2023-01-31',    # Data di fine simulazione
        'timeframes': ['1d', '1h'],   # Timeframe disponibili ('1d' = giornaliero, '1h' = orario)
        'default_timeframe': '1d',    # Timeframe predefinito
    },
    
    # Configurazione trading
    'trading': {
        'initial_capital': 100000,    # Capitale iniziale in USD
        'transaction_fee': 0.001,     # Commissione di transazione (0.1%)
        'order_types': ['market', 'limit', 'stop'],  # Tipi di ordine supportati
        'default_order_type': 'market',  # Tipo di ordine predefinito
        'position_sizing': {
            'default_quantity': 10,   # Quantità di contratti predefinita
            'max_position_size': 0.2,  # Massima percentuale del capitale per posizione (20%)
        },
        'risk_management': {
            'use_stop_loss': True,     # Utilizzare stop loss
            'stop_loss_percentage': 2.0,  # Percentuale di stop loss (2%)
            'use_take_profit': True,    # Utilizzare take profit
            'take_profit_percentage': 5.0,  # Percentuale di take profit (5%)
            'max_daily_loss': 1000,     # Massima perdita giornaliera in USD
        },
    },
    
    # Configurazione strategie
    'strategies': {
        'active_strategy': 'random',  # Strategia attiva
        'available_strategies': [
            'random',              # Strategia casuale
            'mean_reversion',      # Strategia di mean reversion
            'trend_following',     # Strategia di trend following
            'value_investing',     # Strategia di value investing
            'neural_network',      # Strategia basata su rete neurale
        ],
        'strategy_params': {
            'mean_reversion': {
                'window': 20,      # Finestra per la media mobile
            },
            'trend_following': {
                'short_window': 10,  # Finestra breve per media mobile
                'long_window': 50,   # Finestra lunga per media mobile
            },
            'neural_network': {
                'model_type': 'lstm',  # Tipo di modello (lstm, cnn, transformer)
                'sequence_length': 10,  # Lunghezza della sequenza per previsione
            },
        },
    },
    
    # Percorsi file
    'paths': {
        'data_dir': './data',           # Directory dei dati
        'models_dir': './models',       # Directory dei modelli
        'logs_dir': './logs',           # Directory dei log
        'reports_dir': './reports',     # Directory dei report
    },
    
    # Configurazione API e credenziali
    'api': {
        'alpha_vantage': {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'base_url': 'https://www.alphavantage.co/query'
        },
        'yfinance': {
            'use_proxy': False,
            'proxy_url': os.getenv('PROXY_URL', '')
        }
    },
    
    # Configurazione sicurezza
    'security': {
        'encryption_key': os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode()),
        'use_ssl': True,
        'verify_certificates': True
    },
    
    # Configurazione dashboard
    'dashboard': {
        'host': '0.0.0.0',           # Host su cui avviare la dashboard
        'port': 8081,                 # Porta su cui avviare la dashboard
        'debug': False,               # Modalità debug
        'template_dir': 'dashboard/templates',  # Directory dei template
        'static_dir': 'dashboard/static',       # Directory dei file statici
    },
    
    # Configurazione reti neurali
    'neural_network': {
        'default_model': 'lstm',      # Modello predefinito
        'training_epochs': 100,       # Numero di epoche per l'addestramento
        'batch_size': 32,             # Dimensione del batch
        'learning_rate': 0.001,       # Learning rate
        'early_stopping': True,       # Utilizzare early stopping
        'patience': 10,               # Pazienza per early stopping
        'validation_split': 0.2,      # Percentuale di dati per validazione
        'test_split': 0.1,            # Percentuale di dati per test
        'random_seed': 42,            # Seed per riproducibilità
    },
    
    # Configurazione self-play
    'self_play': {
        'enabled': False,             # Abilita il self-play
        'generations': 10,            # Numero di generazioni
        'population_size': 20,        # Dimensione della popolazione
        'mutation_rate': 0.1,         # Tasso di mutazione
        'tournament_size': 5,         # Dimensione del torneo
        'elitism': True,              # Utilizzare elitismo
        'elite_size': 2,              # Numero di elite da preservare
    }
}

def get_config():
    """
    Restituisce la configurazione di base
    
    Returns:
        dict: Configurazione di base
    """
    return BASE_CONFIG.copy() 