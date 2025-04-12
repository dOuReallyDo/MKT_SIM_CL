# Voglio creare un file di configurazione per un sistema di trading algoritmico 
# che mi permetta di configurare i parametri di trading come:
# - simboli di trading
# - timeframe
# - tipo di ordine (compra/vendita)
# - quantità di contratto
# - livelli di take profit e stop loss  
# Per favore implementa questo file di configurazione basandoti sul piano di progetto 
# che si trova nel file piano_progetto.md nella directory principale

import pandas as pd
import json
import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione generale
CONFIG = {
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
        'active_strategy': 'random',  # Cambio da 'neural_network' a 'random'
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
    }
}

class CredentialManager:
    """Gestisce le credenziali in modo sicuro"""
    def __init__(self, encryption_key=None):
        """
        Inizializza il gestore delle credenziali
        
        Args:
            encryption_key: Chiave di crittografia (opzionale)
        """
        self.encryption_key = encryption_key or CONFIG['security']['encryption_key']
        self.cipher_suite = Fernet(self.encryption_key.encode())
    
    def encrypt_credential(self, credential):
        """
        Cripta una credenziale
        
        Args:
            credential: Credenziale da criptare
            
        Returns:
            str: Credenziale criptata
        """
        try:
            return self.cipher_suite.encrypt(credential.encode()).decode()
        except Exception as e:
            raise ValueError(f"Errore nella criptazione della credenziale: {e}")
    
    def decrypt_credential(self, encrypted_credential):
        """
        Decripta una credenziale
        
        Args:
            encrypted_credential: Credenziale criptata
            
        Returns:
            str: Credenziale decriptata
        """
        try:
            return self.cipher_suite.decrypt(encrypted_credential.encode()).decode()
        except Exception as e:
            raise ValueError(f"Errore nella decrittazione della credenziale: {e}")
    
    def save_credentials(self, credentials, file_path='credentials.enc'):
        """
        Salva le credenziali criptate su file
        
        Args:
            credentials: Dizionario di credenziali
            file_path: Percorso del file
        """
        try:
            encrypted_credentials = {
                key: self.encrypt_credential(value)
                for key, value in credentials.items()
            }
            with open(file_path, 'w') as f:
                json.dump(encrypted_credentials, f)
        except Exception as e:
            raise ValueError(f"Errore nel salvataggio delle credenziali: {e}")
    
    def load_credentials(self, file_path='credentials.enc'):
        """
        Carica le credenziali criptate da file
        
        Args:
            file_path: Percorso del file
            
        Returns:
            dict: Dizionario di credenziali decriptate
        """
        try:
            with open(file_path, 'r') as f:
                encrypted_credentials = json.load(f)
            return {
                key: self.decrypt_credential(value)
                for key, value in encrypted_credentials.items()
            }
        except Exception as e:
            raise ValueError(f"Errore nel caricamento delle credenziali: {e}")

def validate_config(config):
    """
    Valida la configurazione del sistema
    
    Args:
        config: Dizionario di configurazione
        
    Returns:
        bool: True se la configurazione è valida, False altrimenti
        
    Raises:
        ValueError: Se la configurazione non è valida
    """
    # Validazione dati di mercato
    if not config['market']['symbols']:
        raise ValueError("La lista dei simboli non può essere vuota")
    
    try:
        start_date = pd.to_datetime(config['market']['start_date'])
        end_date = pd.to_datetime(config['market']['end_date'])
        if start_date >= end_date:
            raise ValueError("La data di inizio deve essere precedente alla data di fine")
    except Exception as e:
        raise ValueError(f"Errore nella validazione delle date: {e}")
    
    # Validazione configurazione trading
    if config['trading']['initial_capital'] <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo")
    
    if not 0 < config['trading']['position_sizing']['max_position_size'] <= 1:
        raise ValueError("La dimensione massima della posizione deve essere tra 0 e 1")
    
    # Validazione gestione del rischio
    risk_config = config['trading']['risk_management']
    if risk_config['use_stop_loss'] and risk_config['stop_loss_percentage'] <= 0:
        raise ValueError("La percentuale di stop loss deve essere positiva")
    if risk_config['use_take_profit'] and risk_config['take_profit_percentage'] <= 0:
        raise ValueError("La percentuale di take profit deve essere positiva")
    
    # Validazione strategie
    if config['strategies']['active_strategy'] not in config['strategies']['available_strategies']:
        raise ValueError("La strategia attiva deve essere presente nelle strategie disponibili")
    
    # Validazione configurazione API
    if not config['api']['alpha_vantage']['api_key']:
        raise ValueError("API key di Alpha Vantage non configurata")
    
    # Validazione configurazione sicurezza
    if not config['security']['encryption_key']:
        raise ValueError("Chiave di crittografia non configurata")
    
    return True

def load_config(config_path='config.json'):
    """
    Carica e valida la configurazione da file
    
    Args:
        config_path: Percorso del file di configurazione
        
    Returns:
        dict: Configurazione validata
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if validate_config(config):
            return config
    except Exception as e:
        raise ValueError(f"Errore nel caricamento della configurazione: {e}")

# Inizializza il gestore delle credenziali
credential_manager = CredentialManager()