"""
User Configuration Module.

Questo modulo gestisce la configurazione dell'utente, permettendo di sovrascrivere
i valori della configurazione di base.
"""

import os
import json
import logging
from config.base_config import BASE_CONFIG

# Configurazione del logger
logger = logging.getLogger('UserConfig')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Percorso del file di configurazione dell'utente
USER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config_user.json')

def deep_update(d, u):
    """Aggiorna ricorsivamente un dizionario."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def load_user_config():
    """
    Carica la configurazione dell'utente dal file.
    
    Returns:
        dict: Configurazione dell'utente o configurazione vuota se il file non esiste
    """
    try:
        if os.path.exists(USER_CONFIG_PATH):
            with open(USER_CONFIG_PATH, 'r') as f:
                return json.load(f)
        else:
            logger.info(f"File di configurazione utente non trovato: {USER_CONFIG_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Errore nel caricamento della configurazione utente: {e}")
        return {}

def save_user_config(config):
    """
    Salva la configurazione dell'utente in un file.
    
    Args:
        config: Dizionario di configurazione
    
    Returns:
        bool: True se il salvataggio è riuscito, False altrimenti
    """
    try:
        with open(USER_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Errore nel salvataggio della configurazione utente: {e}")
        return False

def get_config():
    """
    Ottiene la configurazione completa combinando la configurazione di base con quella dell'utente.
    
    Returns:
        dict: Configurazione completa
    """
    config = BASE_CONFIG.copy()
    user_config = load_user_config()
    
    if user_config:
        config = deep_update(config, user_config)
    
    return config

def update_config(partial_config):
    """
    Aggiorna la configurazione utente con nuovi valori e salva il file.
    
    Args:
        partial_config: Dizionario parziale di configurazione da aggiornare
    
    Returns:
        dict: Configurazione completa aggiornata
    """
    user_config = load_user_config()
    deep_update(user_config, partial_config)
    
    if save_user_config(user_config):
        logger.info("Configurazione utente aggiornata e salvata")
    else:
        logger.warning("Impossibile salvare la configurazione utente")
    
    # Restituisci la configurazione completa
    config = BASE_CONFIG.copy()
    deep_update(config, user_config)
    
    return config

def reset_config():
    """
    Resetta la configurazione utente ai valori predefiniti.
    
    Returns:
        dict: Configurazione di base
    """
    try:
        if os.path.exists(USER_CONFIG_PATH):
            os.remove(USER_CONFIG_PATH)
            logger.info("Configurazione utente resettata ai valori predefiniti")
        return BASE_CONFIG.copy()
    except Exception as e:
        logger.error(f"Errore nel reset della configurazione utente: {e}")
        return BASE_CONFIG.copy() 