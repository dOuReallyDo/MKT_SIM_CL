"""
Configuration Package for the Market Simulation Application.

Questo pacchetto contiene la configurazione del sistema, inclusa la configurazione
di base, la configurazione dell'utente e la configurazione del monitoraggio.
"""

from .monitoring_config import *
from .base_config import BASE_CONFIG
from .user_config import get_config, update_config, reset_config

# Configurazione predefinita - combinazione della configurazione di base e dell'utente
CONFIG = get_config() 