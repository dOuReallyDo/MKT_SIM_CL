#!/usr/bin/env python3
"""
Script temporaneo per il debug
"""
import os
import sys
import logging
from pathlib import Path

# Aggiungi la directory root al path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# Configura il logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('temp_debug')

try:
    # Prova a importare l'app
    logger.info("Tentativo di importare l'app Flask")
    from dashboard.app import app
    logger.info("Importazione riuscita!")
    
    # Stampa tutte le route disponibili
    logger.info("Route disponibili:")
    for rule in app.url_map.iter_rules():
        logger.info(f"{rule.endpoint}: {rule.rule}")
    
except Exception as e:
    logger.error(f"Errore nell'importazione: {e}")
    import traceback
    traceback.print_exc() 