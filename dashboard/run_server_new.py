#!/usr/bin/env python3
"""
Script per avviare il server dashboard
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Aggiungi la directory root al path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# Configura il logging
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"dashboard_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DashboardServer')

def main():
    try:
        # Importa l'app Flask
        from flask import Flask
        from flask_socketio import SocketIO
        import pandas as pd
        import numpy as np
        import os
        import json
        from datetime import datetime, timedelta
        import sys
        import argparse
        
        # Importa il modulo app definendo direttamente le variabili necessarie
        from dashboard.app import app
        
        logger.info("Avvio del server dashboard sulla porta 8081")
        
        # Avvia il server
        app.run(host='0.0.0.0', port=8081, debug=True)
    except Exception as e:
        logger.error(f"Errore nell'avvio del server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 