import os
import sys
import logging
from datetime import datetime

# Aggiungi la directory root al path di Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importazione assoluta di app da dashboard
from dashboard.app import app, websocket_manager

def setup_logging():
    """Configura il logging per il server"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"dashboard_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('DashboardServer')

def main():
    """Funzione principale per l'avvio del server"""
    logger = setup_logging()
    logger.info("Avvio del server della dashboard")
    
    try:
        # Avvia il server con SocketIO
        port = 8081
        host = '0.0.0.0'
        debug = True
        logger.info(f"Avvio del server su {host}:{port} (debug: {debug})")
        websocket_manager.socketio.run(app, host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Errore nell'avvio del server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 