import os
import sys
import signal
import psutil
import logging
from datetime import datetime

def setup_logging():
    """Configura il logging per lo script di arresto"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"dashboard_server_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('DashboardServerStop')

def find_dashboard_processes():
    """Trova i processi della dashboard in esecuzione"""
    dashboard_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Cerca processi Python che eseguono la dashboard
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('dashboard' in arg.lower() for arg in cmdline):
                    dashboard_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return dashboard_processes

def stop_dashboard_processes():
    """Arresta i processi della dashboard"""
    logger = setup_logging()
    logger.info("Ricerca dei processi della dashboard in esecuzione")
    
    processes = find_dashboard_processes()
    
    if not processes:
        logger.info("Nessun processo della dashboard trovato in esecuzione")
        return
    
    logger.info(f"Trovati {len(processes)} processi della dashboard")
    
    for proc in processes:
        try:
            logger.info(f"Arresto del processo {proc.info['pid']}")
            proc.terminate()
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            logger.warning(f"Processo {proc.info['pid']} non terminato entro 5 secondi, forzatura arresto")
            proc.kill()
        except psutil.NoSuchProcess:
            logger.info(f"Processo {proc.info['pid']} già terminato")
        except Exception as e:
            logger.error(f"Errore nell'arresto del processo {proc.info['pid']}: {e}")
    
    logger.info("Arresto dei processi completato")

def main():
    """Funzione principale per l'arresto del server"""
    try:
        stop_dashboard_processes()
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Errore nell'arresto del server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 