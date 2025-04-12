#!/usr/bin/env python3
"""
Script per l'avvio della dashboard.

Questo script avvia la dashboard web per la gestione del sistema di trading algoritmico.
"""

import os
import sys
import time
import logging
import subprocess
import signal
import psutil
import argparse

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard_launcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DashboardLauncher')

def find_flask_processes():
    """
    Trova tutti i processi Flask in esecuzione.
    
    Returns:
        list: Lista di processi Flask
    """
    flask_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('dashboard/app.py' in arg for arg in cmdline):
                flask_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return flask_processes

def stop_flask_processes():
    """
    Ferma tutti i processi Flask in esecuzione.
    
    Returns:
        int: Numero di processi fermati
    """
    count = 0
    for proc in find_flask_processes():
        try:
            os.kill(proc.info['pid'], signal.SIGTERM)
            count += 1
            logger.info(f"Processo Flask fermato (PID: {proc.info['pid']})")
        except (psutil.NoSuchProcess, PermissionError):
            logger.warning(f"Impossibile fermare il processo con PID {proc.info['pid']}")
    
    if count == 0:
        logger.info("Nessun processo Flask trovato in esecuzione.")
    
    return count

def start_dashboard(port=8081, host='127.0.0.1', debug=False):
    """
    Avvia la dashboard.
    
    Args:
        port: Porta su cui avviare la dashboard
        host: Host su cui avviare la dashboard
        debug: Attiva la modalità debug
    
    Returns:
        subprocess.Popen: Processo Flask
    """
    # Ferma eventuali processi Flask in esecuzione
    stop_flask_processes()
    
    # Cambia directory nella cartella principale del progetto
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Assicurati che le directory necessarie esistano
    for dir_path in ['logs', 'data', 'reports', 'models']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Verifica che il file app.py esista
    app_path = os.path.join(project_root, 'dashboard', 'app.py')
    if not os.path.exists(app_path):
        logger.error(f"File app.py non trovato nel percorso: {app_path}")
        return None
    
    # Aggiungi la directory principale al PYTHONPATH
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
    
    # Prepara il comando
    cmd = [sys.executable, app_path]
    if port:
        cmd.extend(['--port', str(port)])
    if host:
        cmd.extend(['--host', str(host)])
    if debug:
        cmd.append('--debug')
    
    # Avvia il processo
    try:
        logger.info(f"Avvio della dashboard con comando: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Attendere un po' per vedere se il processo si avvia correttamente
        time.sleep(2)
        
        # Verifica che il processo sia ancora in esecuzione
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"La dashboard si è arrestata. Errore:\n{stderr}\nOutput:\n{stdout}")
            return None
        
        logger.info(f"Dashboard avviata (PID: {process.pid})")
        logger.info(f"Accedi alla dashboard all'indirizzo: http://{host}:{port}")
        return process
    except Exception as e:
        logger.error(f"Errore nell'avvio della dashboard: {e}")
        return None

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='Avvia la dashboard del sistema di trading algoritmico')
    parser.add_argument('--port', type=int, default=8081, help='Porta su cui avviare la dashboard')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host su cui avviare la dashboard')
    parser.add_argument('--debug', action='store_true', help='Attiva la modalità debug')
    parser.add_argument('--stop', action='store_true', help='Ferma la dashboard')
    
    args = parser.parse_args()
    
    if args.stop:
        stop_flask_processes()
    else:
        process = start_dashboard(port=args.port, host=args.host, debug=args.debug)
        
        if process:
            # Leggi l'output del processo in background
            def read_output():
                while True:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(f"[DASHBOARD] {stdout_line.strip()}")
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        print(f"[DASHBOARD ERROR] {stderr_line.strip()}")
                    if process.poll() is not None:
                        break
            
            import threading
            thread = threading.Thread(target=read_output)
            thread.daemon = True
            thread.start()
            
            # Mantieni il processo in esecuzione
            try:
                while process.poll() is None:
                    time.sleep(1)
                
                # Se siamo qui, il processo è terminato
                return_code = process.returncode
                logger.error(f"La dashboard è terminata con codice: {return_code}")
            except KeyboardInterrupt:
                logger.info("Dashboard arrestata dall'utente.")
                stop_flask_processes()

if __name__ == '__main__':
    main() 