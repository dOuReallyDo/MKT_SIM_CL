#!/usr/bin/env python3
"""
Script per avviare la dashboard con controllo delle porte disponibili.
Verifica quali porte sono utilizzate e sceglie una porta disponibile.
Termina anche eventuali istanze precedenti della dashboard.
"""

import os
import sys
import subprocess
import socket
import signal
import psutil
import time
import logging
from logging.handlers import RotatingFileHandler
import webbrowser

# Configura logging
os.makedirs("logs", exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('logs/dashboard.log', maxBytes=10485760, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger('DashboardLauncher')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Porte da evitare (porte comunemente usate o riservate)
PORTE_DA_EVITARE = [
    # Porte di sistema
    1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 77, 79, 80, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 443, 445, 464, 465, 497, 500, 512, 513, 514, 515, 520, 523, 530, 548, 554, 563, 587, 593, 623, 626, 631, 636, 639, 646, 657, 691, 860, 873, 902, 989, 990, 993, 995, 1194,
    # AirPlay Receiver e altri servizi comuni macOS
    554, 5000, 7000, 49152, 49153, 
    # Database e middleware
    1080, 1521, 1433, 3306, 5432, 6379, 8080, 8443, 9092, 27017, 28015, 
    # Porte usate comunemente da altri framework web
    3000, 4200, 4000, 8000, 8008, 8080, 8081, 8888,
    # Altre porte problematiche su macOS
    12865
]

# ID del processo corrente per evitare auto-terminazione
CURRENT_PID = os.getpid()

def trova_porta_disponibile(porta_iniziale=8050, porta_finale=9000):
    """
    Trova una porta disponibile che non è in uso e non è nella lista delle porte da evitare.
    """
    for porta in range(porta_iniziale, porta_finale):
        if porta in PORTE_DA_EVITARE:
            continue
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', porta))
            if result != 0:  # La porta è disponibile
                return porta
                
    raise RuntimeError(f"Nessuna porta disponibile trovata nell'intervallo {porta_iniziale}-{porta_finale}")

def termina_istanze_flask():
    """
    Termina eventuali istanze precedenti di Flask in esecuzione.
    """
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Salta il processo corrente
            if proc.pid == CURRENT_PID:
                logger.debug(f"Saltato il processo corrente: {CURRENT_PID}")
                continue
                
            cmdline = proc.info.get('cmdline', [])
            if not cmdline:
                continue
                
            # Cerca processi Flask o relativi alla dashboard
            if ('flask' in ' '.join(cmdline).lower() or 
                'dashboard.app' in ' '.join(cmdline) or
                ('python' in proc.info['name'].lower() and any('dashboard' in arg.lower() for arg in cmdline))):
                
                logger.info(f"Terminazione processo Flask: PID {proc.pid} - Comando: {' '.join(cmdline)}")
                os.kill(proc.pid, signal.SIGTERM)
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if count > 0:
        logger.info(f"Terminate {count} istanze precedenti della dashboard")
        # Piccola pausa per assicurarsi che le porte vengano rilasciate
        time.sleep(1)
    else:
        logger.info("Nessuna istanza precedente della dashboard trovata")

def crea_directory_necessarie():
    """
    Crea le directory necessarie per l'esecuzione della dashboard.
    """
    dirs = ["logs", "data", "reports", "dashboard/cache", "dashboard/data"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Directory necessarie create con successo")

def verifica_componenti_dashboard():
    """
    Verifica che i componenti necessari della dashboard siano disponibili.
    """
    try:
        # Verifica se il modulo dashboard.app esiste
        sys.path.append(os.getcwd())
        import dashboard.app
        logger.info("Componenti della dashboard verificati con successo")
        return True
    except ImportError as e:
        logger.error(f"Errore nell'importazione dei componenti della dashboard: {e}")
        logger.error(f"Assicurarsi che la struttura della dashboard esista in: {os.path.join(os.getcwd(), 'dashboard')}")
        logger.error(f"Contenuto della directory dashboard: {os.listdir('dashboard') if os.path.exists('dashboard') else 'Directory non trovata'}")
        return False

def avvia_dashboard(porta):
    """
    Avvia la dashboard sulla porta specificata.
    """
    crea_directory_necessarie()
    
    if not verifica_componenti_dashboard():
        logger.error("Impossibile avviare la dashboard: componenti mancanti")
        return False
    
    try:
        logger.info(f"Avvio della dashboard sulla porta {porta}")
        logger.info(f"URL di accesso: http://localhost:{porta}")
        
        # Impostazione delle variabili d'ambiente
        env = os.environ.copy()
        env["FLASK_APP"] = "dashboard.app"
        env["FLASK_DEBUG"] = "1" 
        env["DASHBOARD_PORT"] = str(porta)
        
        # Avvio di Flask come subprocess per poter monitorare e terminare in modo pulito
        cmd = [sys.executable, "-m", "flask", "run", "--host=0.0.0.0", f"--port={porta}"]
        
        # Esegui il comando con output visibile
        logger.info(f"Esecuzione comando: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, env=env)
        
        # Attendi un momento per permettere a Flask di avviarsi
        time.sleep(2)
        
        # Apri il browser se il processo è ancora in esecuzione
        if process.poll() is None:
            url = f"http://localhost:{porta}"
            logger.info(f"Dashboard avviata con successo. Apro il browser a {url}")
            webbrowser.open(url)
            
            try:
                # Attendi che il processo termini
                process.wait()
            except KeyboardInterrupt:
                logger.info("Interruzione rilevata, termino la dashboard...")
                process.terminate()
                process.wait(timeout=5)
        else:
            exit_code = process.returncode
            logger.error(f"La dashboard si è arrestata con codice {exit_code}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Errore durante l'avvio della dashboard: {e}")
        return False

def controlla_porte_in_uso():
    """
    Mostra le porte principali già in uso sul sistema.
    """
    logger.info("Controllo delle porte in uso sul sistema...")
    
    porte_in_uso = []
    for porta in [5000, 7000, 8000, 8050, 8080, 9000]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', porta))
                if result == 0:  # La porta è in uso
                    porte_in_uso.append(porta)
        except:
            pass
    
    if porte_in_uso:
        logger.info(f"Porte comuni già in uso: {', '.join(map(str, porte_in_uso))}")
    else:
        logger.info("Nessuna delle porte comuni risulta in uso")
    
    return porte_in_uso

if __name__ == "__main__":
    try:
        logger.info("===============================================")
        logger.info("Avvio Dashboard Launcher")
        logger.info("===============================================")
        
        # Controlla le porte in uso
        controlla_porte_in_uso()
        
        # Assicurati che non ci siano istanze precedenti in esecuzione
        termina_istanze_flask()
        
        # Trova una porta disponibile
        porta = trova_porta_disponibile()
        logger.info(f"Porta disponibile trovata: {porta}")
        
        # Avvia la dashboard
        avvia_dashboard(porta)
    except Exception as e:
        logger.error(f"Errore nell'esecuzione dello script: {e}")
        sys.exit(1) 