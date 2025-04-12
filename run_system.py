#!/usr/bin/env python3
"""
Script per eseguire il sistema completo MKT_SIM_CL.

Questo script permette di eseguire il sistema completo con tutti i moduli
integrati: simulazione del mercato, strategie di trading, reti neurali e dashboard.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import webbrowser
import threading
import time
import subprocess

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/system_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SystemRunner')

# Import dei moduli del sistema
from data.collector import DataCollector
from market_simulator.simulation import SimulationManager
from market_simulator.environment import MarketEnvironment
from market_simulator.agents import TradingAgent
from trading_strategy import create_strategy, get_available_strategies

def parse_arguments():
    """
    Analizza gli argomenti da linea di comando
    
    Returns:
        Namespace con gli argomenti
    """
    parser = argparse.ArgumentParser(description='Sistema di simulazione del mercato azionario')
    parser.add_argument('--mode', choices=['simulation', 'dashboard', 'all'], 
                        default='all', help='Modalità di esecuzione')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                        help='Simboli da utilizzare per la simulazione')
    parser.add_argument('--days', type=int, default=60,
                        help='Numero di giorni per la simulazione (partendo da oggi, all\'indietro)')
    parser.add_argument('--strategy', choices=list(get_available_strategies().keys()), 
                        default='random', help='Strategia di trading da utilizzare')
    parser.add_argument('--num-agents', type=int, default=5,
                        help='Numero di agenti per la simulazione')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Capitale iniziale per gli agenti')
    parser.add_argument('--dashboard-port', type=int, default=8080,
                        help='Porta per la dashboard web')
    parser.add_argument('--no-browser', action='store_true',
                        help='Non aprire automaticamente il browser per la dashboard')
    
    return parser.parse_args()

def load_or_create_config(args):
    """
    Carica la configurazione o ne crea una nuova
    
    Args:
        args: Argomenti da linea di comando
        
    Returns:
        Dict: Configurazione del sistema
    """
    config_file = 'config_user.json'
    
    # Date per la simulazione
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Configurazione predefinita
    config = {
        'market': {
            'symbols': args.symbols,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        },
        'trading': {
            'initial_capital': args.initial_capital,
            'strategy': args.strategy
        },
        'strategies': {
            'active_strategy': args.strategy,
            'available_strategies': list(get_available_strategies().keys()),
            'strategy_params': {
                'mean_reversion': {
                    'window': 20
                },
                'trend_following': {
                    'short_window': 10,
                    'long_window': 50
                },
                'neural_network': {
                    'model_type': 'lstm',
                    'sequence_length': 10
                }
            }
        },
        'dashboard': {
            'port': args.dashboard_port,
            'open_browser': not args.no_browser
        }
    }
    
    # Salva la configurazione
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configurazione salvata in {config_file}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio della configurazione: {e}")
    
    return config

def prepare_data(config):
    """
    Prepara i dati per la simulazione
    
    Args:
        config: Configurazione del sistema
        
    Returns:
        Dict: Dati di mercato
    """
    logger.info("Preparazione dei dati...")
    
    # Crea il collector
    collector = DataCollector()
    
    # Verifica integrità dei dati
    collector.verify_data_integrity()
    
    # Ottieni i dati per ogni simbolo
    market_data = {}
    for symbol in config['market']['symbols']:
        logger.info(f"Caricamento dati per {symbol}...")
        df = collector.get_stock_data(
            symbol, 
            config['market']['start_date'], 
            config['market']['end_date']
        )
        
        if df is not None and not df.empty:
            market_data[symbol] = df
            logger.info(f"Dati caricati per {symbol}: {len(df)} righe")
        else:
            logger.warning(f"Nessun dato disponibile per {symbol}")
    
    if not market_data:
        logger.error("Nessun dato disponibile per la simulazione")
        sys.exit(1)
    
    logger.info(f"Preparazione dei dati completata: {len(market_data)} simboli")
    return market_data

def run_simulation(config):
    """
    Esegue la simulazione di mercato
    
    Args:
        config: Configurazione del sistema
        
    Returns:
        Dict: Risultati della simulazione
    """
    logger.info("Avvio della simulazione...")
    
    # Crea il gestore di simulazione
    sim_manager = SimulationManager(config)
    
    # Inizializza la simulazione
    if not sim_manager.initialize_simulation():
        logger.error("Errore nell'inizializzazione della simulazione")
        return None
    
    # Crea gli agenti
    if not sim_manager.create_agents(num_agents=config.get('num_agents', 5)):
        logger.error("Errore nella creazione degli agenti")
        return None
    
    # Esegui la simulazione
    logger.info("Esecuzione simulazione...")
    transactions = sim_manager.run_simulation()
    
    if transactions is None:
        logger.error("Errore nell'esecuzione della simulazione")
        return None
    
    # Ottieni il riepilogo delle transazioni
    summary = sim_manager.get_transactions_summary()
    logger.info(f"Riepilogo transazioni: {summary}")
    
    # Ottieni le performance degli agenti
    performances = sim_manager.get_agents_performance()
    logger.info(f"Performance agenti: {performances}")
    
    # Salva i risultati
    sim_manager.save_results()
    
    logger.info("Simulazione completata con successo")
    
    # Restituisci i risultati
    return {
        'transactions': transactions,
        'summary': summary,
        'performances': performances
    }

def run_dashboard(config):
    """
    Avvia la dashboard
    
    Args:
        config: Configurazione del sistema
    """
    logger.info("Avvio della dashboard...")
    
    # Salva lo stato della dashboard
    dashboard_state = {
        'last_simulation': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'active_tab': 'overview',
        'auto_refresh': True,
        'refresh_interval': 5
    }
    
    dashboard_state_file = 'dashboard_state.json'
    with open(dashboard_state_file, 'w') as f:
        json.dump(dashboard_state, f, indent=4)
    
    # Avvia la dashboard in un processo separato
    port = config.get('dashboard', {}).get('port', 8080)
    
    # Avvia uno script separato per gestire il server
    run_dashboard_script = 'run_dashboard.py'
    
    cmd = [sys.executable, run_dashboard_script, '--port', str(port)]
    
    try:
        process = subprocess.Popen(cmd)
        logger.info(f"Dashboard avviata con PID {process.pid}")
        
        # Attendi che il server sia pronto
        time.sleep(3)
        
        # Apri il browser se richiesto
        if config.get('dashboard', {}).get('open_browser', True):
            url = f"http://localhost:{port}"
            logger.info(f"Apertura del browser: {url}")
            webbrowser.open(url)
        
        return process
    except Exception as e:
        logger.error(f"Errore nell'avvio della dashboard: {e}")
        return None

def main():
    """Funzione principale"""
    # Analizza gli argomenti
    args = parse_arguments()
    
    # Carica o crea la configurazione
    config = load_or_create_config(args)
    
    # Crea le directory necessarie
    for dir_path in ['data', 'logs', 'reports', 'models']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Esecuzione in base alla modalità
    dashboard_process = None
    
    try:
        if args.mode in ['simulation', 'all']:
            # Prepara i dati
            market_data = prepare_data(config)
            
            # Esegui la simulazione
            simulation_results = run_simulation(config)
            
            if simulation_results is None:
                logger.error("La simulazione non ha prodotto risultati")
                if args.mode == 'simulation':
                    return 1
        
        if args.mode in ['dashboard', 'all']:
            # Avvia la dashboard
            dashboard_process = run_dashboard(config)
            
            if dashboard_process is None:
                logger.error("Errore nell'avvio della dashboard")
                return 1
            
            # Se siamo in modalità dashboard, attendi per evitare che il programma termini
            if args.mode == 'dashboard':
                logger.info("Dashboard in esecuzione. Premi Ctrl+C per terminare.")
                while True:
                    time.sleep(1)
        
        if args.mode == 'all':
            logger.info("Sistema completo in esecuzione. Premi Ctrl+C per terminare.")
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interruzione richiesta dall'utente")
    finally:
        # Pulizia
        if dashboard_process is not None:
            logger.info(f"Terminazione del processo dashboard (PID {dashboard_process.pid})")
            try:
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)
            except:
                logger.warning("Impossibile terminare normalmente il processo dashboard")
                try:
                    dashboard_process.kill()
                except:
                    pass
    
    logger.info("Esecuzione completata")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 