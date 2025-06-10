#!/usr/bin/env python3
"""
Script di test per verificare il funzionamento della simulazione.
Questo script esegue una simulazione di base senza dipendere dalla dashboard.
"""

import os
import sys
import logging
import json
from datetime import datetime
import time

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestSimulation')

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importazioni
from market_simulator.simulation import SimulationManager
from market_simulator.monitored_simulation import MonitoredSimulationManager
from dashboard.real_time_monitor import RealTimeMonitor
from dashboard.state_manager import DashboardStateManager
from dashboard.websocket_manager import WebSocketManager

# Classe di test WebSocketManager per intercettare gli aggiornamenti
class DebugWebSocketManager:
    def __init__(self):
        self.updates = []
        self.logger = logging.getLogger('DebugWebSocketManager')
    
    def emit_market_simulation_update(self, data):
        print(f"\n=== DEBUG - DebugWebSocketManager.emit_market_simulation_update ===")
        print(f"Status: {data.get('status', 'N/A')}")
        print(f"Progress: {data.get('progress', 'N/A')}")
        print(f"Current day: {data.get('current_day', 'N/A')}")
        print(f"Agents: {len(data.get('agents', []))} agenti")
        print(f"Transactions: {data.get('transactions_count', 0)} transazioni")
        
        self.updates.append({
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
        print(f"DEBUG - Update #{len(self.updates)} ricevuto.")
    
    def emit_update(self, event, data):
        print(f"DEBUG - DebugWebSocketManager.emit_update(event={event})")
        
    def emit_error(self, error_message, module):
        print(f"DEBUG - DebugWebSocketManager.emit_error(msg={error_message}, module={module})")


# Classe di test StateManager per intercettare gli aggiornamenti
class DebugStateManager:
    def __init__(self):
        self.state = {}
        self.logger = logging.getLogger('DebugStateManager')
    
    def update_market_simulation_state(self, state):
        print(f"DEBUG - DebugStateManager.update_market_simulation_state(). Status: {state.get('status', 'N/A')}")
        self.state = state
    
    def get_dashboard_state(self):
        return {'simulation': self.state}


def run_test():
    """Esegue il test della simulazione"""
    print("\n===== TEST SIMULAZIONE INIZIALIZZATO =====\n")
    
    # Carica la configurazione
    try:
        with open('config_updated.json', 'r') as f:
            config = json.load(f)
        print("CONFIG caricata con successo.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Errore nel caricamento della configurazione: {e}")
        # Config di base
        config = {
            "market": {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "start_date": "2023-01-01",
                "end_date": "2023-03-31",
                "interval": "1d"
            },
            "trading": {
                "strategy": "random",
                "initial_capital": 10000,
                "max_agents": 5
            },
            "strategies": {
                "active_strategy": "random",
                "strategy_params": {}
            }
        }
        print("Usando config di default.")
    
    # Imposta parametri di test
    num_agents = 3
    strategy = "random"  # Assicurati che questa strategia sia implementata
    
    # Crea i manager
    websocket_manager = DebugWebSocketManager()
    state_manager = DebugStateManager()
    real_time_monitor = RealTimeMonitor(websocket_manager, state_manager)
    
    # Crea il simulation manager
    simulator = MonitoredSimulationManager(config, real_time_monitor)
    
    print("\n===== INIZIALIZZAZIONE SIMULAZIONE =====\n")
    # Inizializza la simulazione
    if not simulator.initialize_simulation():
        print("ERRORE: Fallita inizializzazione della simulazione.")
        return False
    
    print("\n===== CREAZIONE AGENTI =====\n")
    # Crea agenti
    if not simulator.create_agents(num_agents):
        print("ERRORE: Fallita creazione degli agenti.")
        return False
    
    print("\n===== AVVIO SIMULAZIONE =====\n")
    # Esegui la simulazione (in thread principale per debug)
    results = simulator.run_simulation()
    
    print("\n===== RISULTATI SIMULAZIONE =====\n")
    if results:
        print(f"Simulazione completata con {len(results.get('agents', []))} agenti e {len(results.get('transactions', []))} transazioni.")
        print(f"Totale giorni simulati: {len(results.get('daily_data', []))}")
    else:
        print("ERRORE: La simulazione ha fallito o è stata interrotta.")
    
    print("\n===== CONTROLLO AGGIORNAMENTI EMESSI =====\n")
    print(f"Totale aggiornamenti emessi: {len(websocket_manager.updates)}")
    if len(websocket_manager.updates) > 0:
        first_update = websocket_manager.updates[0]['data']
        last_update = websocket_manager.updates[-1]['data']
        print(f"Primo aggiornamento status: {first_update.get('status', 'N/A')}")
        print(f"Ultimo aggiornamento status: {last_update.get('status', 'N/A')}")
    
    return True


if __name__ == '__main__':
    run_test()
