from typing import Dict, Any, Optional
import json
import os
from datetime import datetime
import logging

class DashboardStateManager:
    """Gestisce lo stato globale della dashboard"""
    
    def __init__(self):
        """Inizializza il gestore dello stato"""
        # Configurazione del logging
        self.logger = logging.getLogger('DashboardStateManager')
        self.logger.setLevel(logging.INFO)
        
        # Se non ci sono handler, aggiungine uno
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Percorso del file di stato
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dashboard_state.json')
        
        # Crea la directory data se non esiste
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        # Carica lo stato
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Carica lo stato da file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.logger.info(f"Stato caricato da {self.state_file}")
                return state
            else:
                self.logger.info("Nessuno stato precedente trovato, creazione nuovo stato")
                return self._create_default_state()
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dello stato: {e}")
            return self._create_default_state()
    
    def _create_default_state(self):
        """Crea uno stato predefinito"""
        default_state = {
            'data_collection': {
                'last_update': None,
                'symbols': [],
                'start_date': None,
                'end_date': None
            },
            'market_simulation': {
                'last_update': None,
                'status': 'idle',
                'results': None
            },
            'neural_network': {
                'last_update': None,
                'status': 'idle',
                'model_info': None
            },
            'self_play': {
                'last_update': None,
                'status': 'idle',
                'results': None
            },
            'predictions': {
                'last_update': None,
                'status': 'idle',
                'results': None
            },
            'wizard': {
                'last_update': None,
                'status': 'idle',
                'current_step': 0,
                'steps_completed': [],
                'config': None
            }
        }
        self._save_state(default_state)
        return default_state
    
    def _save_state(self, state):
        """Salva lo stato su file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            self.logger.info(f"Stato salvato in {self.state_file}")
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio dello stato: {e}")
    
    def get_dashboard_state(self):
        """Recupera lo stato completo della dashboard"""
        return self.state
    
    def get_tab_state(self, tab_name: str) -> Dict[str, Any]:
        """Recupera lo stato di una specifica tab"""
        return self.state.get(tab_name, {})
    
    def update_tab_state(self, tab_name: str, data: Dict[str, Any]):
        """Aggiorna lo stato di una specifica tab"""
        if tab_name in self.state:
            self.state[tab_name].update(data)
            self.state[tab_name]['last_update'] = datetime.now().isoformat()
            self._save_state(self.state)
            self.logger.info(f"Stato della tab {tab_name} aggiornato")
        else:
            self.logger.warning(f"Tab {tab_name} non trovata nello stato")
    
    def get_data_collection_state(self) -> Dict[str, Any]:
        """Recupera lo stato del modulo di raccolta dati"""
        return self.get_tab_state('data_collection')
    
    def update_data_collection_state(self, new_state: Dict[str, Any]):
        """Aggiorna lo stato del modulo di raccolta dati"""
        self.update_tab_state('data_collection', new_state)
    
    def get_market_simulation_state(self) -> Dict[str, Any]:
        """Recupera lo stato del modulo di simulazione di mercato"""
        return self.get_tab_state('market_simulation')
    
    def update_market_simulation_state(self, new_state: Dict[str, Any]):
        """Aggiorna lo stato del modulo di simulazione di mercato"""
        self.update_tab_state('market_simulation', new_state)
    
    def get_neural_network_state(self) -> Dict[str, Any]:
        """Recupera lo stato del modulo di rete neurale"""
        return self.get_tab_state('neural_network')
    
    def update_neural_network_state(self, new_state: Dict[str, Any]):
        """Aggiorna lo stato del modulo di rete neurale"""
        self.update_tab_state('neural_network', new_state)
    
    def get_self_play_state(self) -> Dict[str, Any]:
        """Recupera lo stato del modulo di self-play"""
        return self.get_tab_state('self_play')
    
    def update_self_play_state(self, new_state: Dict[str, Any]):
        """Aggiorna lo stato del modulo di self-play"""
        self.update_tab_state('self_play', new_state)
    
    def get_prediction_state(self) -> Dict[str, Any]:
        """Recupera lo stato del modulo di previsione"""
        return self.get_tab_state('predictions')
    
    def update_prediction_state(self, new_state: Dict[str, Any]):
        """Aggiorna lo stato del modulo di previsione"""
        self.update_tab_state('predictions', new_state)
    
    def get_wizard_state(self) -> Dict[str, Any]:
        """Recupera lo stato della wizard"""
        return self.get_tab_state('wizard')
    
    def update_wizard_state(self, new_state: Dict[str, Any]):
        """Aggiorna lo stato della wizard"""
        self.update_tab_state('wizard', new_state)
    
    def clear_state(self):
        """Cancella lo stato corrente"""
        self.state = {}
        self.save_state()
        self.logger.info("Stato cancellato")
