from flask_socketio import SocketIO, emit
from typing import Dict, Any, List, Callable
import logging
from datetime import datetime

class WebSocketManager:
    """Gestisce le comunicazioni WebSocket per gli aggiornamenti in tempo reale"""
    
    def __init__(self, app):
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self.subscribers: Dict[str, List[Callable]] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura il logging per il gestore WebSocket"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/websocket_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('WebSocketManager')
    
    def subscribe(self, event: str, callback: Callable):
        """Aggiunge un subscriber per un evento specifico"""
        if event not in self.subscribers:
            self.subscribers[event] = []
        self.subscribers[event].append(callback)
        self.logger.info(f"Nuovo subscriber aggiunto per l'evento {event}")
    
    def unsubscribe(self, event: str, callback: Callable):
        """Rimuove un subscriber per un evento specifico"""
        if event in self.subscribers and callback in self.subscribers[event]:
            self.subscribers[event].remove(callback)
            self.logger.info(f"Subscriber rimosso per l'evento {event}")
    
    def emit_update(self, event: str, data: Any):
        """Emette un aggiornamento a tutti i subscriber"""
        try:
            self.socketio.emit(event, {
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
            
            if event in self.subscribers:
                for callback in self.subscribers[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Errore nell'esecuzione del callback per {event}: {e}")
            
            self.logger.info(f"Evento {event} emesso con successo")
        except Exception as e:
            self.logger.error(f"Errore nell'emissione dell'evento {event}: {e}")
    
    def emit_data_collection_update(self, data: Any):
        """Emette un aggiornamento per il modulo di raccolta dati"""
        self.emit_update('data_collection_update', data)
    
    def emit_market_simulation_update(self, data: Any):
        """Emette un aggiornamento per il modulo di simulazione di mercato"""
        self.emit_update('market_simulation_update', data)
    
    def emit_neural_network_update(self, data: Any):
        """Emette un aggiornamento per il modulo di rete neurale"""
        self.emit_update('neural_network_update', data)
    
    def emit_self_play_update(self, data: Any):
        """Emette un aggiornamento per il modulo di self-play"""
        self.emit_update('self_play_update', data)
    
    def emit_prediction_update(self, data: Any):
        """Emette un aggiornamento per il modulo di previsione"""
        self.emit_update('prediction_update', data)
    
    def emit_error(self, error_message: str, module: str):
        """Emette un evento di errore"""
        self.emit_update('error', {
            'message': error_message,
            'module': module,
            'timestamp': datetime.now().isoformat()
        })
        self.logger.error(f"Errore nel modulo {module}: {error_message}")
    
    def emit_progress(self, module: str, progress: float, message: str = ""):
        """Emette un aggiornamento di progresso"""
        self.emit_update('progress', {
            'module': module,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        self.logger.info(f"Progresso nel modulo {module}: {progress}% - {message}")
        
    def emit_dashboard_state(self, state: Dict[str, Any]):
        """Emette l'intero stato della dashboard ai client connessi"""
        try:
            self.socketio.emit('dashboard_state', {
                'state': state,
                'timestamp': datetime.now().isoformat()
            })
            self.logger.info("Stato della dashboard emesso con successo")
        except Exception as e:
            self.logger.error(f"Errore nell'emissione dello stato della dashboard: {e}")
    
    def emit_tab_state(self, client_sid: str, tab_name: str, state: Dict[str, Any]):
        """Emette lo stato di una specifica tab a un client specifico"""
        try:
            self.socketio.emit('tab_state', {
                'tab': tab_name,
                'state': state,
                'timestamp': datetime.now().isoformat()
            }, room=client_sid)
            self.logger.info(f"Stato della tab {tab_name} emesso al client {client_sid}")
        except Exception as e:
            self.logger.error(f"Errore nell'emissione dello stato della tab {tab_name} al client {client_sid}: {e}")
    
    def subscribe_client(self, client_sid: str, tab_name: str):
        """Sottoscrive un client a una tab specifica"""
        # Qui potremmo aggiungere il client a una stanza specifica per la tab
        self.logger.info(f"Client {client_sid} sottoscritto alla tab {tab_name}")
    
    def unsubscribe_client(self, client_sid: str, tab_name: str):
        """Annulla la sottoscrizione di un client da una tab specifica"""
        # Qui potremmo rimuovere il client da una stanza specifica per la tab
        self.logger.info(f"Client {client_sid} non più sottoscritto alla tab {tab_name}") 