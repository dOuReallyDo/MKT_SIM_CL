"""
Monitored Simulation Manager Module.

Questo modulo contiene la classe MonitoredSimulationManager che estende
SimulationManager aggiungendo il supporto per il monitoraggio in tempo reale.
"""

import logging
from datetime import datetime
from .simulation import SimulationManager

class MonitoredSimulationManager(SimulationManager):
    """
    Gestore della simulazione con supporto per il monitoraggio in tempo reale
    
    Questa classe estende SimulationManager implementando i metodi di callback
    per integrarsi con i sistemi di monitoraggio.
    """
    
    def __init__(self, config, real_time_monitor=None):
        """
        Inizializza il gestore della simulazione monitorata
        
        Args:
            config: Configurazione della simulazione
            real_time_monitor: Istanza di RealTimeMonitor (opzionale)
        """
        print("DEBUG - MonitoredSimulationManager.__init__ - INIZIO")
        super().__init__(config)
        self.real_time_monitor = real_time_monitor
        self.simulation_state = {
            'status': 'idle',
            'start_time': None,
            'end_time': None,
            'progress': 0,
            'current_date': None,
            'error': None
        }
        
        if self.real_time_monitor:
            print("DEBUG - MonitoredSimulationManager - Attaching real_time_monitor")
            self.real_time_monitor.attach_simulation_manager(self)
        else:
            print("DEBUG - MonitoredSimulationManager - real_time_monitor is None")
            
        print("DEBUG - MonitoredSimulationManager.__init__ - COMPLETATO")
    
    def set_real_time_monitor(self, monitor):
        """
        Imposta il monitor in tempo reale
        
        Args:
            monitor: Istanza di RealTimeMonitor
        """
        self.real_time_monitor = monitor
        if self.real_time_monitor:
            self.real_time_monitor.attach_simulation_manager(self)
    
    def on_simulation_start(self, initial_state):
        """
        Callback per l'avvio della simulazione
        
        Args:
            initial_state: Stato iniziale della simulazione
        """
        print("DEBUG - MonitoredSimulationManager.on_simulation_start - INIZIO")
        self.simulation_state['status'] = 'running'
        self.simulation_state['start_time'] = datetime.now().isoformat()
        self.simulation_state['progress'] = 0
        self.simulation_state['error'] = None
        
        # Verifica che market_env e agents siano impostati
        if hasattr(self, 'market_env') and self.market_env:
            print(f"DEBUG - market_env è impostato: {type(self.market_env)}")
            if hasattr(self.market_env, 'trading_days'):
                print(f"DEBUG - market_env.trading_days: {len(self.market_env.trading_days)} giorni")
        else:
            print("DEBUG - ERROR: market_env non è impostato!")
            
        if hasattr(self, 'agents') and self.agents:
            print(f"DEBUG - agents è impostato: {len(self.agents)} agenti")
        else:
            print("DEBUG - ERROR: agents non è impostato o è vuoto!")
        
        # Notifica l'avvio tramite il monitor
        if self.real_time_monitor:
            print("DEBUG - Chiamando real_time_monitor.start_monitoring()")
            self.real_time_monitor.start_monitoring()
        else:
            print("DEBUG - ERROR: real_time_monitor è None, impossibile avviare il monitoraggio")
            
        print("DEBUG - MonitoredSimulationManager.on_simulation_start - COMPLETATO")
    
    def on_simulation_progress(self, progress, current_date, state):
        """
        Callback per l'avanzamento della simulazione
        
        Args:
            progress: Percentuale di completamento (0-100)
            current_date: Data corrente della simulazione
            state: Stato corrente della simulazione
        """
        self.simulation_state['status'] = 'running'
        self.simulation_state['progress'] = progress
        self.simulation_state['current_date'] = current_date
        
        # Non è necessario fare altro qui, il monitor aggiornerà automaticamente lo stato
    
    def on_simulation_complete(self, results):
        """
        Callback per il completamento della simulazione
        
        Args:
            results: Risultati della simulazione
        """
        self.simulation_state['status'] = 'completed'
        self.simulation_state['end_time'] = datetime.now().isoformat()
        self.simulation_state['progress'] = 100
        
        # Recupera i risultati finali tramite il monitor
        if self.real_time_monitor:
            final_results = self.real_time_monitor.get_simulation_results()
            self.logger.info(f"Risultati finali della simulazione: {len(final_results.get('agents', []))} agenti, {len(final_results.get('transactions', []))} transazioni")
            
            # Interrompi il monitoraggio
            self.real_time_monitor.stop_monitoring()
    
    def on_simulation_error(self, error_message):
        """
        Callback per errori durante la simulazione
        
        Args:
            error_message: Messaggio di errore
        """
        self.simulation_state['status'] = 'error'
        self.simulation_state['error'] = error_message
        self.simulation_state['end_time'] = datetime.now().isoformat()
        
        # Notifica l'errore tramite il monitor
        if self.real_time_monitor:
            # Stoppa il loop di polling
            self.real_time_monitor.stop_monitoring()
            
            # Aggiorna lo stato per riflettere l'errore
            # Il real_time_monitor ha accesso al websocket_manager e invierà l'errore
            # tramite il suo prossimo aggiornamento di stato
    
    def on_simulation_interrupted(self):
        """Callback per interruzione manuale, sovrascrive quello base."""
        self.logger.warning("Callback on_simulation_interrupted chiamato.")
        self.simulation_state['status'] = 'interrupted'
        self.simulation_state['end_time'] = datetime.now().isoformat()
        # Il progresso rimane quello raggiunto
        
        # Notifica l'interruzione tramite il monitor (che la invierà via websocket)
        if self.real_time_monitor:
            # Potremmo volere un evento websocket specifico per l'interruzione
            # Ma per ora, il monitor invierà lo stato 'interrupted' al prossimo poll
            self.real_time_monitor.stop_monitoring() # Stoppa il loop di polling
            
            # Non forziamo un aggiornamento diretto, lasciamo che sia il real_time_monitor
            # a gestire l'aggiornamento dello stato tramite il suo meccanismo standard
    
    def get_simulation_state(self):
        """
        Restituisce lo stato corrente della simulazione
        
        Returns:
            Dizionario con lo stato corrente
        """
        # Restituisce una copia per evitare modifiche esterne
        return self.simulation_state.copy()
