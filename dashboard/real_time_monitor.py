"""
Real-Time Monitor Module.

Questo modulo fornisce le funzionalità di monitoraggio in tempo reale 
degli agenti di trading durante la simulazione.
"""

import threading
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np # Aggiunto per calcoli aggregati

class RealTimeMonitor:
    """Gestisce il monitoraggio in tempo reale degli agenti di trading"""
    
    def __init__(self, websocket_manager, state_manager):
        """
        Inizializza il monitor in tempo reale
        
        Args:
            websocket_manager: Istanza di WebSocketManager per inviare aggiornamenti
            state_manager: Istanza di DashboardStateManager per aggiornare lo stato
        """
        self.websocket_manager = websocket_manager
        self.state_manager = state_manager
        self.is_monitoring = False
        self.monitoring_thread = None
        self.simulation_manager = None
        
        # Configura il logging
        self.logger = logging.getLogger('RealTimeMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Se non ci sono handler, aggiungine uno
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            
        self.logger.info("RealTimeMonitor inizializzato")
    
    def attach_simulation_manager(self, simulation_manager):
        """
        Collega il monitor al SimulationManager
        
        Args:
            simulation_manager: Istanza di SimulationManager da monitorare
        """
        self.simulation_manager = simulation_manager
        self.logger.info(f"Collegato al SimulationManager")
    
    def start_monitoring(self, update_interval=1.0):
        """
        Avvia il monitoraggio in tempo reale
        
        Args:
            update_interval: Intervallo di aggiornamento in secondi
        """
        print(f"DEBUG - RealTimeMonitor.start_monitoring(update_interval={update_interval}) - START")
        if self.is_monitoring:
            self.logger.warning("Il monitoraggio è già in corso")
            print("DEBUG - RealTimeMonitor.start_monitoring - Monitoraggio già in corso, uscita")
            return
        
        if not self.simulation_manager:
            self.logger.error("Nessun SimulationManager collegato")
            print("DEBUG - RealTimeMonitor.start_monitoring - Nessun SimulationManager collegato, uscita")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"DEBUG - RealTimeMonitor.start_monitoring - Thread avviato: {self.monitoring_thread.ident}")
        self.logger.info(f"Monitoraggio avviato con intervallo di {update_interval}s")
    
    def stop_monitoring(self):
        """Interrompe il monitoraggio in tempo reale"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("Monitoraggio interrotto")
    
    def _monitoring_loop(self, update_interval):
        """
        Loop principale di monitoraggio
        
        Args:
            update_interval: Intervallo di aggiornamento in secondi
        """
        print(f"DEBUG - RealTimeMonitor._monitoring_loop(update_interval={update_interval}) - STARTING")
        last_update_time = time.time()
        loop_count = 0
        forced_update_count = 0  # Contatore per forzare aggiornamenti
        force_update = True  # Forza un aggiornamento al primo ciclo
        
        while self.is_monitoring:
            try:
                loop_count += 1
                current_time = time.time()
                
                # Forza un aggiornamento ogni 5 secondi indipendentemente
                if current_time - last_update_time >= update_interval or force_update:
                    print(f"DEBUG - RealTimeMonitor._monitoring_loop - Iterazione #{loop_count} - Inviando aggiornamento")
                    
                    # Recupera lo stato attuale del simulatore e degli agenti
                    simulation_state = self._get_simulation_state()
                    
                    # IMPORTANTE: Se non abbiamo stato, creiamo uno stato fittizio per testing
                    if not simulation_state and forced_update_count < 3:
                        forced_update_count += 1
                        print(f"DEBUG - FORZANDO EMISSIONE stato simulato #{forced_update_count}")
                        
                        # Stato fittizio per debug
                        simulation_state = {
                            'status': 'running',
                            'timestamp': datetime.now().isoformat(),
                            'progress': forced_update_count * 10,  # 10%, 20%, 30%
                            'current_day': '2025-01-01',
                            'agents': [],
                            'forced_debug_msg': f"Stato simulato #{forced_update_count}",
                        }
                    
                    if simulation_state:
                        print(f"DEBUG - RealTimeMonitor._monitoring_loop - Stato ottenuto, status={simulation_state.get('status', 'N/A')}")
                        
                        # FORZARE STATO RUNNING per test
                        if 'status' not in simulation_state:
                            simulation_state['status'] = 'running'
                        
                        # Invia l'aggiornamento tramite WebSocket
                        self.websocket_manager.emit_market_simulation_update(simulation_state)
                        
                        # Aggiorna lo stato salvato
                        self.state_manager.update_market_simulation_state(simulation_state)
                        
                        self.logger.debug("Aggiornamento inviato")
                        print(f"DEBUG - Aggiornamento #{loop_count} inviato a websocket_manager")
                    else:
                        print("DEBUG - RealTimeMonitor._monitoring_loop - _get_simulation_state ha restituito None o vuoto")
                    
                    last_update_time = current_time
                    force_update = False  # Resetta dopo il primo aggiornamento forzato
                
                # Breve pausa ma stampa per debug
                if loop_count % 20 == 0:  # Ogni 2 secondi (20 * 0.1s)
                    print(f"DEBUG - RealTimeMonitor._monitoring_loop - Still alive... loop #{loop_count}")
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop di monitoraggio: {e}")
                time.sleep(1.0)  # Pausa più lunga in caso di errore
    
    def _get_simulation_state(self) -> Optional[Dict[str, Any]]:
        """
        Recupera lo stato attuale della simulazione
        
        Returns:
            Dizionario con lo stato della simulazione o None in caso di errore
        """
        try:
            # Inizio Blocco Try Principale
            if not self.simulation_manager:
                print("DEBUG - _get_simulation_state: simulation_manager è None")
                return None
            
            # Verifica se la simulazione è in corso
            sim_state = self.simulation_manager.get_simulation_state() # Usa lo stato interno del manager
            print(f"DEBUG - _get_simulation_state: sim_state = {sim_state}")
            
            # IMPORTANTE: Return uno stato base anche se non è in running
            # Il frontend può ancora mostrare 'in attesa' o 'completato'
            if sim_state.get('status') != 'running':
                status_state = {
                    'status': sim_state.get('status', 'idle'),
                    'timestamp': datetime.now().isoformat(),
                    'progress': sim_state.get('progress', 0),
                    'error': sim_state.get('error')
                }
                print(f"DEBUG - _get_simulation_state: ritorno stato non-running: {status_state['status']}")
                return status_state

            # Verifica presenza di market_env e agents (attributi fondamentali)
            if not hasattr(self.simulation_manager, 'market_env') or not hasattr(self.simulation_manager, 'agents'):
                self.logger.warning("Simulazione in stato running ma market_env o agents non trovati.")
                print("DEBUG - _get_simulation_state: market_env o agents non trovati")
                error_state = {
                    'status': 'error',
                    'error': 'Inconsistenza interna: market_env o agents mancanti',
                    'timestamp': datetime.now().isoformat(),
                }
                return error_state
            
            # Recupera lo stato attuale
            agents_state = []
            total_portfolio_value_all_agents = 0
            all_recent_transactions = [] # Lista per tutte le transazioni recenti
            current_market_data = self.simulation_manager.market_env.get_current_market_data()
            
            # --- Inizio Ciclo For Agenti ---
            for agent in self.simulation_manager.agents:
                try:
                    # Inizio Blocco Try Interno (per singolo agente)
                    # Calcola le performance
                    performance = agent.get_performance_metrics(current_market_data)
                    total_portfolio_value_all_agents += performance.get('current_value', 0)
                    
                    # Recupera le transazioni recenti dell'agente (es. ultime 5)
                    agent_recent_tx = []
                    if hasattr(agent, 'transactions') and agent.transactions:
                        for tx in agent.transactions[-5:]:
                            tx_copy = tx.copy()
                            tx_copy['agent_id'] = agent.id
                            # Assicurati che la data sia una stringa
                            if isinstance(tx_copy.get('date'), datetime):
                                tx_copy['date'] = tx_copy['date'].strftime('%Y-%m-%d')
                            agent_recent_tx.append(tx_copy)
                        all_recent_transactions.extend(agent_recent_tx) # Aggiungi alla lista globale

                    # Calcola il valore del portafoglio specifico dell'agente
                    portfolio_items = []
                    if isinstance(agent.portfolio, dict):
                        # --- Inizio Ciclo For Portfolio --- 
                        for symbol, quantity in agent.portfolio.items(): 
                            # Codice interno al ciclo for portfolio (correttamente indentato)
                            price = None
                            if symbol in current_market_data and quantity > 0:
                                if 'Close' in current_market_data[symbol]:
                                    price = current_market_data[symbol]['Close']
                                elif 'close' in current_market_data[symbol]: # Prova lowercase
                                    price = current_market_data[symbol]['close']
                                
                            if price is not None:
                                value = price * quantity
                                portfolio_items.append({
                                    'symbol': symbol,
                                    'quantity': quantity,
                                    'price': price,
                                    'value': value
                                })
                        # --- Fine Ciclo For Portfolio --- 
                    
                    agents_state.append({
                        'id': agent.id,
                        'initial_capital': agent.initial_capital,
                        'cash': agent.cash,
                        'portfolio': portfolio_items, # Valore attuale dettagliato
                        'performance': performance, # Contiene già pnl, return, current_value, etc.
                        'strategy': agent.strategy.__class__.__name__
                    })
                    # Fine Blocco Try Interno
                except Exception as e:
                    # Inizio Blocco Except Interno (correttamente allineato con il try interno)
                    self.logger.error(f"Errore nell'elaborazione dello stato dell'agente {agent.id}: {e}", exc_info=True)
                    # Fine Blocco Except Interno
            # --- Fine Ciclo For Agenti ---
            
            # Codice dopo il ciclo for agenti (indentato dentro il try principale)
            # Ordina tutte le transazioni recenti per data (più recenti prima) e prendi le ultime 10
            all_recent_transactions.sort(key=lambda x: x.get('date', '1970-01-01'), reverse=True)
            recent_transactions_to_send = all_recent_transactions[:10]

            # Recupera lo stato complessivo del mercato e progresso
            current_date_obj = self.simulation_manager.market_env.current_date
            current_date_str = current_date_obj.strftime('%Y-%m-%d') if current_date_obj else None
            total_days = len(self.simulation_manager.market_env.trading_days) if hasattr(self.simulation_manager.market_env, 'trading_days') else 0
            current_day_idx = self.simulation_manager.market_env.current_day_idx if hasattr(self.simulation_manager.market_env, 'current_day_idx') else 0
            progress = (current_day_idx / total_days) * 100 if total_days > 0 else 0
            
            # Prepara il punto dati per il grafico
            chart_point = None
            if current_date_str:
                chart_point = {
                    'date': current_date_str,
                    'value': total_portfolio_value_all_agents
                }

            # Crea stato running da ritornare
            running_state = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'progress': progress,
                'current_day': current_date_str,
                'total_days': total_days,
                'current_day_idx': current_day_idx,
                'agents': agents_state,
                'transactions_count': len(self.simulation_manager.transaction_history) if hasattr(self.simulation_manager, 'transaction_history') else 0,
                'recent_transactions': recent_transactions_to_send, # Aggiunto: Ultime 10 transazioni aggregate
                'chart_point': chart_point # Aggiunto: Punto per grafico
            }
            
            print(f"DEBUG - _get_simulation_state: ritorno stato running con progress={progress}")
            return running_state
            
        except Exception as e:
            # Inizio Blocco Except Principale (correttamente allineato con il try principale)
            self.logger.error(f"Errore grave nel recupero dello stato della simulazione: {e}", exc_info=True)
            # Invia un errore via websocket se possibile
            if self.websocket_manager:
                self.websocket_manager.emit_error(f"Errore monitor: {e}", 'monitoring')
            return {
                'status': 'error',
                'error': f'Errore interno monitor: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """
        Recupera i risultati finali della simulazione
        
        Returns:
            Dizionario con i risultati della simulazione
        """
        try:
            if not self.simulation_manager:
                return {'error': 'Nessun SimulationManager collegato'}
            
            # Verifica se la simulazione è stata eseguita
            if not hasattr(self.simulation_manager, 'agents'):
                return {'error': 'Simulazione non eseguita'}
            
            # Recupera i risultati
            results = {
                'timestamp': datetime.now().isoformat(),
                'agents': [],
                'transactions': self.simulation_manager.transaction_history if hasattr(self.simulation_manager, 'transaction_history') else []
            }
            
            # Dati degli agenti
            for agent in self.simulation_manager.agents:
                try:
                    market_data = self.simulation_manager.market_env.get_current_market_data()
                    performance = agent.get_performance_metrics(market_data)
                    
                    agent_data = {
                        'id': agent.id,
                        'initial_capital': agent.initial_capital,
                        'final_capital': performance['current_value'],
                        'return': performance['percentage_return'],
                        'transactions': len(agent.transactions),
                        'portfolio': agent.portfolio,
                        'strategy': agent.strategy.__class__.__name__
                    }
                    results['agents'].append(agent_data)
                except Exception as e:
                    self.logger.error(f"Errore nel recupero dei risultati dell'agente {agent.id}: {e}")
            
            return results
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei risultati della simulazione: {e}")
            return {'error': str(e)}
