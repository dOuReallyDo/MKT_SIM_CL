"""
Simulation Manager Module.

Questo modulo contiene la classe SimulationManager per la gestione della simulazione di mercato.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
import json

# Aggiungiamo il percorso principale per poter importare altri moduli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_simulator.environment import MarketEnvironment
from market_simulator.agents import TradingAgent
from trading_strategy.strategies import RandomStrategy, MeanReversionStrategy, TrendFollowingStrategy
from trading_strategy import get_available_strategies

class SimulationManager:
    """Gestore della simulazione di mercato"""
    def __init__(self, config):
        """
        Inizializza il gestore della simulazione
        
        Args:
            config: Dizionario di configurazione della simulazione
        """
        self.config = config
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.market_env = None
        self.agents = []
        self.market_data = {}
        self._stop_requested = False
        
        # Configurazione del logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura il logging per il SimulationManager"""
        os.makedirs('./logs', exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger('SimulationManager')
        self.logger.setLevel(logging.INFO)
        
        # Rimuovi gli handler esistenti per evitare duplicazioni
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Handler del file
        file_handler = logging.FileHandler('./logs/simulation_manager.log')
        file_handler.setLevel(logging.INFO)
        
        # Handler della console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Aggiungi gli handler al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Evita la propagazione al logger root
        self.logger.propagate = False
        
    def load_market_data(self):
        """Carica i dati di mercato dai file di cache"""
        try:
            self.market_data = {}
            for symbol in self.config['market']['symbols']:
                cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
                if os.path.exists(cache_file):
                    try:
                        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        if not df.empty:
                            # Verifica che le colonne siano standardizzate
                            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                            col_mapping = {}
                            
                            # Creazione mapping
                            for std_col in expected_columns:
                                for col in df.columns:
                                    if col.lower() == std_col.lower():
                                        col_mapping[col] = std_col
                            
                            # Rinomina se necessario
                            if col_mapping:
                                df = df.rename(columns=col_mapping)
                            
                            # Verifica che le colonne necessarie esistano
                            missing_cols = [col for col in expected_columns if col not in df.columns]
                            if missing_cols:
                                self.logger.warning(f"Colonne mancanti per {symbol}: {missing_cols}")
                                continue
                            
                            # Assicura che l'indice sia datetime
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            
                            # Aggiungi i dati al dizionario
                            self.market_data[symbol] = df
                    except Exception as e:
                        self.logger.error(f"Errore nella lettura dei dati per {symbol}: {e}")
            
            if not self.market_data:
                self.logger.error("Nessun dato disponibile per la simulazione")
                return False
            
            self.logger.info(f"Dati di mercato caricati per {len(self.market_data)} simboli")
            return True
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dei dati di mercato: {e}")
            return False

    def initialize_simulation(self):
        """Inizializza l'ambiente di simulazione"""
        try:
            # Carica i dati di mercato
            if not self.load_market_data():
                return False
            
            # Converti le date in pd.Timestamp se sono stringhe
            start_date = pd.to_datetime(self.config['market']['start_date'])
            end_date = pd.to_datetime(self.config['market']['end_date'])
            
            # Inizializza l'ambiente di mercato
            self.market_env = MarketEnvironment(
                data=self.market_data,
                start_date=start_date,
                end_date=end_date
            )
            
            self.logger.info(f"Ambiente di mercato inizializzato con {len(self.market_data)} simboli e {len(self.market_env.get_dates())} giorni di trading")
            return True
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione della simulazione: {e}")
            return False

    def create_agents(self, num_agents=5):
        """Crea gli agenti di trading"""
        try:
            self.agents = []
            strategy_class = self.get_strategy_class()
            
            # Recupera i parametri della strategia attiva
            strategy_params = {}
            if 'strategies' in self.config and 'strategy_params' in self.config['strategies']:
                active_strategy = self.config['strategies']['active_strategy']
                if active_strategy in self.config['strategies']['strategy_params']:
                    strategy_params = self.config['strategies']['strategy_params'][active_strategy]
            
            # ++ MODIFICA: Verifica preliminare se la strategia è implementata usando la factory ++ 
            # Creiamo un'istanza "temporanea" solo per verificare
            temp_strategy_instance = self.create_strategy_instance(strategy_class, strategy_params)
            if temp_strategy_instance is None:
                 # La funzione create_strategy_instance (che usa create_strategy) ora ritorna None
                 # se la strategia non è implementata (o classe non trovata)
                 active_strategy_name = self.config.get('strategies', {}).get('active_strategy', 'N/A')
                 self.logger.error(f"Impossibile creare agenti: la strategia '{active_strategy_name}' non è implementata o valida.")
                 return False # Indica fallimento

            for i in range(num_agents):
                # Crea l'istanza della strategia con i parametri corretti
                # Riusiamo la funzione factory che ora gestisce i parametri
                strategy = self.create_strategy_instance(strategy_class, strategy_params)
                
                # Verifica (anche se ridondante dopo il check iniziale)
                if strategy is None:
                     self.logger.error(f"Errore critico: strategy è None per agente {i} anche dopo check iniziale.")
                     continue # Salta questo agente

                # Crea l'agente
                initial_capital = self.config['trading']['initial_capital']
                agent = TradingAgent(
                    id=i,
                    initial_capital=initial_capital,
                    strategy=strategy
                )
                
                # Aggiungi l'agente alla lista
                self.agents.append(agent)
                
                # Aggiungi l'agente all'ambiente di mercato
                if self.market_env:
                    self.market_env.add_agent(agent)
            
            active_strategy_name = self.config.get('strategies', {}).get('active_strategy', 'N/A')
            if not self.agents:
                 self.logger.warning(f"Nessun agente creato per la strategia {active_strategy_name} (potrebbe essere non valida?).")
                 # Non necessariamente un errore fatale, ma la simulazione sarà vuota
                 return True # La funzione in sé non è fallita, ma non ci sono agenti
            
            self.logger.info(f"Creati {len(self.agents)} agenti con strategia {active_strategy_name}")
            return True
        except Exception as e:
            self.logger.error(f"Errore nella creazione degli agenti: {e}", exc_info=True)
            return False

    def get_strategy_class(self):
        """
        Restituisce la classe della strategia di trading in base alla configurazione
        
        Returns:
            Classe della strategia di trading
        """
        strategy_name = None
        if 'strategies' in self.config and 'active_strategy' in self.config['strategies']:
            strategy_name = self.config['strategies']['active_strategy']
        else:
            # Se non è specificata, usa quella in config.trading.strategy
            strategy_name = self.config.get('trading', {}).get('strategy', 'random')
        
        # Mappa il nome alla classe
        strategy_mapping = {
            'random': RandomStrategy,
            'mean_reversion': MeanReversionStrategy,
            'trend_following': TrendFollowingStrategy
        }
        
        return strategy_mapping.get(strategy_name, RandomStrategy)
    
    def create_strategy_instance(self, strategy_class, params=None):
        """
        Crea un'istanza della strategia con i parametri specificati.
        Verifica anche se la strategia è implementata tramite trading_strategy.create_strategy.
        
        Args:
            strategy_class: Classe della strategia (potrebbe non essere più necessaria)
            params: Parametri per l'inizializzazione
            
        Returns:
            Istanza della strategia o None se non implementata/valida.
        """
        # Ottieni il nome stringa della strategia dalla classe o dalla config
        strategy_name = None
        if hasattr(strategy_class, '__name__'):
            # Trova il nome chiave corrispondente alla classe (un po' macchinoso)
            available_strategies = get_available_strategies() # Da trading_strategy
            for name, info in available_strategies.items():
                if info['class'] == strategy_class:
                    strategy_name = name
                    break
        
        # Se non trovato, prova dalla config (fallback)
        if strategy_name is None:
             strategy_name = self.config.get('strategies', {}).get('active_strategy')

        if strategy_name is None:
             self.logger.error("Impossibile determinare il nome della strategia per crearne un'istanza.")
             return None

        # Usa la factory create_strategy da trading_strategy che gestisce lo stato 'implemented'
        from trading_strategy import create_strategy # Importa qui per evitare dipendenze circolari
        strategy_instance = create_strategy(strategy_name, **(params or {}))
        
        if strategy_instance is None:
             self.logger.warning(f"Tentativo di creare istanza per strategia non implementata o non valida: {strategy_name}")
        
        return strategy_instance

    def request_stop(self):
        """Imposta il flag per richiedere l'interruzione della simulazione."""
        self.logger.info("Richiesta di interruzione simulazione ricevuta.")
        self._stop_requested = True

    def run_simulation(self):
        """Esegue la simulazione"""
        try:
            # Verifica che la simulazione sia stata inizializzata
            if not hasattr(self, 'market_env') or not hasattr(self, 'agents'):
                self.logger.error("Simulazione non inizializzata correttamente")
                return None
            
            # Resetta la storia delle transazioni
            self.transaction_history = []
            self._stop_requested = False
            
            # Inizializza i risultati
            # Gestisci sia stringhe che oggetti datetime per le date
            start_date = self.config['market']['start_date']
            end_date = self.config['market']['end_date']
            
            # Converti in stringa se è un oggetto datetime
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')
                
            results = {
                'start_date': start_date,
                'end_date': end_date,
                'symbols': list(self.market_data.keys()),
                'agents': [],
                'transactions': [],
                'daily_data': []
            }
            
            # Notifica l'avvio della simulazione
            self.on_simulation_start(results)
            
            # Loop principale della simulazione
            self.logger.info("Avvio della simulazione")
            day_counter = 0
            total_days = len(self.market_env.trading_days)
            simulation_interrupted = False
            
            while self.market_env.step() and not self._stop_requested:
                day_counter += 1
                
                # Recupera lo stato corrente
                current_market_data = self.market_env.get_current_market_data()
                current_date = self.market_env.current_date
                
                # Registra lo stato giornaliero
                daily_state = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'market_data': current_market_data,
                    'agents': []
                }
                
                # Calcola le performance degli agenti
                for agent in self.agents:
                    agent_performance = agent.get_performance_metrics(current_market_data)
                    daily_state['agents'].append(agent_performance)
                
                # Aggiungi lo stato giornaliero ai risultati
                results['daily_data'].append(daily_state)
                
                # Notifica l'avanzamento della simulazione
                progress = (day_counter / total_days) * 100
                self.on_simulation_progress(progress, current_date.strftime('%Y-%m-%d'), daily_state)
            
            if self._stop_requested:
                self.logger.warning("Simulazione interrotta dall'utente.")
                simulation_interrupted = True
                self.on_simulation_interrupted()
            
            # Calcola i risultati finali
            final_market_data = self.market_env.get_current_market_data()
            
            for agent in self.agents:
                agent_result = agent.get_performance_metrics(final_market_data)
                agent_result['transactions'] = len(agent.transactions)
                agent_result['strategy'] = agent.strategy.__class__.__name__
                results['agents'].append(agent_result)
            
            # Aggiungi tutte le transazioni
            results['transactions'] = self.market_env.transactions
            self.transaction_history = self.market_env.transactions
            
            # Notifica il completamento (o l'interruzione)
            if not simulation_interrupted:
                self.on_simulation_complete(results)
                self.logger.info(f"Simulazione completata con {len(self.transaction_history)} transazioni")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della simulazione: {e}")
            # Notifica l'errore
            self.on_simulation_error(str(e))
            return None

    def on_simulation_start(self, initial_state):
        """
        Callback per l'avvio della simulazione
        
        Args:
            initial_state: Stato iniziale della simulazione
        """
        # Questo metodo può essere sovrascitto da sottoclassi o gestito esternamente
        pass
    
    def on_simulation_progress(self, progress, current_date, state):
        """
        Callback per l'avanzamento della simulazione
        
        Args:
            progress: Percentuale di completamento (0-100)
            current_date: Data corrente della simulazione
            state: Stato corrente della simulazione
        """
        # Questo metodo può essere sovrascitto da sottoclassi o gestito esternamente
        pass
    
    def on_simulation_complete(self, results):
        """
        Callback per il completamento della simulazione
        
        Args:
            results: Risultati della simulazione
        """
        # Questo metodo può essere sovrascitto da sottoclassi o gestito esternamente
        pass
    
    def on_simulation_error(self, error_message):
        """
        Callback per errori durante la simulazione
        
        Args:
            error_message: Messaggio di errore
        """
        # Questo metodo può essere sovrascitto da sottoclassi o gestito esternamente
        pass

    def on_simulation_interrupted(self):
        """Callback per l'interruzione manuale della simulazione"""
        # Aggiorna lo stato interno per riflettere l'interruzione
        # Questo callback verrà chiamato solo se la simulazione viene interrotta
        # Verrà sovrascritto in MonitoredSimulationManager per aggiornare lo stato lì
        pass

    def get_transactions_summary(self):
        """Restituisce un riepilogo delle transazioni della simulazione"""
        if not self.market_env:
            return None
        
        transactions = self.market_env.transactions
        
        # Se non ci sono transazioni, restituisci un riepilogo vuoto
        if not transactions:
            return {
                'total_transactions': 0,
                'buy_transactions': 0,
                'sell_transactions': 0,
                'total_buy_value': 0,
                'total_sell_value': 0,
                'net_value': 0
            }
        
        # Analisi delle transazioni
        buy_transactions = [t for t in transactions if t['action'] == 'buy']
        sell_transactions = [t for t in transactions if t['action'] == 'sell']
        
        total_buy_value = sum(t['total'] for t in buy_transactions)
        total_sell_value = sum(t['total'] for t in sell_transactions)
        
        return {
            'total_transactions': len(transactions),
            'buy_transactions': len(buy_transactions),
            'sell_transactions': len(sell_transactions),
            'total_buy_value': total_buy_value,
            'total_sell_value': total_sell_value,
            'net_value': total_sell_value - total_buy_value
        }
    
    def get_agents_performance(self):
        """Restituisce le performance degli agenti"""
        if not self.market_env:
            self.logger.warning("Non è possibile calcolare le performance: ambiente di mercato non inizializzato")
            return []
        
        return self.market_env.get_all_agent_performances()
    
    def save_results(self, output_dir=None):
        """
        Salva i risultati della simulazione
        
        Args:
            output_dir: Directory dove salvare i risultati (default: './reports')
            
        Returns:
            bool: True se il salvataggio è riuscito, False altrimenti
        """
        try:
            # Directory di output
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Timestamp per il nome file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f'simulation_{timestamp}.json')
            
            # Raccogli i risultati
            results = {
                'config': self.config,
                'summary': self.get_transactions_summary(),
                'performances': self.get_agents_performance(),
                'transactions': self.market_env.transactions if self.market_env else []
            }
            
            # Converti le date in stringhe per JSON
            for transaction in results['transactions']:
                if isinstance(transaction['date'], pd.Timestamp):
                    transaction['date'] = transaction['date'].strftime('%Y-%m-%d')
            
            # Salva il file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Risultati salvati in {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio dei risultati: {e}")
            return False
