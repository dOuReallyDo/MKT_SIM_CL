import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random
import uuid
from functools import lru_cache
import redis
import psutil
import time
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from trading_strategy.strategies import TradingStrategy, RandomStrategy, MeanReversionStrategy, TrendFollowingStrategy, ValueInvestingStrategy, NeuralNetworkStrategy
from neural_network.model_trainer import ModelTrainer
from logging.handlers import RotatingFileHandler
import pickle

# Configurazione del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Metriche Prometheus
TRADING_OPERATIONS = Counter('trading_operations_total', 'Numero totale di operazioni di trading')
CACHE_HITS = Counter('cache_hits_total', 'Numero di hit nella cache')
CACHE_MISSES = Counter('cache_misses_total', 'Numero di miss nella cache')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Uso della memoria in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'Uso della CPU in percentuale')
OPERATION_LATENCY = Histogram('operation_latency_seconds', 'Latenza delle operazioni')

class DistributedCache:
    """Cache distribuita basata su Redis"""
    def __init__(self, host='localhost', port=6379, db=0):
        """
        Inizializza la cache distribuita
        
        Args:
            host: Host del server Redis
            port: Porta del server Redis
            db: Indice del database Redis
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False
            )
            self.logger = logging.getLogger('DistributedCache')
        except Exception as e:
            logger.error(f"Errore nella connessione a Redis: {e}")
            self.redis_client = None
    
    def get(self, key):
        """
        Recupera un valore dalla cache
        
        Args:
            key: Chiave del valore da recuperare
            
        Returns:
            Il valore recuperato o None se non trovato
        """
        try:
            if not self.redis_client:
                return None
            
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Errore nel recupero dalla cache: {e}")
            return None
    
    def set(self, key, value, expire=3600):
        """
        Salva un valore nella cache
        
        Args:
            key: Chiave del valore
            value: Valore da salvare
            expire: Tempo di scadenza in secondi
            
        Returns:
            True se il salvataggio è riuscito, False altrimenti
        """
        try:
            if not self.redis_client:
                return False
            
            data = pickle.dumps(value)
            return self.redis_client.setex(key, expire, data)
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio nella cache: {e}")
            return False
    
    def delete(self, key):
        """
        Elimina un valore dalla cache
        
        Args:
            key: Chiave del valore da eliminare
            
        Returns:
            True se l'eliminazione è riuscita, False altrimenti
        """
        try:
            if not self.redis_client:
                return False
            
            return bool(self.redis_client.delete(key))
        except Exception as e:
            self.logger.error(f"Errore nell'eliminazione dalla cache: {e}")
            return False

class PerformanceMonitor:
    """Monitora le performance del sistema"""
    def __init__(self):
        """
        Inizializza il monitor di performance
        """
        self.logger = logging.getLogger('PerformanceMonitor')
        self.start_time = time.time()
        self.operations = {}
        self.memory_samples = []
        self.cpu_samples = []
        
        # Avvia il server Prometheus
        try:
            # Prova diverse porte
            for port in range(8000, 8010):
                try:
                    start_http_server(port)
                    self.logger.info(f"Server Prometheus avviato sulla porta {port}")
                    break
                except Exception:
                    continue
            else:
                self.logger.warning("Impossibile avviare il server Prometheus su nessuna porta")
        except Exception as e:
            self.logger.error(f"Errore nell'avvio del server Prometheus: {e}")
    
    def record_operation(self, operation_name, duration):
        """Registra un'operazione con la sua durata"""
        if operation_name not in self.operations:
            self.operations[operation_name] = {
                'count': 0,
                'total_time': 0,
                'times': []
            }
        
        self.operations[operation_name]['count'] += 1
        self.operations[operation_name]['total_time'] += duration
        self.operations[operation_name]['times'].append(duration)
    
    def record_memory_usage(self):
        """Registra l'uso corrente della memoria"""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_samples.append(memory_info.rss)
        MEMORY_USAGE.set(memory_info.rss)
    
    def record_cpu_usage(self):
        """Registra l'uso corrente della CPU"""
        cpu_percent = psutil.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        CPU_USAGE.set(cpu_percent)
    
    def get_performance_report(self):
        """Genera un report delle performance"""
        return {
            'uptime': time.time() - self.start_time,
            'operations': {
                name: {
                    'count': data['count'],
                    'total_time': data['total_time'],
                    'avg_time': data['total_time'] / data['count'] if data['count'] > 0 else 0,
                    'max_time': max(data['times']) if data['times'] else 0,
                    'min_time': min(data['times']) if data['times'] else 0,
                    'times': data['times']
                }
                for name, data in self.operations.items()
            },
            'memory': {
                'current': self.memory_samples[-1] if self.memory_samples else 0,
                'avg': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
                'max': max(self.memory_samples) if self.memory_samples else 0,
                'min': min(self.memory_samples) if self.memory_samples else 0
            },
            'cpu': {
                'current': self.cpu_samples[-1] if self.cpu_samples else 0,
                'avg': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
                'max': max(self.cpu_samples) if self.cpu_samples else 0,
                'min': min(self.cpu_samples) if self.cpu_samples else 0
            }
        }

class MarketEnvironment:
    def __init__(self, data, start_date, end_date, opening_time="09:30", closing_time="16:00"):
        """
        Inizializza l'ambiente di mercato
        
        Args:
            data: Dizionario con i dati delle azioni
            start_date: Data di inizio
            end_date: Data di fine
            opening_time: Orario di apertura del mercato
            closing_time: Orario di chiusura del mercato
        """
        self.stocks_data = data
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.opening_time = opening_time
        self.closing_time = closing_time
        
        # Inizializzazione delle cache
        self._price_cache = {}
        self._market_data_cache = {}
        
        # Inizializzazione del monitor di performance
        self.performance_monitor = PerformanceMonitor()
        
        # Inizializzazione della cache distribuita
        self.distributed_cache = DistributedCache()
        
        # Inizializzazione del logger
        self.logger = logging.getLogger('MarketEnvironment')
        
        # Genera le date di trading
        self.trading_days = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        self.current_index = 0
        self.current_date = self.trading_days[0]
        self.transactions = []
        
        # Configurazione del logging
        self._setup_logging()
        
        # Ottimizzazione memoria
        self._optimize_memory_usage()
        
        # Inizializzazione cache
        self._cache_size = 1000  # Dimensione massima della cache
    
    def _setup_logging(self):
        """Configura il sistema di logging con rotazione dei file"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurazione del logger principale
        self.logger.setLevel(logging.INFO)
        
        # Handler per file con rotazione
        file_handler = RotatingFileHandler(
            f'{log_dir}/market_simulator.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Handler per console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formattazione
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Aggiunta degli handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("MarketEnvironment inizializzato con logging configurato")
    
    def _optimize_memory_usage(self):
        """Ottimizza l'uso della memoria convertendo i tipi di dati"""
        try:
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for symbol, df in self.stocks_data.items():
                for col in numeric_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                        except Exception as e:
                            self.logger.warning(f"Impossibile convertire la colonna {col} del simbolo {symbol} in float32: {e}")
            return True
        except Exception as e:
            self.logger.error(f"Errore nell'ottimizzazione della memoria: {e}")
            return False
    
    def get_dates(self):
        """Restituisce la lista delle date di trading"""
        # Genera tutte le date tra start_date e end_date
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        return dates
    
    def get_market_data(self, date):
        """
        Recupera i dati di mercato per una data specifica con caching distribuito
        
        Args:
            date: Data per cui recuperare i dati
            
        Returns:
            dict: Dizionario con i dati di mercato
        """
        start_time = time.time()
        try:
            # Controlla la cache locale
            if date in self._market_data_cache:
                self.performance_monitor.record_operation('local_cache_hit', time.time() - start_time)
                return self._market_data_cache[date]
            
            # Controlla la cache distribuita
            cache_key = f"market_data:{date}"
            cached_data = self.distributed_cache.get(cache_key)
            if cached_data is not None:
                CACHE_HITS.inc()
                self.performance_monitor.record_operation('cache_hit', time.time() - start_time)
                # Salva anche nella cache locale
                self._market_data_cache[date] = cached_data
                return cached_data
            
            CACHE_MISSES.inc()
            
            # Recupera i dati
            market_data = {}
            for symbol, df in self.stocks_data.items():
                if date in df.index:
                    market_data[symbol] = {
                        'open': float(df.loc[date, 'Open']),
                        'high': float(df.loc[date, 'High']),
                        'low': float(df.loc[date, 'Low']),
                        'close': float(df.loc[date, 'Close']),
                        'volume': int(df.loc[date, 'Volume'])
                    }
            
            # Salva in entrambe le cache
            self._market_data_cache[date] = market_data
            self.distributed_cache.set(cache_key, market_data)
            
            self.performance_monitor.record_operation('market_data_fetch', time.time() - start_time)
            return market_data
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei dati di mercato: {e}")
            return {}
    
    def get_current_price(self, symbol):
        """Recupera il prezzo corrente per un simbolo"""
        try:
            if symbol not in self.stocks_data:
                raise ValueError(f"Simbolo {symbol} non trovato")
            
            if self.current_index >= len(self.trading_days):
                raise ValueError("Indice corrente fuori range")
            
            price = float(self.stocks_data[symbol]['Close'].iloc[self.current_index])
            return price
        except Exception as e:
            self.logger.error(f"Errore nel recupero del prezzo per {symbol}: {e}")
            return None
    
    def execute_transaction(self, agent_id, symbol, action, quantity, price):
        """Esegue una transazione"""
        try:
            start_time = time.time()
            if symbol not in self.stocks_data:
                raise ValueError(f"Simbolo {symbol} non trovato")
            
            if self.current_index >= len(self.trading_days):
                raise ValueError("Indice corrente fuori range")
            
            transaction = {
                'date': self.current_date,
                'agent_id': agent_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'total': quantity * price
            }
            
            self.transactions.append(transaction)
            self.performance_monitor.record_operation('trade_execution', time.time() - start_time)
            return transaction
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione della transazione: {e}")
            return None

    def clear_cache(self):
        """Pulisce tutte le cache"""
        self._price_cache.clear()
        self._market_data_cache.clear()
        self.get_current_price.cache_clear()
        self.get_market_data.cache_clear()
        
        # Pulisce la cache distribuita
        if self.distributed_cache.redis_client:
            self.distributed_cache.redis_client.flushdb()
        
        self.logger.info("Cache pulita")

    def get_performance_metrics(self):
        """
        Recupera le metriche di performance
        
        Returns:
            dict: Metriche di performance
        """
        self.performance_monitor.record_memory_usage()
        self.performance_monitor.record_cpu_usage()
        return self.performance_monitor.get_performance_report()

class TradingAgent:
    def __init__(self, id, initial_capital, strategy):
        """
        Inizializza un agente di trading
        
        Args:
            id: Identificatore univoco dell'agente
            initial_capital: Capitale iniziale
            strategy: Strategia di trading da utilizzare
        """
        self.id = id
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.portfolio = {}  # {symbol: quantity}
        self.strategy = strategy
        self.transactions = []
        
        # Configurazione del logging
        self.logger = logging.getLogger(f'TradingAgent_{id}')
    
    def generate_signal(self, market_data):
        """
        Genera un segnale di trading basato sulla strategia
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            return self.strategy.generate_signal(market_data)
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale: {e}")
            return None
    
    def get_portfolio_value(self, market_data):
        """
        Calcola il valore totale del portafoglio
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Valore totale del portafoglio
        """
        portfolio_value = self.cash
        
        for symbol, quantity in self.portfolio.items():
            if symbol in market_data:
                price = market_data[symbol]['close']
                portfolio_value += price * quantity
        
        return portfolio_value
    
    def get_performance_metrics(self, market_data):
        """
        Calcola le metriche di performance dell'agente
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con le metriche di performance
        """
        current_value = self.get_portfolio_value(market_data)
        absolute_return = current_value - self.initial_capital
        percentage_return = (absolute_return / self.initial_capital) * 100
        
        return {
            'id': self.id,
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'absolute_return': absolute_return,
            'percentage_return': percentage_return
        }


class SimulationManager:
    """Gestore della simulazione di mercato"""
    def __init__(self, config):
        """
        Inizializza il gestore della simulazione
        
        Args:
            config: Dizionario di configurazione della simulazione
        """
        self.config = config
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.market_env = None
        self.agents = []
        self.transaction_history = []
        self.market_data = {}
        self.performance_monitor = PerformanceMonitor()
        
        # Configurazione del logging
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/simulation_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SimulationManager')
        
    def load_market_data(self):
        """Carica i dati di mercato dai file di cache"""
        try:
            self.market_data = {}
            for symbol in self.config['market']['symbols']:
                cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
                if os.path.exists(cache_file):
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        self.market_data[symbol] = df
            
            if not self.market_data:
                raise ValueError("Nessun dato disponibile per la simulazione")
            
            return True
        except Exception as e:
            logger.error(f"Errore nel caricamento dei dati di mercato: {e}")
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
            
            # Aggiorna le date nel config
            self.config['market']['start_date'] = start_date
            self.config['market']['end_date'] = end_date
            
            # Inizializza l'ambiente di mercato
            self.market_env = MarketEnvironment(
                data=self.market_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Inizializza il registro delle transazioni
            self.transaction_history = []
            
            return True
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione della simulazione: {e}")
            return False

    def create_agents(self, num_agents=5):
        """Crea gli agenti di trading"""
        try:
            self.agents = []
            strategy_class = self.get_strategy_class()
            
            for i in range(num_agents):
                agent = TradingAgent(
                    id=i,
                    initial_capital=self.config['trading']['initial_capital'],
                    strategy=strategy_class()
                )
                self.agents.append(agent)
            
            return True
        except Exception as e:
            logger.error(f"Errore nella creazione degli agenti: {e}")
            return False

    def run_simulation(self):
        """Esegue la simulazione"""
        try:
            if not hasattr(self, 'market_env') or not hasattr(self, 'agents'):
                raise ValueError("Simulazione non inizializzata correttamente")
            
            # Esegui la simulazione giorno per giorno
            for date in self.market_env.get_dates():
                # Aggiorna i prezzi di mercato
                market_data = self.market_env.get_market_data(date)
                
                # Esegui le operazioni di trading per ogni agente
                for agent in self.agents:
                    # Ottieni il segnale di trading
                    signal = agent.strategy.generate_signal(market_data)
                    
                    # Esegui l'operazione
                    if signal:
                        transaction = self.market_env.execute_transaction(
                            agent_id=agent.id,
                            symbol=signal['symbol'],
                            action=signal['action'],
                            quantity=signal['quantity'],
                            price=market_data[signal['symbol']]['close']
                        )
                        if transaction:
                            # Aggiungi il segnale e il trend alla transazione
                            transaction['signal'] = signal
                            if isinstance(agent.strategy, TrendFollowingStrategy):
                                # Calcola il trend per la strategia di trend following
                                short_prices = agent.strategy.historical_prices[signal['symbol']][-agent.strategy.short_window:]
                                long_prices = agent.strategy.historical_prices[signal['symbol']][-agent.strategy.long_window:]
                                short_ma = sum(short_prices) / len(short_prices)
                                long_ma = sum(long_prices) / len(long_prices)
                                transaction['trend'] = 'up' if short_ma > long_ma else 'down'
                            
                            self.transaction_history.append(transaction)
            
            return self.transaction_history
        except Exception as e:
            logger.error(f"Errore nell'esecuzione della simulazione: {e}")
            return None
    
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
        
        # Conversione in DataFrame per analisi
        transactions_df = pd.DataFrame(transactions)
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        # Analisi delle transazioni
        total_buy_value = transactions_df[transactions_df['type'] == 'buy']['value'].sum() if 'type' in transactions_df.columns else 0
        total_sell_value = transactions_df[transactions_df['type'] == 'sell']['value'].sum() if 'type' in transactions_df.columns else 0
        buy_count = len(transactions_df[transactions_df['type'] == 'buy']) if 'type' in transactions_df.columns else 0
        sell_count = len(transactions_df[transactions_df['type'] == 'sell']) if 'type' in transactions_df.columns else 0
        
        return {
            'total_transactions': len(transactions),
            'buy_transactions': buy_count,
            'sell_transactions': sell_count,
            'total_buy_value': float(total_buy_value) if isinstance(total_buy_value, pd.Series) else total_buy_value,
            'total_sell_value': float(total_sell_value) if isinstance(total_sell_value, pd.Series) else total_sell_value,
            'net_value': float(total_sell_value - total_buy_value) if isinstance(total_sell_value, pd.Series) or isinstance(total_buy_value, pd.Series) else total_sell_value - total_buy_value
        }
    
    def get_agents_performance(self):
        """Restituisce le performance degli agenti"""
        if not self.market_env or not self.simulation_results:
            return None
        
        agents_performance = []
        
        for agent in self.market_env.agents:
            initial_state = self.simulation_results[0]
            final_state = self.simulation_results[-1]
            
            # Trova l'agente nei risultati
            agent_initial = next((a for a in initial_state['agents'] if a['id'] == agent.id), None)
            agent_final = next((a for a in final_state['agents'] if a['id'] == agent.id), None)
            
            if agent_initial and agent_final:
                initial_value = float(agent_initial['total_value']) if isinstance(agent_initial['total_value'], pd.Series) else agent_initial['total_value']
                final_value = float(agent_final['total_value']) if isinstance(agent_final['total_value'], pd.Series) else agent_final['total_value']
                absolute_return = final_value - initial_value
                percentage_return = (absolute_return / initial_value) * 100 if initial_value > 0 else 0
                
                agents_performance.append({
                    'id': agent.id,
                    'initial_value': float(initial_value),
                    'final_value': float(final_value),
                    'absolute_return': float(absolute_return),
                    'percentage_return': float(percentage_return)
                })
        
        return agents_performance

    def get_strategy_class(self):
        """
        Restituisce la classe della strategia di trading in base alla configurazione
        
        Returns:
            Classe della strategia di trading
        """
        strategy_name = self.config['strategies']['active_strategy']
        
        if strategy_name == 'random':
            return RandomStrategy
        elif strategy_name == 'mean_reversion':
            return MeanReversionStrategy
        elif strategy_name == 'trend_following':
            return TrendFollowingStrategy
        elif strategy_name == 'value_investing':
            return ValueInvestingStrategy
        elif strategy_name == 'neural_network':
            return NeuralNetworkStrategy
        else:
            self.logger.warning(f"Strategia {strategy_name} non riconosciuta, uso strategia casuale")
            return RandomStrategy 