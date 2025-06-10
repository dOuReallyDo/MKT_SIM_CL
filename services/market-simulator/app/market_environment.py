import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import redis
import psutil
import time
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import pickle
from typing import Dict, List, Optional, Any
from .models import MarketData, Transaction

# Metriche Prometheus
TRADING_OPERATIONS = Counter('trading_operations_total', 'Numero totale di operazioni di trading')
CACHE_HITS = Counter('cache_hits_total', 'Numero di hit nella cache')
CACHE_MISSES = Counter('cache_misses_total', 'Numero di miss nella cache')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Uso della memoria in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'Uso della CPU in percentuale')
OPERATION_LATENCY = Histogram('operation_latency_seconds', 'Latenza delle operazioni')

class DistributedCache:
    """Cache distribuita basata su Redis"""
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False
            )
            self.logger = logging.getLogger('DistributedCache')
        except Exception as e:
            logging.error(f"Errore nella connessione a Redis: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        try:
            if not self.redis_client:
                return None
            data = self.redis_client.get(key)
            return pickle.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Errore nel recupero dalla cache: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        try:
            if not self.redis_client:
                return False
            data = pickle.dumps(value)
            return bool(self.redis_client.setex(key, expire, data))
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio nella cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
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
        self.logger = logging.getLogger('PerformanceMonitor')
        self.start_time = time.time()
        self.operations: Dict[str, Dict] = {}
        self.memory_samples: List[int] = []
        self.cpu_samples: List[float] = []
        
        try:
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
    
    def record_operation(self, operation_name: str, duration: float) -> None:
        if operation_name not in self.operations:
            self.operations[operation_name] = {
                'count': 0,
                'total_time': 0,
                'times': []
            }
        
        self.operations[operation_name]['count'] += 1
        self.operations[operation_name]['total_time'] += duration
        self.operations[operation_name]['times'].append(duration)
    
    def record_memory_usage(self) -> None:
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_samples.append(memory_info.rss)
        MEMORY_USAGE.set(memory_info.rss)
    
    def record_cpu_usage(self) -> None:
        cpu_percent = psutil.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        CPU_USAGE.set(cpu_percent)
    
    def get_performance_report(self) -> Dict[str, Any]:
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
    def __init__(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str, 
                 opening_time: str = "09:30", closing_time: str = "16:00"):
        self.stocks_data = data
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.opening_time = opening_time
        self.closing_time = closing_time
        
        self._price_cache: Dict[str, float] = {}
        self._market_data_cache: Dict[str, Any] = {}
        
        self.performance_monitor = PerformanceMonitor()
        self.distributed_cache = DistributedCache()
        self.logger = logging.getLogger('MarketEnvironment')
        
        self.trading_days = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        self.current_index = 0
        self.current_date = self.trading_days[0]
        self.transactions: List[Transaction] = []
        
        self._setup_logging()
        self._optimize_memory_usage()
        self._cache_size = 1000
    
    def _setup_logging(self) -> None:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'market_environment_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
    
    def _optimize_memory_usage(self) -> None:
        for symbol, data in self.stocks_data.items():
            self.stocks_data[symbol] = data.astype({
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'volume': 'int32'
            })
    
    def get_dates(self) -> List[datetime]:
        return self.trading_days.tolist()
    
    def get_market_data(self, date: datetime) -> Dict[str, MarketData]:
        cache_key = f"market_data_{date.strftime('%Y%m%d')}"
        
        # Prova a recuperare dalla cache distribuita
        cached_data = self.distributed_cache.get(cache_key)
        if cached_data:
            CACHE_HITS.inc()
            return cached_data
        
        CACHE_MISSES.inc()
        market_data = {}
        
        for symbol, data in self.stocks_data.items():
            if date in data.index:
                row = data.loc[date]
                market_data[symbol] = MarketData(
                    symbol=symbol,
                    date=date,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
        
        # Salva nella cache distribuita
        self.distributed_cache.set(cache_key, market_data)
        return market_data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        cache_key = f"price_{symbol}_{self.current_date.strftime('%Y%m%d')}"
        
        # Prova a recuperare dalla cache distribuita
        cached_price = self.distributed_cache.get(cache_key)
        if cached_price is not None:
            CACHE_HITS.inc()
            return cached_price
        
        CACHE_MISSES.inc()
        if symbol in self.stocks_data and self.current_date in self.stocks_data[symbol].index:
            price = float(self.stocks_data[symbol].loc[self.current_date, 'close'])
            self.distributed_cache.set(cache_key, price)
            return price
        return None
    
    def execute_transaction(self, agent_id: str, symbol: str, action: str, 
                          quantity: int, price: float) -> Optional[Transaction]:
        transaction = Transaction(
            agent_id=agent_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )
        
        self.transactions.append(transaction)
        TRADING_OPERATIONS.inc()
        return transaction
    
    def clear_cache(self) -> None:
        self._price_cache.clear()
        self._market_data_cache.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        return self.performance_monitor.get_performance_report() 