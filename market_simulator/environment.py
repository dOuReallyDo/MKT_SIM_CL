"""
Market Environment Module.

Questo modulo contiene la classe MarketEnvironment per la simulazione dell'ambiente di mercato.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random
import uuid
from functools import lru_cache
import psutil
import time
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import pickle
from logging.handlers import RotatingFileHandler

# Importazione condizionale di Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis non è installato. La cache distribuita non sarà disponibile.")

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
        self.logger = logging.getLogger('DistributedCache')
        self.redis_client = None
        
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis non è installato. La cache distribuita utilizzerà una cache in memoria.")
            return
            
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False
            )
            # Verifica che Redis sia disponibile con un ping
            self.redis_client.ping()
        except Exception as e:
            self.logger.error(f"Errore nella connessione a Redis: {e}")
            self.redis_client = None
            self.logger.warning("Redis non è disponibile. La cache distribuita utilizzerà una cache in memoria.")
    
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
    def __init__(self, data=None, start_date=None, end_date=None, opening_time="09:30", closing_time="16:00"):
        """
        Inizializza l'ambiente di mercato
        
        Args:
            data: Dizionario con i dati delle azioni, formato {symbol: dataframe}
            start_date: Data di inizio della simulazione (str o datetime)
            end_date: Data di fine della simulazione (str o datetime)
            opening_time: Orario di apertura del mercato
            closing_time: Orario di chiusura del mercato
        """
        self.logger = logging.getLogger('MarketEnvironment')
        
        # Inizializza i dati di mercato
        self.stocks_data = data or {}
        
        # Converti le date se necessario
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # Orari di mercato
        self.opening_time = opening_time
        self.closing_time = closing_time
        
        # Genera le date di trading
        self._initialize_trading_days()
        
        # Stato corrente della simulazione
        self.current_index = 0
        self.current_date = self.trading_days[0] if len(self.trading_days) > 0 else None
        
        # Strutture dati di base
        self.agents = []  # Lista degli agenti di trading
        self.transactions = []  # Storico delle transazioni
        self._price_cache = {}  # Cache dei prezzi per efficienza
        
        # Performance monitor
        self.performance_monitor = None  # Sarà inizializzato se necessario
        
        self.logger.info(f"Ambiente di mercato inizializzato: {len(self.stocks_data)} simboli, {len(self.trading_days)} giorni di trading")
    
    def _initialize_trading_days(self):
        """Inizializza le date di trading basate sui dati disponibili"""
        if not self.stocks_data:
            self.trading_days = []
            return
        
        # Trova tutte le date disponibili nei dati
        all_dates = set()
        for symbol, data in self.stocks_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                all_dates.update(data.index.date)
            else:
                try:
                    # Tenta di convertire l'indice in datetime
                    dates = pd.to_datetime(data.index).date
                    all_dates.update(dates)
                except:
                    self.logger.warning(f"Impossibile convertire l'indice in date per {symbol}")
        
        if not all_dates:
            self.trading_days = []
            return
        
        # Converti in lista e ordina
        all_dates = sorted(list(all_dates))
        
        # Filtra per date di inizio e fine
        if self.start_date:
            all_dates = [d for d in all_dates if pd.to_datetime(d) >= self.start_date]
        if self.end_date:
            all_dates = [d for d in all_dates if pd.to_datetime(d) <= self.end_date]
        
        # Converti le date in pd.Timestamp per coerenza
        self.trading_days = [pd.Timestamp(d) for d in all_dates]
        self.logger.info(f"Trading days inizializzati: {len(self.trading_days)} giorni")
    
    def add_agent(self, agent):
        """
        Aggiunge un agente alla simulazione
        
        Args:
            agent: Istanza di TradingAgent da aggiungere
        """
        if agent not in self.agents:
            self.agents.append(agent)
            self.logger.info(f"Agente {agent.id} aggiunto alla simulazione")
    
    def get_dates(self):
        """
        Restituisce la lista delle date di trading
        
        Returns:
            List: Lista delle date di trading
        """
        return self.trading_days
    
    def get_market_data(self, date):
        """
        Restituisce i dati di mercato per una data specifica
        
        Args:
            date: Data per cui ottenere i dati (str o datetime)
            
        Returns:
            Dict: Dizionario con i dati di mercato per tutti i simboli disponibili, formato:
                {
                    'AAPL': {'open': 150.0, 'high': 152.0, 'low': 149.0, 'close': 151.0, 'volume': 100000},
                    'MSFT': {...}
                }
        """
        # Converti la data in Timestamp se necessario
        date_ts = pd.to_datetime(date)
        
        # Dizionario dei risultati
        market_data = {}
        
        # Ottieni i dati per ciascun simbolo
        for symbol, data in self.stocks_data.items():
            try:
                # Verifica se la data è presente nei dati
                if date_ts in data.index:
                    # Ottieni i dati per questa data
                    market_data[symbol] = {
                        'open': float(data.loc[date_ts, 'Open']),
                        'high': float(data.loc[date_ts, 'High']),
                        'low': float(data.loc[date_ts, 'Low']),
                        'close': float(data.loc[date_ts, 'Close']),
                        'volume': float(data.loc[date_ts, 'Volume'])
                    }
            except Exception as e:
                self.logger.warning(f"Errore nell'ottenere i dati per {symbol} nella data {date_ts}: {e}")
        
        return market_data
    
    def get_current_market_data(self):
        """
        Recupera i dati di mercato della data corrente
        
        Returns:
            Dizionario con i dati di mercato correnti
        """
        if not self.current_date or self.current_index >= len(self.trading_days):
            return {}
        
        market_data = {}
        
        for symbol, data in self.stocks_data.items():
            try:
                if self.current_date in data.index:
                    row = data.loc[self.current_date]
                    market_data[symbol] = {
                        'open': float(row['Open']) if 'Open' in row else 0.0,
                        'high': float(row['High']) if 'High' in row else 0.0,
                        'low': float(row['Low']) if 'Low' in row else 0.0,
                        'close': float(row['Close']) if 'Close' in row else 0.0,
                        'volume': float(row['Volume']) if 'Volume' in row else 0.0,
                        'date': self.current_date
                    }
            except Exception as e:
                self.logger.error(f"Errore nel recupero dei dati di mercato per {symbol} in data {self.current_date}: {e}")
        
        return market_data
    
    def step(self):
        """
        Avanza la simulazione di un giorno
        
        Returns:
            bool: True se la simulazione è avanzata, False se è terminata
        """
        # Verifica se la simulazione è terminata
        if self.current_index >= len(self.trading_days) - 1:
            self.logger.info("La simulazione è terminata")
            return False
        
        # Avanza al giorno successivo
        self.current_index += 1
        self.current_date = self.trading_days[self.current_index]
        
        # Ottieni i dati di mercato per il giorno corrente
        market_data = self.get_current_market_data()
        
        # Esegui le azioni degli agenti
        day_transactions = []
        for agent in self.agents:
            # Ottieni il segnale di trading dall'agente
            signal = agent.generate_signal(market_data)
            
            # Esegui la transazione se c'è un segnale
            if signal:
                # Esegui l'operazione
                transaction = self.execute_transaction(
                    agent_id=agent.id,
                    symbol=signal['symbol'],
                    action=signal['action'],
                    quantity=signal['quantity'],
                    price=market_data[signal['symbol']]['close'] if signal['symbol'] in market_data else None
                )
                
                if transaction:
                    day_transactions.append(transaction)
        
        self.logger.info(f"Giorno {self.current_date}: {len(day_transactions)} transazioni eseguite")
        return True
    
    def execute_transaction(self, agent_id, symbol, action, quantity, price=None):
        """
        Esegue una transazione per un agente
        
        Args:
            agent_id: ID dell'agente
            symbol: Simbolo dell'asset
            action: Tipo di azione ('buy' o 'sell')
            quantity: Quantità da comprare/vendere
            price: Prezzo (se None, usa il prezzo di chiusura corrente)
            
        Returns:
            Dict: Dettagli della transazione o None se non riuscita
        """
        # Trova l'agente
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            self.logger.warning(f"Agente {agent_id} non trovato")
            return None
        
        # Verifica che il simbolo esista e ottieni il prezzo se non specificato
        if price is None:
            market_data = self.get_current_market_data()
            if symbol not in market_data:
                self.logger.warning(f"Simbolo {symbol} non disponibile per la data {self.current_date}")
                return None
            price = market_data[symbol]['close']
        
        # Esegui l'azione
        success = False
        if action.lower() == 'buy':
            success = agent.execute_buy(symbol, quantity, price)
        elif action.lower() == 'sell':
            success = agent.execute_sell(symbol, quantity, price)
        else:
            self.logger.warning(f"Azione {action} non valida")
            return None
        
        # Se l'operazione è riuscita, crea e salva la transazione
        if success:
            transaction = {
                'id': str(uuid.uuid4()),
                'date': self.current_date,
                'agent_id': agent_id,
                'symbol': symbol,
                'action': action.lower(),
                'quantity': quantity,
                'price': price,
                'total': price * quantity
            }
            self.transactions.append(transaction)
            return transaction
        
        return None
    
    def run_full_simulation(self):
        """
        Esegue la simulazione completa fino alla fine
        
        Returns:
            List: Lista delle transazioni effettuate
        """
        self.logger.info("Avvio simulazione completa")
        
        # Reset lo stato attuale se necessario
        self.current_index = 0
        if len(self.trading_days) > 0:
            self.current_date = self.trading_days[0]
        
        # Esegui la simulazione passo dopo passo
        while self.step():
            pass
        
        self.logger.info(f"Simulazione completata: {len(self.transactions)} transazioni effettuate")
        return self.transactions
    
    def get_agent_performance(self, agent_id):
        """
        Calcola le performance di un agente specifico
        
        Args:
            agent_id: ID dell'agente
            
        Returns:
            Dict: Metriche di performance dell'agente
        """
        # Trova l'agente
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            self.logger.warning(f"Agente {agent_id} non trovato")
            return None
        
        # Calcola le metriche usando il market data dell'ultimo giorno
        if len(self.trading_days) > 0 and self.current_date:
            market_data = self.get_market_data(self.current_date)
            return agent.get_performance_metrics(market_data)
        
        return None
    
    def get_all_agent_performances(self):
        """
        Calcola le performance di tutti gli agenti
        
        Returns:
            List: Lista di dizionari con le metriche di performance per ciascun agente
        """
        performances = []
        for agent in self.agents:
            perf = self.get_agent_performance(agent.id)
            if perf:
                performances.append(perf)
        
        return performances
    
    def get_transaction_history(self, agent_id=None, symbol=None, start_date=None, end_date=None):
        """
        Ottiene lo storico delle transazioni, con possibilità di filtraggio
        
        Args:
            agent_id: Filtra per ID agente
            symbol: Filtra per simbolo
            start_date: Filtra per data di inizio
            end_date: Filtra per data di fine
            
        Returns:
            List: Lista di transazioni filtrate
        """
        # Converte le date se necessario
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
        
        # Filtra le transazioni
        filtered_transactions = self.transactions.copy()
        
        if agent_id is not None:
            filtered_transactions = [t for t in filtered_transactions if t['agent_id'] == agent_id]
        
        if symbol is not None:
            filtered_transactions = [t for t in filtered_transactions if t['symbol'] == symbol]
        
        if start_date is not None:
            filtered_transactions = [t for t in filtered_transactions if pd.to_datetime(t['date']) >= start_date]
        
        if end_date is not None:
            filtered_transactions = [t for t in filtered_transactions if pd.to_datetime(t['date']) <= end_date]
        
        return filtered_transactions
    
    def get_portfolio_history(self, agent_id):
        """
        Ottiene lo storico del portafoglio di un agente
        
        Args:
            agent_id: ID dell'agente
            
        Returns:
            Dict: Storico del portafoglio per data
        """
        # Trova l'agente
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            self.logger.warning(f"Agente {agent_id} non trovato")
            return None
        
        # Filtra le transazioni dell'agente
        agent_transactions = self.get_transaction_history(agent_id=agent_id)
        
        # Ordina per data
        agent_transactions.sort(key=lambda x: pd.to_datetime(x['date']))
        
        # Inizializza lo storico del portafoglio
        portfolio_history = {
            'dates': [],
            'cash': [],
            'portfolio_value': [],
            'total_value': [],
            'asset_quantities': {},  # {symbol: [quantities]}
            'asset_values': {}       # {symbol: [values]}
        }
        
        # Inizializza i valori iniziali
        current_cash = agent.initial_capital
        current_portfolio = {}  # {symbol: quantity}
        
        # Itera su tutte le date di trading
        for date in self.trading_days:
            # Aggiorna il portafoglio con le transazioni di questa data
            date_transactions = [t for t in agent_transactions if pd.to_datetime(t['date']) == date]
            
            for transaction in date_transactions:
                symbol = transaction['symbol']
                action = transaction['action']
                quantity = transaction['quantity']
                total = transaction['total']
                
                if action == 'buy':
                    current_cash -= total
                    current_portfolio[symbol] = current_portfolio.get(symbol, 0) + quantity
                elif action == 'sell':
                    current_cash += total
                    current_portfolio[symbol] = current_portfolio.get(symbol, 0) - quantity
                    # Rimuovi se la quantità è 0
                    if current_portfolio[symbol] <= 0:
                        current_portfolio[symbol] = 0
            
            # Ottieni i dati di mercato per calcolare il valore del portafoglio
            market_data = self.get_market_data(date)
            portfolio_value = 0
            
            # Calcola il valore di ogni asset
            for symbol, quantity in current_portfolio.items():
                if symbol in market_data:
                    price = market_data[symbol]['close']
                    asset_value = price * quantity
                    portfolio_value += asset_value
                    
                    # Aggiungi alla storia dei valori degli asset
                    if symbol not in portfolio_history['asset_quantities']:
                        portfolio_history['asset_quantities'][symbol] = []
                        portfolio_history['asset_values'][symbol] = []
                    
                    portfolio_history['asset_quantities'][symbol].append(quantity)
                    portfolio_history['asset_values'][symbol].append(asset_value)
            
            # Aggiungi alla storia
            portfolio_history['dates'].append(date)
            portfolio_history['cash'].append(current_cash)
            portfolio_history['portfolio_value'].append(portfolio_value)
            portfolio_history['total_value'].append(current_cash + portfolio_value)
        
        return portfolio_history 