Piano di Progetto: Sistema di Trading Algoritmico basato su IA
1. Panoramica del Progetto
1.1 Descrizione
Sviluppo di un sistema integrato di simulazione del mercato azionario e trading algoritmico basato su reti neurali. Il sistema è composto da un simulatore di mercato, agenti di trading virtuali con diverse strategie, un framework di addestramento di reti neurali per la previsione dei prezzi, e un motore di inferenza per l'esecuzione di previsioni in tempo reale.
1.2 Obiettivi

Creare un ambiente di simulazione realistico per il mercato azionario
Implementare diverse strategie di trading algoritmico
Addestrare modelli di deep learning per la previsione dei prezzi delle azioni
Fornire un'interfaccia per l'analisi e la visualizzazione dei risultati
Ottimizzare le performance per l'esecuzione su Apple Silicon (M4)

1.3 Requisiti Funzionali

Simulazione del mercato con dati storici reali
Supporto per multiple strategie di trading
Addestramento di vari tipi di reti neurali (LSTM, CNN, Transformer)
Analisi della performance degli agenti e dei modelli
Visualizzazione dei risultati tramite dashboard interattiva
Ottimizzazione per hardware Apple Silicon

2. Architettura del Sistema
2.1 Diagramma dell'Architettura
Copia+------------------------+     +-------------------------+     +------------------------+
|                        |     |                         |     |                        |
|  Data Collection &     |---->|  Market Simulation &    |---->|  Neural Network        |
|  Preprocessing         |     |  Agent Management       |     |  Training              |
|                        |     |                         |     |                        |
+------------------------+     +-------------------------+     +------------------------+
                                                                         |
                                                                         v
+------------------------+     +-------------------------+     +------------------------+
|                        |     |                         |     |                        |
|  Dashboard &           |<----|  Trading Strategy       |<----|  Inference Engine &    |
|  Visualization         |     |  Execution              |     |  Prediction            |
|                        |     |                         |     |                        |
+------------------------+     +-------------------------+     +------------------------+
2.2 Componenti Principali

Data Collection & Preprocessing

Raccolta dati storici di mercato
Normalizzazione e feature engineering
Gestione dei dataset per training e testing


Market Simulation & Agent Management

Ambiente di simulazione del mercato
Gestione degli agenti di trading
Implementazione di varie strategie di trading


Neural Network Training

Implementazione di diverse architetture di reti neurali
Addestramento e validazione dei modelli
Gestione dei modelli addestrati


Inference Engine & Prediction

Caricamento dei modelli addestrati
Esecuzione di previsioni in tempo reale
Valutazione delle performance predittive


Trading Strategy Execution

Esecuzione di strategie basate su AI
Integrazione con i modelli di previsione
Backtesting e ottimizzazione delle strategie


Dashboard & Visualization

Interfaccia utente per la visualizzazione dei risultati
Analisi delle performance degli agenti e dei modelli
Generazione di report e grafici



3. Implementazione Dettagliata
3.1 Data Collection & Preprocessing
3.1.1 Funzionalità

Download di dati storici da varie fonti (Yahoo Finance, Alpha Vantage, ecc.)
Pulizia e normalizzazione dei dati
Feature engineering per la preparazione dei dati di addestramento
Gestione della persistenza dei dati

3.1.2 Implementazione
pythonCopia# data/collector.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

class DataCollector:
    """Classe per la raccolta e gestione dei dati di mercato"""
    def __init__(self, data_dir='./data'):
        """
        Inizializza il collector dei dati
        
        Args:
            data_dir: Directory per il salvataggio dei dati
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Configurazione del logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{data_dir}/data_collection.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataCollector')
    
    def get_stock_data(self, symbol, start_date, end_date, force_download=False):
        """
        Ottiene i dati storici di un titolo
        
        Args:
            symbol: Simbolo del titolo
            start_date: Data di inizio
            end_date: Data di fine
            force_download: Se True, forza il download anche se i dati sono in cache
            
        Returns:
            DataFrame con i dati del titolo
        """
        # Percorso del file di cache
        cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
        
        # Controllo se i dati sono già disponibili in cache
        if os.path.exists(cache_file) and not force_download:
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Verifica delle date
                if df.index.min() <= pd.to_datetime(start_date) and df.index.max() >= pd.to_datetime(end_date):
                    self.logger.info(f"Dati per {symbol} caricati dalla cache")
                    return df
            except Exception as e:
                self.logger.warning(f"Errore nel caricamento dei dati dalla cache: {e}")
        
        # Download dei dati
        try:
            self.logger.info(f"Download dei dati per {symbol} da {start_date} a {end_date}")
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                self.logger.warning(f"Nessun dato disponibile per {symbol}")
                return None
            
            # Salvataggio dei dati in cache
            df.to_csv(cache_file)
            
            return df
        except Exception as e:
            self.logger.error(f"Errore nel download dei dati per {symbol}: {e}")
            return None
    
    def get_multiple_stocks_data(self, symbols, start_date, end_date):
        """
        Ottiene i dati storici per multiple azioni
        
        Args:
            symbols: Lista di simboli
            start_date: Data di inizio
            end_date: Data di fine
            
        Returns:
            Dizionario di DataFrame con i dati delle azioni
        """
        result = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, start_date, end_date)
            if data is not None:
                result[symbol] = data
        
        return result
    
    def prepare_features(self, df, window_sizes=[5, 10, 20, 50]):
        """
        Calcola feature aggiuntive dai dati di prezzo
        
        Args:
            df: DataFrame con i dati di prezzo
            window_sizes: Liste di dimensioni delle finestre per le medie mobili
            
        Returns:
            DataFrame con le feature aggiunte
        """
        features_df = df.copy()
        
        # Calcolo dei rendimenti
        features_df['Returns'] = df['Close'].pct_change()
        
        # Calcolo delle medie mobili
        for window in window_sizes:
            features_df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            features_df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Calcolo della volatilità
        features_df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        features_df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Calcolo del MACD
        features_df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        features_df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        features_df['MACD'] = features_df['EMA_12'] - features_df['EMA_26']
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Rimozione delle righe con NaN
        features_df = features_df.dropna()
        
        return features_df
3.2 Market Simulation & Agent Management
3.2.1 Funzionalità

Simulazione dell'ambiente di mercato (prezzi, volumi, orari di trading)
Implementazione di agenti virtuali di trading
Supporto per diverse strategie di trading
Registrazione e analisi delle transazioni

3.2.2 Implementazione
pythonCopia# market_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid
import os
import logging

class MarketEnvironment:
    def __init__(self, stocks_data, trading_days, opening_time="09:30", closing_time="16:00"):
        """
        Ambiente di simulazione del mercato
        
        Args:
            stocks_data: Dict di DataFrame con dati storici per ogni titolo
            trading_days: Lista di date per la simulazione
            opening_time: Orario di apertura del mercato
            closing_time: Orario di chiusura del mercato
        """
        self.stocks_data = stocks_data
        self.trading_days = trading_days
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.current_day_idx = 0
        self.agents = []
        self.transactions = []
        
        # Configurazione del logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/market_simulation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MarketEnvironment')
        
        os.makedirs('./logs', exist_ok=True)
        
    def add_agent(self, agent):
        """Aggiunge un agente alla simulazione"""
        self.agents.append(agent)
        self.logger.info(f"Agente {agent.id} aggiunto alla simulazione")
    
    def step(self):
        """Avanza di un giorno nella simulazione"""
        if self.current_day_idx >= len(self.trading_days):
            return False  # Simulazione terminata
        
        current_date = self.trading_days[self.current_day_idx]
        self.logger.info(f"Simulazione del giorno {current_date}")
        
        # Raccolta prezzi correnti
        current_prices = {}
        for symbol, data in self.stocks_data.items():
            if current_date in data.index:
                current_prices[symbol] = {
                    'open': data.loc[current_date, 'Open'],
                    'high': data.loc[current_date, 'High'],
                    'low': data.loc[current_date, 'Low'],
                    'close': data.loc[current_date, 'Close'],
                    'volume': data.loc[current_date, 'Volume']
                }
        
        # Esecuzione delle azioni degli agenti
        day_transactions = []
        for agent in self.agents:
            actions = agent.decide_actions(current_prices, current_date)
            for action in actions:
                success, transaction = self._execute_transaction(agent, action, current_prices, current_date)
                if success:
                    day_transactions.append(transaction)
                    self.transactions.append(transaction)
        
        self.logger.info(f"Giorno {current_date}: {len(day_transactions)} transazioni eseguite")
        self.current_day_idx += 1
        return True
    
    def _execute_transaction(self, agent, action, prices, date):
        """Esegue una transazione di un agente"""
        symbol = action['symbol']
        quantity = action['quantity']
        price_data = prices.get(symbol)
        
        if price_data is None:
            self.logger.warning(f"Simbolo {symbol} non disponibile per la data {date}")
            return False, None
        
        price = price_data['close']
        
        transaction = {
            'id': str(uuid.uuid4()),
            'date': date,
            'agent_id': agent.id,
            'symbol': symbol,
            'type': action['type'],
            'quantity': quantity,
            'price': price,
            'value': price * quantity
        }
        
        if action['type'] == 'buy':
            cost = price * quantity
            if agent.cash >= cost:
                agent.cash -= cost
                agent.portfolio[symbol] = agent.portfolio.get(symbol, 0) + quantity
                self.logger.debug(f"Agente {agent.id} ha acquistato {quantity} azioni di {symbol} a {price}")
                return True, transaction
            else:
                self.logger.debug(f"Agente {agent.id} non ha abbastanza contante per acquistare {symbol}")
        elif action['type'] == 'sell':
            if agent.portfolio.get(symbol, 0) >= quantity:
                agent.cash += price * quantity
                agent.portfolio[symbol] -= quantity
                self.logger.debug(f"Agente {agent.id} ha venduto {quantity} azioni di {symbol} a {price}")
                return True, transaction
            else:
                self.logger.debug(f"Agente {agent.id} non ha abbastanza azioni di {symbol} da vendere")
        
        return False, None
    
    def get_market_state(self):
        """Restituisce lo stato corrente del mercato"""
        if self.current_day_idx >= len(self.trading_days):
            return None
            
        current_date = self.trading_days[self.current_day_idx]
        market_state = {
            'date': current_date,
            'prices': {},
            'agents': []
        }
        
        for symbol, data in self.stocks_data.items():
            if current_date in data.index:
                market_state['prices'][symbol] = {
                    'open': data.loc[current_date, 'Open'],
                    'high': data.loc[current_date, 'High'],
                    'low': data.loc[current_date, 'Low'],
                    'close': data.loc[current_date, 'Close'],
                    'volume': data.loc[current_date, 'Volume']
                }
        
        for agent in self.agents:
            market_state['agents'].append({
                'id': agent.id,
                'cash': agent.cash,
                'portfolio': agent.portfolio,
                'total_value': agent.get_total_value(market_state['prices'])
            })
        
        return market_state
    
    def run_simulation(self):
        """Esegue la simulazione completa"""
        self.logger.info(f"Avvio simulazione: {len(self.trading_days)} giorni, {len(self.agents)} agenti")
        
        results = []
        while self.step():
            state = self.get_market_state()
            if state:
                results.append(state)
        
        self.logger.info(f"Simulazione completata: {len(self.transactions)} transazioni totali")
        return results


class Agent:
    def __init__(self, id, initial_capital, strategy):
        """
        Agente virtuale per la simulazione di mercato
        
        Args:
            id: Identificatore univoco dell'agente
            initial_capital: Capitale iniziale in USD
            strategy: Strategia di trading da utilizzare
        """
        self.id = id
        self.cash = initial_capital
        self.portfolio = {}  # {symbol: quantity}
        self.strategy = strategy
        self.transaction_history = []
        
        # Configurazione del logging
        self.logger = logging.getLogger(f'Agent_{id}')
    
    def decide_actions(self, current_prices, current_date):
        """
        Decide le azioni da intraprendere in base alla strategia
        
        Returns:
            Lista di azioni del tipo {'type': 'buy/sell', 'symbol': symbol, 'quantity': quantity}
        """
        return self.strategy.generate_actions(self, current_prices, current_date)
    
    def get_total_value(self, prices):
        """Calcola il valore patrimoniale totale dell'agente"""
        portfolio_value = sum(prices.get(symbol, {}).get('close', 0) * quantity 
                              for symbol, quantity in self.portfolio.items())
        return self.cash + portfolio_value
    
    def get_performance_metrics(self, initial_date, final_date, prices_history):
        """
        Calcola le metriche di performance dell'agente
        
        Args:
            initial_date: Data iniziale
            final_date: Data finale
            prices_history: Storico dei prezzi
            
        Returns:
            Dizionario con le metriche di performance
        """
        # Calcolo del valore iniziale
        initial_value = self.cash
        
        # Calcolo del valore finale
        final_prices = {}
        for symbol, data in prices_history.items():
            if final_date in data.index:
                final_prices[symbol] = data.loc[final_date, 'Close']
        
        final_value = self.cash + sum(final_prices.get(symbol, 0) * quantity 
                                      for symbol, quantity in self.portfolio.items())
        
        # Calcolo delle metriche
        absolute_return = final_value - initial_value
        percentage_return = (absolute_return / initial_value) * 100 if initial_value > 0 else 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'absolute_return': absolute_return,
            'percentage_return': percentage_return
        }


class TradingStrategy:
    """Classe base per le strategie di trading"""
    def generate_actions(self, agent, prices, date):
        raise NotImplementedError("Metodo da implementare nelle sottoclassi")


class RandomStrategy(TradingStrategy):
    """Strategia casuale per il trading"""
    def generate_actions(self, agent, prices, date):
        actions = []
        # Implementazione di una strategia casuale
        if random.random() < 0.3:  # 30% di probabilità di agire
            action_type = random.choice(['buy', 'sell'])
            
            if action_type == 'buy':
                # Scelta casuale di un titolo
                available_symbols = list(prices.keys())
                if not available_symbols:
                    return []
                    
                symbol = random.choice(available_symbols)
                price = prices[symbol]['close']
                max_quantity = int(agent.cash / price * 0.9) if price > 0 else 0  # 90% del capitale disponibile
                
                if max_quantity > 0:
                    quantity = random.randint(1, max_quantity)
                    actions.append({
                        'type': 'buy',
                        'symbol': symbol,
                        'quantity': quantity
                    })
            else:
                # Vendita casuale da portafoglio
                portfolio_symbols = [s for s, q in agent.portfolio.items() if q > 0]
                if portfolio_symbols:
                    symbol = random.choice(portfolio_symbols)
                    quantity = random.randint(1, agent.portfolio[symbol])
                    actions.append({
                        'type': 'sell',
                        'symbol': symbol,
                        'quantity': quantity
                    })
        
        return actions


class MeanReversionStrategy(TradingStrategy):
    """Strategia di mean reversion per il trading"""
    def __init__(self, window=20):
        self.window = window
        self.historical_prices = {}  # {symbol: [prices]}
    
    def generate_actions(self, agent, prices, date):
        actions = []
        
        # Aggiornamento prezzi storici
        for symbol, price_data in prices.items():
            if symbol not in self.historical_prices:
                self.historical_prices[symbol] = []
            self.historical_prices[symbol].append(price_data['close'])
            
            # Calcolo media mobile
            if len(self.historical_prices[symbol]) >= self.window:
                recent_prices = self.historical_prices[symbol][-self.window:]
                mean_price = sum(recent_prices) / len(recent_prices)
                current_price = price_data['close']
                
                # Decisione di trading
                if current_price < mean_price * 0.95:  # Prezzo inferiore del 5% alla media
                    # Acquisto
                    max_quantity = int(agent.cash / current_price * 0.5) if current_price > 0 else 0  # 50% del capitale disponibile
                    if max_quantity > 0:
                        actions.append({
                            'type': 'buy',
                            'symbol': symbol,
                            'quantity': max_quantity
                        })
                elif current_price > mean_price * 1.05:  # Prezzo superiore del 5% alla media
                    # Vendita
                    if symbol in agent.portfolio and agent.portfolio[symbol] > 0:
                        actions.append({
                            'type': 'sell',
                            'symbol': symbol,
                            'quantity': agent.portfolio[symbol]  # Vendita totale
                        })
        
        return actions


class TrendFollowingStrategy(TradingStrategy):
    """Strategia di trend following per il trading"""
    def __init__(self, short_window=10, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        self.historical_prices = {}  # {symbol: [prices]}
    
    def generate_actions(self, agent, prices, date):
        actions = []
        
        # Aggiornamento prezzi storici
        for symbol, price_data in prices.items():
            if symbol not in self.historical_prices:
                self.historical_prices[symbol] = []
            self.historical_prices[symbol].append(price_data['close'])
            
            # Calcolo medie mobili
            if len(self.historical_prices[symbol]) >= self.long_window:
                short_prices = self.historical_prices[symbol][-self.short_window:]
                short_ma = sum(short_prices) / len(short_prices)
                
                long_prices = self.historical_prices[symbol][-self.long_window:]
                long_ma = sum(long_prices) / len(long_prices)
                
                # Decisione di trading
                if short_ma > long_ma:  # Trend rialzista
                    # Acquisto
                    max_quantity = int(agent.cash / price_data['close'] * 0.5) if price_data['close'] > 0 else 0  # 50% del capitale disponibile
                    if max_quantity > 0:
                        actions.append({
                            'type': 'buy',
                            'symbol': symbol,
                            'quantity': max_quantity
                        })
                elif short_ma < long_ma:  # Trend ribassista
                    # Vendita
                    if symbol in agent.portfolio and agent.portfolio[symbol] > 0:
                        actions.append({
                            'type': 'sell',
                            'symbol': symbol,
                            'quantity': agent.portfolio[symbol]  # Vendita totale
                        })
        
        return actions


class ValueInvestingStrategy(TradingStrategy):
    """Strategia di value investing per il trading"""
    def __init__(self, min_pe_ratio=10, max_pe_ratio=20):
        self.min_pe_ratio = min_pe_ratio
        self.max_pe_ratio = max_pe_ratio
        self.fundamentals = {}  # {symbol: {'pe_ratio': pe_ratio, ...}}
    
    def update_fundamentals(self, symbol, pe_ratio):
        """Aggiorna i dati fondamentali di un titolo"""
        self.fundamentals[symbol] = {'pe_ratio': pe_ratio}
    
    def generate_actions(self, agent, prices, date):
        actions = []
        
        for symbol, price_data in prices.items():
            if symbol in self.fundamentals:
                pe_ratio = self.fundamentals[symbol]['pe_ratio']
                
                # Decisione di trading
                if pe_ratio < self.min_pe_ratio:  # Titolo sottovalutato
                    # Acquisto
                    max_quantity = int(agent.cash / price_data['close'] * 0.5) if price_data['close'] > 0 else 0  # 50% del capitale disponibile
                    if max_quantity > 0:
                        actions.append({
                            'type': 'buy',
                            'symbol': symbol,
                            'quantity': max_quantity
                        })
                elif pe_ratio > self.max_pe_ratio:  # Titolo sopravvalutato
                    # Vendita
                    if symbol in agent.portfolio and agent.portfolio[symbol] > 0:
                        actions.append({
                            'type': 'sell',
                            'symbol': symbol,
                            'quantity': agent.portfolio[symbol]  # Vendita totale
                        })
        
        return actions


class NeuralNetworkStrategy(TradingStrategy):
    """Strategia basata su reti neurali per il trading"""
    def __init__(self, model_trainer, sequence_length=10):
        self.model_trainer = model_trainer
        self.sequence_length = sequence_length
        self.historical_data = {}  # {symbol: DataFrame}
    
    def update_historical_data(self, symbol, data):
        """Aggiorna i dati storici per un titolo"""
        self.historical_data[symbol] = data
    
    def generate_actions(self, agent, prices, date):
        actions = []
        
        for symbol, price_data in prices.items():
            if symbol in self.historical_data and len(self.historical_data[symbol]) >= self.sequence_length:
                # Preparazione dei dati per la previsione
                recent_data = self.historical_data[symbol].tail(self.sequence_length)
                
                try:
                    # Previsione del prezzo
                    next_price = self.model_trainer.predict(recent_data.values)
                    current_price = price_data['close']
                    
                    # Decisione di trading basata sulla previsione
                    price_change_pct = (next_price - current_price) / current_price * 100
                    
                    if price_change_pct > 2:  # Aumento previsto superiore al 2%
                        # Acquisto
                        max_quantity = int(agent.cash / current_price * 0.3) if current_price > 0 else 0  # 30% del capitale disponibile
                        if max_quantity > 0:
                            actions.append({
                                'type': 'buy',
                                'symbol': symbol,
                                'quantity': max_quantity
                            })
                    elif price_change_pct < -2:  # Diminuzione prevista superiore al 2%
                        # Vendita
                        if symbol in agent.portfolio and agent.portfolio[symbol] > 0:
                            actions.append({
                                'type': 'sell',
                                'symbol': symbol,
                                'quantity': agent.portfolio[symbol]  # Vendita totale
                            })
                except Exception as e:
                    logging.error(f"Errore nella previsione per {symbol}: {e}")
        
        return actions


class SimulationManager:
    """Gestore della simulazione di mercato"""
    def __init__(self, config):
        """
        Inizializza il gestore della simulazione
        
        Args:
            config: Dizionario di configurazione della simulazione
        """
        self.config = config
        self.data_collector = DataCollector()
        self.market_env = None
        self.simulation_results = None
        
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
        
    def initialize_simulation(self):
        """Inizializza l'ambiente di simulazione"""
        # Raccolta dati
        symbols = self.config['symbols']
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        
        # Creazione directory dati
        os.makedirs('./data/simulation', exist_ok=True)
        
        # Download dati
        self.logger.info(f"Download dei dati per {len(symbols)} simboli")
        stocks_data = {}
        for symbol in symbols:
            data = self.data_collector.get_stock_data(symbol, start_date, end_date)
            if data is not None:
                stocks_data[symbol] = data
        
        # Creazione ambiente di mercato
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
        self.market_env = MarketEnvironment(
            stocks_data, 
            trading_days,
            opening_time=self.config['opening_time'],
            closing_time=self.config['closing_time']
        )
        
        # Creazione agenti
        self.logger.info(f"Creazione di {self.config['num_agents']} agenti")
        
        # Definizione delle strategie disponibili
        strategies = {
            'random': RandomStrategy(),
            'mean_reversion': MeanReversionStrategy(window=self.config.get('mean_reversion_window', 20)),
            'trend_following': TrendFollowingStrategy(
                short_window=self.config.get('trend_short_window', 10),
                long_window=self.config.get('trend_long_window', 50)
            ),
            'value_investing': ValueInvestingStrategy(
                min_pe_ratio=self.config.get('min_pe_ratio', 10),
                max_pe_ratio=self.config.get('max_pe_ratio', 20)
            )
        }
        
        # Creazione degli agenti con strategia specificata o casuale
        for i in range(self.config['num_agents']):
            strategy_type = self.config.get('strategy_type', 'random')
            if strategy_type == 'mixed':
                strategy = random.choice(list(strategies.values()))
            else:
                strategy = strategies.get(strategy_type, strategies['random'])
            
            min_capital = self.config.get('min_initial_capital', 10000)
            max_capital = self.config.get('max_initial_capital', 100000)
            initial_capital = random.uniform(min_capital, max_capital)
            
            agent = Agent(f"agent_{i}", initial_capital, strategy)
            self.market_env.add_agent(agent)
            
            # Aggiunta di dati fondamentali per le strategie di value investing
            if isinstance(strategy, ValueInvestingStrategy):
                for symbol in symbols:
                    pe_ratio = random.uniform(5, 30)  # Valore casuale per la simulazione
                    strategy.update_fundamentals(symbol, pe_ratio)
    
    def run_simulation(self):
        """Esegue la simulazione completa"""
        if self.market_env is None:
            self.logger.info("Inizializzazione della simulazione")
            self.initialize_simulation()
        
        self.logger.info("Avvio della simulazione")
        self.simulation_results = self.market_env.run_simulation()
        
        # Salvataggio risultati
        self.logger.info("Salvataggio dei risultati della simulazione")
        self._save_simulation_results()
        
        return self.simulation_results
    
    def _save_simulation_results(self):
        """Salva i risultati della simulazione per l'addestramento della rete neurale"""
        if self.simulation_results is None:
            self.logger.warning("Nessun risultato da salvare")
            return
        
        # Creazione DataFrame dei prezzi
        prices_data = []
        for state in self.simulation_results:
            date = state['date']
            for symbol, price_data in state['prices'].items():
                prices_data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': price_data['open'],
                    'high': price_data['high'],
                    'low': price_data['low'],
                    'close': price_data['close'],
                    'volume': price_data['volume']
                })
        
        prices_df = pd.DataFrame(prices_data)
        prices_df.to_csv('./data/simulation/prices.csv', index=False)
        self.logger.info(f"Salvato DataFrame dei prezzi: {len(prices_df)} righe")
        
        # Creazione DataFrame degli agenti
        agents_data = []
        for state in self.simulation_results:
            date = state['date']
            for agent_state in state['agents']:
                agents_data.append({
                    'date': date,
                    'agent_id': agent_state['id'],
                    'cash': agent_state['cash'],
                    'total_value': agent_state['total_value']
                })
        
        agents_df = pd.DataFrame(agents_data)
        agents_df.to_csv('./data/simulation/agents.csv', index=False)
        self.logger.info(f"Salvato DataFrame degli agenti: {len(agents_df)} righe")
        
        # Creazione DataFrame del portafoglio
        portfolio_data = []
        for state in self.simulation_results:
            date = state['date']
            for agent_state in state['agents']:
                for symbol, quantity in agent_state['portfolio'].items():
                    portfolio_data.append({
                        'date': date,
                        'agent_id': agent_state['id'],
                        'symbol': symbol,
                        'quantity': quantity
                    })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv('./data/simulation/portfolio.csv', index=False)
        self.logger.info(f"Salvato DataFrame del portafoglio: {len(portfolio_df)} righe")
        
        # Salvataggio delle transazioni
        transactions_df = pd.DataFrame(self.market_env.transactions)
        transactions_df.to_csv('./data/simulation/transactions.csv', index=False)
        self.logger.info(f"Salvato DataFrame delle transazioni: {len(transactions_df)} righe")
3.3 Neural Network Trainer
3.3.1 Funzionalità

Preparazione dei dati di training da simulazioni di mercato
Implementazione di diverse architetture di reti neurali
Addestramento configurabile (num. layers, neuroni, epoche, ecc.)
Salvataggio e caricamento dei modelli addestrati
Visualizzazione delle performance e delle metriche di addestramento

3.3.2 Implementazione
pythonCopia# neural_network/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging

class DataPreprocessor:
    """Classe per il preprocessing dei dati di mercato"""
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        
        # Configurazione del logging
        self.logger = logging.getLogger('DataPreprocessor')
    
    def prepare_data(self, prices_df, target_symbols=None):
        """
        Prepara i dati per l'addestramento della rete neurale
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            target_symbols: Lista di simboli target (None = tutti)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if target_symbols is None:
            target_symbols = prices_df['symbol'].unique().tolist()
        
        self.logger.info(f"Preparazione dati per {len(target_symbols)} simboli")
        
        # Filtraggio dati
        df = prices_df[prices_df['symbol'].isin(target_symbols)].copy()
        
        # Ordinamento per data
        df = df.sort_values(['symbol', 'date'])
        
        # Normalizzazione
        df_grouped = df.groupby('symbol')
        
        sequences = []
        targets = []
        
        for symbol, group in df_grouped:
            prices = group[['open', 'high', 'low', 'close']].values
            volumes = group['volume'].values.reshape(-1, 1)
            
            # Normalizzazione
            normalized_prices = self.price_scaler.fit_transform(prices)
            normalized_volumes = self.volume_scaler.fit_transform(volumes)
            
            # Creazione sequenze
            for i in range(len(group) - self.sequence_length - 1):
                # Sequenza di input
                seq_prices = normalized_prices[i:i+self.sequence_length]
                seq_volumes = normalized_volumes[i:i+self.sequence_length]
                
                # Target (prezzo di chiusura normalizzato del giorno successivo)
                target = normalized_prices[i+self.sequence_length, 3]  # Indice 3 = prezzo di chiusura
                
                # Combinazione di prezzi e volumi
                seq_combined = np.column_stack((seq_prices, seq_volumes))
                
                sequences.append(seq_combined)
                targets.append(target)
        
        # Conversione in array numpy
        X = np.array(sequences)
        y = np.array(targets)
        
        self.logger.info(f"Creati {len(sequences)} esempi di training")
        
        # Split in train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_price(self, normalized_price):
        """Converte un prezzo normalizzato nel suo valore originale"""
        # Creazione di un array fittizio con tutti i valori necessari
        dummy = np.zeros((1, 4))
        dummy[0, 3] = normalized_price  # Impostiamo il prezzo di chiusura
        
        # Denormalizzazione
        return self.price_scaler.inverse_transform(dummy)[0, 3]


class LSTMModel(nn.Module):
    """Modello LSTM per la previsione dei prezzi"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Inizializzazione dello stato nascosto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagation
        out, _ = self.lstm(x, (h0, c0))
        
        # Prendiamo solo l'output dell'ultimo timestep
        out = self.fc(out[:, -1, :])
        
        return out


class CNNModel(nn.Module):
    """Modello CNN per la previsione dei prezzi"""
    def __init__(self, input_dim, seq_length, num_filters=64, kernel_size=3, dropout=0.2):
        super(CNNModel, self).__init__()
        
        self.cnn1 = nn.Conv1d(input_dim, num_filters, kernel_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)
        
        self.cnn2 = nn.Conv1d(num_filters, num_filters*2, kernel_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2)
        
        # Calcolo della dimensione dell'output dopo le convoluzioni
        def calc_output_size(size, kernel_size, stride=1):
            return (size - kernel_size) // stride + 1
        
        cnn1_out = calc_output_size(seq_length, kernel_size)
        pool1_out = cnn1_out // 2
        cnn2_out = calc_output_size(pool1_out, kernel_size)
        pool2_out = cnn2_out // 2
        
        self.fc_input_dim = num_filters * 2 * pool2_out
        self.fc1 = nn.Linear(self.fc_input_dim, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # Trasposizione per la convoluzione 1D
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_length)
        
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Appiattimento
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerModel(nn.Module):
    """Modello Transformer per la previsione dei prezzi"""
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layer per trasformare l'input in dimensione d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Layer di codifica posizionale
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        
        # Encoder Transformer
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers=num_encoder_layers)
        
        # Layer di output
        self.decoder = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Embedding dell'input
        x = self.input_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Trasposizione per il transformer (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Codifica Transformer
        output = self.transformer_encoder(x)
        
        # Utilizzo dell'output dell'ultimo timestep
        output = output[-1, :, :]  # (batch_size, d_model)
        
        # Layer di output
        output = self.decoder(output)  # (batch_size, 1)
        
        return output


class ModelTrainer:
    """Classe per l'addestramento dei modelli di rete neurale"""
    def __init__(self, model_type, config):
        """
        Inizializza il trainer del modello
        
        Args:
            model_type: Tipo di modello ('lstm', 'cnn', 'transformer')
            config: Configurazione del modello e dell'addestramento
        """
        self.model_type = model_type
        self.config = config
        self.model = None
        self.data_preprocessor = DataPreprocessor(sequence_length=config.get('sequence_length', 10))
        
        # Configurazione dell'hardware
        use_mps = torch.backends.mps.is_available() and config.get('use_mps', True)
        use_cuda = torch.cuda.is_available() and config.get('use_cuda', False)
        
        if use_mps:
            self.device = torch.device('mps')
        elif use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Configurazione del logging
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/model_trainer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'ModelTrainer_{model_type}')
        self.logger.info(f"Inizializzato trainer per modello {model_type} su dispositivo {self.device}")
    
    def create_model(self, input_dim):
        """Crea il modello specificato"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=self.config.get('hidden_dim', 64),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'cnn':
            self.model = CNNModel(
                input_dim=input_dim,
                seq_length=self.config.get('sequence_length', 10),
                num_filters=self.config.get('num_filters', 64),
                kernel_size=self.config.get('kernel_size', 3),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=input_dim,
                d_model=self.config.get('d_model', 64),
                nhead=self.config.get('nhead', 8),
                num_encoder_layers=self.config.get('num_encoder_layers', 6),
                dropout=self.config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Tipo di modello non supportato: {self.model_type}")
        
        self.model.to(self.device)
        self.logger.info(f"Creato modello {self.model_type}")
    
    def train(self, prices_df, target_symbols=None):
        """
        Addestra il modello sui dati forniti
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            target_symbols: Lista di simboli target (None = tutti)
        """
        # Preparazione dei dati
        X_train, X_test, y_train, y_test = self.data_preprocessor.prepare_data(prices_df, target_symbols)
        
        # Creazione del modello
        input_dim = X_train.shape[2]
        self.create_model(input_dim)
        
        # Conversione a tensori PyTorch
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Definizione dell'ottimizzatore e della funzione di perdita
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        # Addestramento
        num_epochs = self.config.get('num_epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        self.logger.info(f"Inizio addestramento: {num_epochs} epoche, batch size {batch_size}")
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        epoch_losses = []
        test_losses = []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                
                # Backward pass e ottimizzazione
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            epoch_losses.append(epoch_loss)
            
            # Valutazione sul set di test
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs.squeeze(), y_test).item()
                test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}')
        
        # Calcolo delle metriche finali
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test).squeeze().cpu().numpy()
            y_true = y_test.cpu().numpy()
            
            mse = ((y_pred - y_true) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = np.abs(y_pred - y_true).mean()
            
            self.logger.info(f"Metriche finali - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
        
        # Salvataggio del modello
        model_path, preprocessor_path = self._save_model()
        
        # Grafico delle perdite
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Training and Test Loss for {self.model_type.upper()} Model')
        plt.savefig(f'./models/{self.model_type}_loss.png')
        
        return {
            'epoch_losses': epoch_losses,
            'test_losses': test_losses,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path
        }
    
    def _save_model(self):
        """Salva il modello addestrato"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'./models/{self.model_type}_{timestamp}.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_type': self.model_type
        }, model_path)
        
        # Salvataggio dei parametri del data preprocessor
        preprocessor_path = f'./models/{self.model_type}_{timestamp}_preprocessor.pkl'
        torch.save({
            'price_scaler': self.data_preprocessor.price_scaler,
            'volume_scaler': self.data_preprocessor.volume_scaler,
            'sequence_length': self.data_preprocessor.sequence_length
        }, preprocessor_path)
        
        self.logger.info(f"Modello salvato in {model_path}")
        self.logger.info(f"Preprocessor salvato in {preprocessor_path}")
        
        return model_path, preprocessor_path
    
    def load_model(self, model_path, preprocessor_path):
        """Carica un modello salvato"""
        self.logger.info(f"Caricamento modello da {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.config = checkpoint['config']
        
        # Caricamento del preprocessor
        preprocessor_checkpoint = torch.load(preprocessor_path, map_location=self.device)
        self.data_preprocessor.price_scaler = preprocessor_checkpoint['price_scaler']
        self.data_preprocessor.volume_scaler = preprocessor_checkpoint['volume_scaler']
        self.data_preprocessor.sequence_length = preprocessor_checkpoint['sequence_length']
        
        # Creazione del modello
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_dim=5,  # 4 prezzi + volume
                hidden_dim=self.config.get('hidden_dim', 64),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'cnn':
            self.model = CNNModel(
                input_dim=5,  # 4 prezzi + volume
                seq_length=self.config.get('sequence_length', 10),
                num_filters=self.config.get('num_filters', 64),
                kernel_size=self.config.get('kernel_size', 3),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=5,  # 4 prezzi + volume
                d_model=self.config.get('d_model', 64),
                nhead=self.config.get('nhead', 8),
                num_encoder_layers=self.config.get('num_encoder_layers', 6),
                dropout=self.config.get('dropout', 0.1)
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Modello {self.model_type} caricato con successo")
        
        return self
    
    def predict(self, sequence):
        """
        Effettua una previsione con il modello addestrato
        
        Args:
            sequence: Sequenza di dati di input
        
        Returns:
            Previsione del prezzo
        """
        if self.model is None:
            raise ValueError("Modello non caricato o addestrato")
        
        # Conversione a tensore PyTorch
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Previsione
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        # Conversione della previsione al valore originale
        normalized_prediction = prediction.item()
        actual_prediction = self.data_preprocessor.inverse_transform_price(normalized_prediction)
        
        return actual_prediction
3.4 Inference Engine
3.4.1 Funzionalità

Caricamento di modelli addestrati
Preparazione dei dati per l'inferenza
Esecuzione delle previsioni
Valutazione delle performance predittive

3.4.2 Implementazione
pythonCopia# neural_network/inference.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

class InferenceEngine:
    """Motore di inferenza per l'esecuzione di previsioni con modelli addestrati"""
    def __init__(self, model_trainer=None):
        """
        Inizializza il motore di inferenza
        
        Args:
            model_trainer: Istanza di ModelTrainer con un modello caricato
        """
        self.model_trainer = model_trainer
        
        # Configurazione del logging
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/inference_engine.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('InferenceEngine')
    
    def prepare_inference_data(self, prices_df, symbol, date=None):
        """
        Prepara i dati per l'inferenza
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            symbol: Simbolo del titolo
            date: Data per cui fare la previsione (None = ultima disponibile)
            
        Returns:
            Sequenza normalizzata per l'inferenza
        """
        # Filtraggio dati
        df = prices_df[prices_df['symbol'] == symbol].copy()
        
        # Ordinamento per data
        df = df.sort_values('date')
        
        # Selezione della data per la previsione
        if date is not None:
            # Troviamo la posizione della data
            date_idx = df[df['date'] == date].index
            if len(date_idx) == 0:
                self.logger.error(f"Data non trovata: {date}")
                raise ValueError(f"Data non trovata: {date}")
            
            end_idx = date_idx[0]
        else:
            # Utilizziamo l'ultima data disponibile
            end_idx = df.index[-1]
        
        # Estrazione della sequenza di input
        sequence_length = self.model_trainer.data_preprocessor.sequence_length
        start_idx = max(0, end_idx - sequence_length)
        
        if end_idx - start_idx + 1 < sequence_length:
            self.logger.error(f"Dati insufficienti per la sequenza di lunghezza {sequence_length}")
            raise ValueError(f"Dati insufficienti per la sequenza di lunghezza {sequence_length}")
        
        # Estrazione dei dati
        prices = df.iloc[start_idx:end_idx+1][['open', 'high', 'low', 'close']].values
        volumes = df.iloc[start_idx:end_idx+1]['volume'].values.reshape(-1, 1)
        
        # Normalizzazione
        normalized_prices = self.model_trainer.data_preprocessor.price_scaler.transform(prices)
        normalized_volumes = self.model_trainer.data_preprocessor.volume_scaler.transform(volumes)
        
        # Combinazione di prezzi e volumi
        sequence = np.column_stack((normalized_prices, normalized_volumes))
        
        # Assicurarsi che la sequenza abbia la lunghezza corretta
        if len(sequence) > sequence_length:
            sequence = sequence[-sequence_length:]
        
        return sequence
    
    def predict_next_price(self, prices_df, symbol, date=None):
        """
        Predice il prezzo di chiusura del giorno successivo
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            symbol: Simbolo del titolo
            date: Data per cui fare la previsione (None = ultima disponibile)
            
        Returns:
            Previsione del prezzo di chiusura
        """
        try:
            # Preparazione dei dati
            sequence = self.prepare_inference_data(prices_df, symbol, date)
            
            # Previsione
            prediction = self.model_trainer.predict(sequence)
            
            self.logger.info(f"Previsione per {symbol}: {prediction:.2f}")
            
            return prediction
        
        except Exception as e:
            self.logger.error(f"Errore nella previsione: {e}")
            raise
    
    def predict_multiple_days(self, prices_df, symbol, start_date, num_days=5):
        """
        Predice i prezzi per multipli giorni consecutivi
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            symbol: Simbolo del titolo
            start_date: Data di inizio delle previsioni
            num_days: Numero di giorni da prevedere
            
        Returns:
            Lista di previsioni
        """
        self.logger.info(f"Previsione multipla per {symbol} da {start_date} per {num_days} giorni")
        
        df = prices_df[prices_df['symbol'] == symbol].copy()
        df = df.sort_values('date')
        
        # Troviamo la posizione della data di inizio
        start_idx = df[df['date'] == start_date].index
        if len(start_idx) == 0:
            self.logger.error(f"Data di inizio non trovata: {start_date}")
            raise ValueError(f"Data di inizio non trovata: {start_date}")
        
        start_idx = start_idx[0]
        
        predictions = []
        current_df = df.iloc[:start_idx+1].copy()
        
        for _ in range(num_days):
            try:
                # Preparazione dei dati
                sequence = self.prepare_inference_data(current_df, symbol)
                
                # Previsione
                prediction = self.model_trainer.predict(sequence)
                predictions.append(prediction)
                
                # Aggiungiamo la previsione al DataFrame per la previsione successiva
                last_date = current_df['date'].max()
                next_date = pd.to_datetime(last_date) + pd.DateOffset(1)
                
                new_row = pd.DataFrame({
                    'date': [next_date],
                    'symbol': [symbol],
                    'open': [prediction],
                    'high': [prediction],
                    'low': [prediction],
                    'close': [prediction],
                    'volume': [current_df['volume'].mean()]
                })
                
                current_df = pd.concat([current_df, new_row])
            
            except Exception as e:
                self.logger.error(f"Errore nella previsione multipla: {e}")
                break
        
        return predictions
    
    def evaluate_predictions(self, prices_df, symbol, test_dates):
        """
        Valuta le performance predittive del modello
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            symbol: Simbolo del titolo
            test_dates: Lista di date per il test
            
        Returns:
            Dizionario con le metriche di valutazione
        """
        self.logger.info(f"Valutazione delle previsioni per {symbol} su {len(test_dates)} date")
        
        df = prices_df[prices_df['symbol'] == symbol].copy()
        df = df.sort_values('date')
        
        predictions = []
        actual_values = []
        
        for date in test_dates:
            try:
                # Troviamo l'indice della data corrente
                date_idx = df[df['date'] == date].index
                if len(date_idx) == 0:
                    continue
                
                date_idx = date_idx[0]
                
                # Troviamo la data successiva
                next_dates = df[df['date'] > date]['date']
                if len(next_dates) == 0:
                    continue
                
                next_date = next_dates.min()
                next_idx = df[df['date'] == next_date].index[0]
                
                # Previsione
                prediction = self.predict_next_price(df.iloc[:date_idx+1], symbol, date)
                actual = df.loc[next_idx, 'close']
                
                predictions.append(prediction)
                actual_values.append(actual)
            
            except Exception as e:
                self.logger.warning(f"Errore nella valutazione per la data {date}: {e}")
        
        # Calcolo delle metriche
        if len(predictions) == 0:
            self.logger.error("Nessuna previsione valida")
            return None
        
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        self.logger.info(f"Metriche - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Grafico delle previsioni
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates[:len(actual_values)], actual_values, 'b-', label='Valori Reali')
        plt.plot(test_dates[:len(predictions)], predictions, 'r--', label='Previsioni')
        plt.xlabel('Data')
        plt.ylabel('Prezzo di Chiusura')
        plt.title(f'Previsioni vs Valori Reali per {symbol}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./reports/prediction_evaluation_{symbol}.png')
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actual_values': actual_values
        }
3.5 Trading Strategy Execution
3.5.1 Funzionalità

Implementazione di strategie basate su IA
Integrazione con i modelli di previsione
Esecuzione delle strategie in ambiente simulato
Backtesting e ottimizzazione delle performance

3.5.2 Implementazione
pythonCopia# trading_strategy/executor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import json

class AITradingStrategy:
    """Strategia di trading basata su reti neurali"""
    def __init__(self, inference_engine, threshold_buy=0.02, threshold_sell=-0.02, risk_level=0.5):
        """
        Inizializza la strategia di trading basata su IA
        
        Args:
            inference_engine: Motore di inferenza per le previsioni
            threshold_buy: Soglia per l'acquisto (% di aumento previsto)
            threshold_sell: Soglia per la vendita (% di diminuzione previsto)
            risk_level: Livello di rischio (0-1) che influenza la quantità da investire
        """
        self.inference_engine = inference_engine
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.risk_level = risk_level
        
        # Configurazione del logging
        self.logger = logging.getLogger('AITradingStrategy')
    
    def generate_actions(self, agent, prices, date):
        """
        Genera azioni di trading basate sulle previsioni del modello
        
        Args:
            agent: Agente di trading
            prices: Prezzi correnti del mercato
            date: Data corrente
            
        Returns:
            Lista di azioni di trading
        """
        actions = []
        
        try:
            for symbol, price_data in prices.items():
                current_price = price_data['close']
                
                # Preparazione dataframe per la previsione
                df = pd.DataFrame({
                    'date': [date],
                    'symbol': [symbol],
                    'open': [price_data['open']],
                    'high': [price_data['high']],
                    'low': [price_data['low']],
                    'close': [current_price],
                    'volume': [price_data['volume']]
                })
                
                # Previsione del prezzo successivo
                predicted_price = self.inference_engine.predict_next_price(df, symbol)
                
                # Calcolo della variazione percentuale prevista
                price_change_pct = (predicted_price - current_price) / current_price
                
                self.logger.debug(f"Simbolo: {symbol}, Prezzo: {current_price:.2f}, Previsione: {predicted_price:.2f}, Variazione: {price_change_pct:.2%}")
                
                # Decisione di trading
                if price_change_pct > self.threshold_buy:
                    # Acquisto
                    invest_amount = agent.cash * self.risk_level * min(1, price_change_pct * 10)
                    quantity = int(invest_amount / current_price) if current_price > 0 else 0
                    
                    if quantity > 0:
                        actions.append({
                            'type': 'buy',
                            'symbol': symbol,
                            'quantity': quantity
                        })
                elif price_change_pct < self.threshold_sell:
                    # Vendita
                    if symbol in agent.portfolio and agent.portfolio[symbol] > 0:
                        sell_ratio = min(1, abs(price_change_pct) * 10)
                        quantity = int(agent.portfolio[symbol] * sell_ratio)
                        
                        if quantity > 0:
                            actions.append({
                                'type': 'sell',
                                'symbol': symbol,
                                'quantity': quantity
                            })
        
        except Exception as e:
            self.logger.error(f"Errore nella generazione delle azioni: {e}")
        
        return actions


class BacktestEngine:
    """Motore di backtesting per le strategie di trading"""
    def __init__(self, strategy, initial_capital=100000):
        """
        Inizializza il motore di backtesting
        
        Args:
            strategy: Strategia di trading da testare
            initial_capital: Capitale iniziale per il backtesting
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        
        # Configurazione del logging
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./reports', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/backtest_engine.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BacktestEngine')
    
    def run_backtest(self, prices_df, start_date, end_date):
        """
        Esegue un backtest della strategia
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            start_date: Data di inizio del backtest
            end_date: Data di fine del backtest
            
        Returns:
            Risultati del backtest
        """
        self.logger.info(f"Avvio backtest da {start_date} a {end_date}")
        
        # Filtriamo il dataframe per il periodo di backtest
        df = prices_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Ordinamento per data
        df = df.sort_values('date')
        
        # Lista di date uniche nel periodo
        dates = df['date'].unique()
        
        # Inizializzazione del portafoglio

4. Gestione dello Stato e Integrazione in Tempo Reale
4.1 Sistema di Gestione dello Stato

4.1.1 Componenti
- Gestore dello stato globale dell'applicazione
- Sistema di persistenza delle scelte utente
- Cache per i dati frequentemente utilizzati
- Sistema di eventi per la comunicazione tra moduli

4.1.2 Implementazione
```python
# dashboard/state_manager.py
from typing import Dict, Any
import json
import os

class DashboardStateManager:
    def __init__(self, state_file='dashboard_state.json'):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
    
    def get_tab_state(self, tab_name: str) -> Dict[str, Any]:
        return self.state.get(tab_name, {})
    
    def update_tab_state(self, tab_name: str, new_state: Dict[str, Any]):
        self.state[tab_name] = new_state
        self.save_state()
```

4.2 Sistema di Eventi in Tempo Reale

4.2.1 Componenti
- WebSocket server per la comunicazione in tempo reale
- Sistema di eventi per la propagazione degli aggiornamenti
- Gestione delle sottoscrizioni per i vari moduli

4.2.2 Implementazione
```python
# dashboard/websocket_manager.py
from flask_socketio import SocketIO, emit
from typing import Dict, Any

class WebSocketManager:
    def __init__(self, app):
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self.subscribers: Dict[str, list] = {}
    
    def subscribe(self, event: str, callback):
        if event not in self.subscribers:
            self.subscribers[event] = []
        self.subscribers[event].append(callback)
    
    def emit_update(self, event: str, data: Any):
        self.socketio.emit(event, data)
        if event in self.subscribers:
            for callback in self.subscribers[event]:
                callback(data)
```

4.3 Integrazione dei Moduli

4.3.1 Data Collection Module
- Visualizzazione in tempo reale dei dati scaricati
- Grafici interattivi per l'analisi dei dati
- Sistema di notifiche per il completamento dei download

4.3.2 Market Simulation Module
- Grafici candlestick interattivi
- Visualizzazione del bookkeeping in tempo reale
- Monitoraggio delle performance degli agenti

4.3.3 Neural Network Module
- Grafici delle performance di training
- Visualizzazione delle previsioni vs realtà
- Monitoraggio dei KPI in tempo reale

4.3.4 Self-Play Module
- Integrazione con il modulo di mercato
- Visualizzazione delle performance dell'agente
- Sistema di controllo e interazione con l'agente