import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_simulator import MarketEnvironment, SimulationManager
from trading_strategy.strategies import (
    RandomStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    ValueInvestingStrategy,
    NeuralNetworkStrategy
)
import os

class TestStrategyIntegration(unittest.TestCase):
    def setUp(self):
        """Prepara l'ambiente di test"""
        # Crea dati di test
        self.start_date = pd.Timestamp('2023-01-01')
        self.end_date = pd.Timestamp('2023-12-31')
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        
        # Genera dati simulati per AAPL e GOOGL
        self.test_data = {}
        for symbol in ['AAPL', 'GOOGL']:
            # Genera prezzi con trend e volatilità
            base_price = 100 if symbol == 'AAPL' else 2000
            trend = np.linspace(0, 50, len(dates))  # Trend lineare
            volatility = np.random.normal(0, 5, len(dates))  # Volatilità casuale
            prices = base_price + trend + volatility
            
            # Assicura che i prezzi siano positivi
            prices = np.maximum(prices, 1)
            
            # Crea il DataFrame
            self.test_data[symbol] = pd.DataFrame({
                'Open': prices * (1 + np.random.uniform(-0.02, 0.02, len(dates))),
                'High': prices * (1 + np.random.uniform(0, 0.04, len(dates))),
                'Low': prices * (1 - np.random.uniform(0, 0.04, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
        
        # Salva i dati in file CSV
        os.makedirs('data', exist_ok=True)
        for symbol, df in self.test_data.items():
            df.to_csv(f'data/{symbol}.csv')
        
        self.config = {
            'market': {
                'symbols': ['AAPL', 'GOOGL'],
                'start_date': self.start_date,
                'end_date': self.end_date
            },
            'strategies': {
                'active_strategy': 'random',
                'strategy_params': {}
            },
            'trading': {
                'initial_capital': 100000,
                'transaction_fee': 0.001
            }
        }

    def test_random_strategy_integration(self):
        """Testa l'integrazione della strategia casuale"""
        sim_manager = SimulationManager(self.config)
        sim_manager.initialize_simulation()
        sim_manager.create_agents(num_agents=1)
        
        # Esegui la simulazione
        results = sim_manager.run_simulation()
        
        # Verifica i risultati
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], dict)
        self.assertIn('action', results[0])
        self.assertIn('price', results[0])
    
    def test_mean_reversion_strategy_integration(self):
        """Testa l'integrazione della strategia di mean reversion"""
        self.config['strategies'] = {
            'active_strategy': 'mean_reversion',
            'strategy_params': {
                'mean_reversion': {
                    'window': 20,
                    'threshold': 2.0
                }
            }
        }
        
        sim_manager = SimulationManager(self.config)
        sim_manager.initialize_simulation()
        sim_manager.create_agents(num_agents=1)
        
        results = sim_manager.run_simulation()
        
        # Verifica i risultati
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # Verifica che le transazioni seguano la logica di mean reversion
        for result in results:
            self.assertIn('action', result)
            self.assertIn('price', result)
            self.assertIn('signal', result)
    
    def test_trend_following_strategy_integration(self):
        """Testa l'integrazione della strategia di trend following"""
        self.config['strategies'] = {
            'active_strategy': 'trend_following',
            'strategy_params': {
                'trend_following': {
                    'short_window': 10,
                    'long_window': 50
                }
            }
        }
        
        sim_manager = SimulationManager(self.config)
        sim_manager.initialize_simulation()
        sim_manager.create_agents(num_agents=1)
        
        results = sim_manager.run_simulation()
        
        # Verifica i risultati
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # Verifica che le transazioni seguano la logica di trend following
        for result in results:
            self.assertIn('action', result)
            self.assertIn('price', result)
            self.assertIn('trend', result)

class TestPerformance(unittest.TestCase):
    def setUp(self):
        """Prepara l'ambiente di test"""
        # Crea dati di test
        self.start_date = pd.Timestamp('2023-01-01')
        self.end_date = pd.Timestamp('2023-12-31')
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        
        # Genera dati simulati per AAPL e GOOGL
        self.test_data = {}
        for symbol in ['AAPL', 'GOOGL']:
            # Genera prezzi con trend e volatilità
            base_price = 100 if symbol == 'AAPL' else 2000
            trend = np.linspace(0, 50, len(dates))  # Trend lineare
            volatility = np.random.normal(0, 5, len(dates))  # Volatilità casuale
            prices = base_price + trend + volatility
            
            # Assicura che i prezzi siano positivi
            prices = np.maximum(prices, 1)
            
            # Crea il DataFrame
            self.test_data[symbol] = pd.DataFrame({
                'Open': prices * (1 + np.random.uniform(-0.02, 0.02, len(dates))),
                'High': prices * (1 + np.random.uniform(0, 0.04, len(dates))),
                'Low': prices * (1 - np.random.uniform(0, 0.04, len(dates))),
                'Close': prices,
                'Volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
        
        # Salva i dati in file CSV
        os.makedirs('data', exist_ok=True)
        for symbol, df in self.test_data.items():
            df.to_csv(f'data/{symbol}.csv')
        
        self.config = {
            'market': {
                'symbols': ['AAPL', 'GOOGL'],
                'start_date': self.start_date,
                'end_date': self.end_date
            },
            'strategies': {
                'active_strategy': 'random',
                'strategy_params': {}
            },
            'trading': {
                'initial_capital': 100000,
                'transaction_fee': 0.001
            }
        }

    def test_memory_usage(self):
        """Testa l'uso della memoria durante la simulazione"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Esegui una simulazione lunga
        sim_manager = SimulationManager(self.config)
        sim_manager.initialize_simulation()
        sim_manager.create_agents(num_agents=5)
        results = sim_manager.run_simulation()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verifica che l'aumento di memoria sia ragionevole (meno di 1GB)
        self.assertLess(memory_increase, 1024 * 1024 * 1024)
    
    def test_execution_time(self):
        """Testa il tempo di esecuzione della simulazione"""
        import time
        
        sim_manager = SimulationManager(self.config)
        sim_manager.initialize_simulation()
        sim_manager.create_agents(num_agents=5)
        
        start_time = time.time()
        results = sim_manager.run_simulation()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verifica che il tempo di esecuzione sia ragionevole (meno di 5 minuti)
        self.assertLess(execution_time, 300)
    
    def test_cache_performance(self):
        """Testa le performance del sistema di caching"""
        import time
        
        sim_manager = SimulationManager(self.config)
        sim_manager.initialize_simulation()
        
        # Prima chiamata (senza cache)
        start_time = time.time()
        data1 = sim_manager.market_env.get_market_data('2023-01-01')
        first_call_time = time.time() - start_time
        
        # Seconda chiamata (con cache)
        start_time = time.time()
        data2 = sim_manager.market_env.get_market_data('2023-01-01')
        second_call_time = time.time() - start_time
        
        # Verifica che la seconda chiamata sia più veloce
        self.assertLess(second_call_time, first_call_time)

if __name__ == '__main__':
    unittest.main() 