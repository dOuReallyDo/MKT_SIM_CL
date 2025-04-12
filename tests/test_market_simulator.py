import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_simulator import MarketEnvironment, TradingAgent, SimulationManager
from trading_strategy.strategies import RandomStrategy

class TestMarketEnvironment(unittest.TestCase):
    def setUp(self):
        """Prepara i dati di test"""
        # Crea dati di test
        self.start_date = pd.Timestamp('2023-01-01')
        self.end_date = pd.Timestamp('2023-01-10')
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        self.test_data = {
            'AAPL': pd.DataFrame({
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(100, 200, len(dates)),
                'Low': np.random.uniform(100, 200, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        }
        self.market_env = MarketEnvironment(
            data=self.test_data,
            start_date=self.start_date,
            end_date=self.end_date
        )
    
    def test_initialization(self):
        """Testa l'inizializzazione dell'ambiente di mercato"""
        self.assertIsNotNone(self.market_env)
        self.assertEqual(len(self.market_env.stocks_data), 1)
        self.assertEqual(len(self.market_env.trading_days), 7)  # 5 giorni lavorativi
    
    def test_get_current_price(self):
        """Testa il recupero del prezzo corrente"""
        price = self.market_env.get_current_price('AAPL')
        self.assertIsNotNone(price)
        self.assertIsInstance(price, float)
    
    def test_execute_transaction(self):
        """Testa l'esecuzione di una transazione"""
        agent = TradingAgent(id=1, initial_capital=10000, strategy=RandomStrategy())
        transaction = self.market_env.execute_transaction(
            agent_id=agent.id,
            symbol='AAPL',
            action='buy',
            quantity=1,
            price=150.0
        )
        self.assertIsNotNone(transaction)
        self.assertEqual(transaction['symbol'], 'AAPL')
        self.assertEqual(transaction['action'], 'buy')
        self.assertEqual(transaction['price'], 150.0)

class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        """Prepara l'agente di test"""
        self.agent = TradingAgent(
            id=1,
            initial_capital=10000,
            strategy=RandomStrategy()
        )
    
    def test_initialization(self):
        """Testa l'inizializzazione dell'agente"""
        self.assertEqual(self.agent.id, 1)
        self.assertEqual(self.agent.initial_capital, 10000)
        self.assertEqual(self.agent.cash, 10000)
        self.assertEqual(len(self.agent.portfolio), 0)
    
    def test_portfolio_value(self):
        """Testa il calcolo del valore del portafoglio"""
        market_data = {
            'AAPL': {'close': 150.0},
            'MSFT': {'close': 200.0}
        }
        self.agent.portfolio = {'AAPL': 2, 'MSFT': 1}
        value = self.agent.get_portfolio_value(market_data)
        self.assertEqual(value, 10000 + (150.0 * 2) + (200.0 * 1))

class TestSimulationManager(unittest.TestCase):
    def setUp(self):
        """Prepara il gestore di simulazione"""
        self.config = {
            'market': {
                'symbols': ['AAPL'],
                'start_date': '2023-01-01',
                'end_date': '2023-01-10',
                'timeframes': ['1d'],
                'default_timeframe': '1d'
            },
            'trading': {
                'initial_capital': 10000,
                'order_types': ['market'],
                'default_order_type': 'market',
                'position_sizing': {
                    'default_quantity': 10,
                    'max_position_size': 0.2
                },
                'risk_management': {
                    'use_stop_loss': True,
                    'stop_loss_percentage': 2.0,
                    'use_take_profit': True,
                    'take_profit_percentage': 5.0,
                    'max_daily_loss': 1000
                }
            },
            'strategies': {
                'active_strategy': 'random',
                'available_strategies': ['random'],
                'strategy_params': {}
            }
        }
        self.sim_manager = SimulationManager(self.config)
    
    def test_initialization(self):
        """Testa l'inizializzazione del gestore di simulazione"""
        self.assertIsNotNone(self.sim_manager)
        self.assertEqual(self.sim_manager.config, self.config)
    
    def test_create_agents(self):
        """Testa la creazione degli agenti"""
        success = self.sim_manager.create_agents(num_agents=3)
        self.assertTrue(success)
        self.assertEqual(len(self.sim_manager.agents), 3)

if __name__ == '__main__':
    unittest.main() 