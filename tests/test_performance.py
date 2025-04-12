import unittest
import numpy as np
import pandas as pd
import time
import psutil
import os
from market_simulator import MarketEnvironment, PerformanceMonitor
from config.monitoring_config import (
    REDIS_CONFIG,
    PROMETHEUS_CONFIG,
    METRICS_CONFIG,
    PERFORMANCE_LOGGING,
    ALERTING_CONFIG,
    SAMPLING_CONFIG
)

class TestPerformanceMonitor(unittest.TestCase):
    def setUp(self):
        """Prepara l'ambiente di test"""
        self.monitor = PerformanceMonitor()
    
    def test_operation_recording(self):
        """Testa la registrazione delle operazioni"""
        # Simula alcune operazioni
        operations = [
            ('cache_hit', 0.1),
            ('cache_miss', 0.5),
            ('trade_execution', 1.0)
        ]
        
        for op_name, duration in operations:
            self.monitor.record_operation(op_name, duration)
        
        # Verifica le metriche
        report = self.monitor.get_performance_report()
        
        for op_name, duration in operations:
            self.assertIn(op_name, report['operations'])
            self.assertEqual(len(report['operations'][op_name]['times']), 1)
            self.assertEqual(report['operations'][op_name]['avg_time'], duration)
    
    def test_memory_monitoring(self):
        """Testa il monitoraggio della memoria"""
        # Registra uso memoria
        self.monitor.record_memory_usage()
        
        # Verifica le metriche
        report = self.monitor.get_performance_report()
        
        self.assertGreater(report['memory']['current'], 0)
    
    def test_cpu_monitoring(self):
        """Testa il monitoraggio della CPU"""
        # Registra uso CPU
        self.monitor.record_cpu_usage()
        
        # Verifica le metriche
        report = self.monitor.get_performance_report()
        
        self.assertGreaterEqual(report['cpu']['current'], 0)
        self.assertLessEqual(report['cpu']['current'], 100)
        self.assertGreater(report['cpu']['max'], 0)
        self.assertGreater(report['cpu']['avg'], 0)
    
    def test_uptime_tracking(self):
        """Testa il tracciamento dell'uptime"""
        # Attendi un breve periodo
        time.sleep(0.1)
        
        # Verifica l'uptime
        report = self.monitor.get_performance_report()
        self.assertGreater(report['uptime'], 0.1)

class TestMarketEnvironmentPerformance(unittest.TestCase):
    def setUp(self):
        """Prepara l'ambiente di test"""
        # Crea dati di test
        self.start_date = pd.Timestamp('2023-01-01')
        self.end_date = pd.Timestamp('2023-01-03')
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        
        self.test_data = {
            'AAPL': pd.DataFrame({
                'Open': [100.0, 101.0, 102.0],
                'High': [101.0, 102.0, 103.0],
                'Low': [99.0, 100.0, 101.0],
                'Close': [100.5, 101.5, 102.5],
                'Volume': [1000, 1100, 1200]
            }, index=dates)
        }
        
        self.market_env = MarketEnvironment(
            data=self.test_data,
            start_date=self.start_date,
            end_date=self.end_date
        )
    
    def test_market_data_performance(self):
        """Testa le performance del recupero dati di mercato"""
        # Prima chiamata (senza cache)
        start_time = time.time()
        data1 = self.market_env.get_market_data('2023-01-01')
        first_call_time = time.time() - start_time
        
        # Seconda chiamata (con cache)
        start_time = time.time()
        data2 = self.market_env.get_market_data('2023-01-01')
        second_call_time = time.time() - start_time
        
        # Verifica che la seconda chiamata sia più veloce
        self.assertLess(second_call_time, first_call_time)
        
        # Verifica le metriche
        metrics = self.market_env.get_performance_metrics()
        self.assertIn('cache_hit', metrics['operations'])
        self.assertIn('local_cache_hit', metrics['operations'])
    
    def test_trade_execution_performance(self):
        """Testa le performance dell'esecuzione dei trade"""
        # Esegui alcuni trade
        trades = []
        for _ in range(5):
            trade = self.market_env.execute_transaction(
                agent_id='test_agent',
                symbol='AAPL',
                action='buy',
                quantity=10,
                price=100.0
            )
            trades.append(trade)
        
        # Verifica le metriche
        metrics = self.market_env.get_performance_metrics()
        self.assertIn('trade_execution', metrics['operations'])
        self.assertEqual(metrics['operations']['trade_execution']['count'], 5)
    
    def test_memory_optimization(self):
        """Testa l'ottimizzazione della memoria"""
        # Verifica l'uso della memoria prima
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Esegui operazioni intensive
        for _ in range(100):
            self.market_env.get_market_data('2023-01-01')
        
        # Verifica l'uso della memoria dopo
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verifica che l'aumento di memoria sia ragionevole
        self.assertLess(memory_increase, 50 * 1024 * 1024)  # Meno di 50MB

if __name__ == '__main__':
    unittest.main() 