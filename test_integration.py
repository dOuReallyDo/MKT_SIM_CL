#!/usr/bin/env python3
"""
Test di integrazione per il progetto MKT_SIM_CL.

Questo script verifica che tutti i componenti principali del sistema
funzionino correttamente insieme.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestIntegration')

# Importa i moduli del progetto
from data.collector import DataCollector
from market_simulator.environment import MarketEnvironment
from market_simulator.agents import TradingAgent
from market_simulator.simulation import SimulationManager
from trading_strategy import create_strategy, get_available_strategies

def setup_test_config():
    """
    Configura un ambiente di test minimale
    
    Returns:
        dict: Configurazione di test
    """
    # Date recenti per avere dati disponibili
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Ultimi 60 giorni
    
    return {
        'market': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],  # Simboli comuni e liquidi
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        },
        'trading': {
            'initial_capital': 10000.0,
            'strategy': 'random'
        },
        'strategies': {
            'active_strategy': 'random',
            'available_strategies': list(get_available_strategies().keys()),
            'strategy_params': {
                'mean_reversion': {
                    'window': 20
                },
                'trend_following': {
                    'short_window': 10,
                    'long_window': 50
                }
            }
        }
    }

def test_data_collector():
    """Testa il modulo di raccolta dati"""
    logger.info("Test del DataCollector...")
    
    # Crea una directory temporanea per il test
    test_data_dir = os.path.join('tests', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Crea il collector
    collector = DataCollector(data_dir=test_data_dir)
    
    # Ottieni i dati per un simbolo recente
    symbol = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = collector.get_stock_data(symbol, start_date, end_date, force_download=True)
    
    # Verifica che i dati siano stati scaricati
    if df is None or df.empty:
        logger.error(f"Errore: nessun dato ottenuto per {symbol}")
        return False
    
    logger.info(f"Dati scaricati per {symbol}: {len(df)} righe")
    
    # Verifica integrità dei dati
    integrity_report = collector.verify_data_integrity()
    logger.info(f"Report integrità dati: {integrity_report['valid_files']}/{integrity_report['total_files']} file validi")
    
    return True

def test_market_environment():
    """Testa l'ambiente di mercato"""
    logger.info("Test del MarketEnvironment...")
    
    # Crea un collector
    collector = DataCollector()
    
    # Ottieni alcuni dati recenti
    symbols = ['AAPL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Carica i dati
    market_data = {}
    for symbol in symbols:
        df = collector.get_stock_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            market_data[symbol] = df
    
    if not market_data:
        logger.error("Errore: nessun dato ottenuto per i test")
        return False
    
    # Crea l'ambiente di mercato
    market_env = MarketEnvironment(
        data=market_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verifica che le date di trading siano state inizializzate
    if not market_env.trading_days:
        logger.error("Errore: nessuna data di trading inizializzata")
        return False
    
    logger.info(f"Date di trading inizializzate: {len(market_env.trading_days)} giorni")
    
    # Ottieni i dati di mercato per la prima data
    first_date = market_env.trading_days[0]
    market_data_first_day = market_env.get_market_data(first_date)
    
    if not market_data_first_day:
        logger.error(f"Errore: nessun dato di mercato per {first_date}")
        return False
    
    logger.info(f"Dati di mercato per {first_date} ottenuti: {list(market_data_first_day.keys())}")
    
    return True

def test_trading_strategies():
    """Testa le strategie di trading"""
    logger.info("Test delle strategie di trading...")
    
    # Testa la creazione di strategie
    for strategy_name in get_available_strategies().keys():
        try:
            strategy = create_strategy(strategy_name)
            logger.info(f"Strategia {strategy_name} creata: {strategy.__class__.__name__}")
        except Exception as e:
            logger.error(f"Errore nella creazione della strategia {strategy_name}: {e}")
            return False
    
    # Testa la generazione di segnali
    random_strategy = create_strategy('random')
    market_data = {
        'AAPL': {
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
    }
    
    signal = random_strategy.generate_signal(market_data)
    logger.info(f"Segnale generato: {signal}")
    
    return True

def test_trading_agent():
    """Testa l'agente di trading"""
    logger.info("Test dell'agente di trading...")
    
    # Crea una strategia
    strategy = create_strategy('random')
    
    # Crea l'agente
    agent = TradingAgent(id=1, initial_capital=10000, strategy=strategy)
    
    # Dati di mercato di test
    market_data = {
        'AAPL': {
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
    }
    
    # Testa l'esecuzione di un'operazione di acquisto
    success = agent.execute_buy('AAPL', 10, 153.0)
    if not success:
        logger.error("Errore nell'esecuzione dell'acquisto")
        return False
    
    logger.info(f"Acquisto eseguito, nuovo cash: {agent.cash}, portafoglio: {agent.portfolio}")
    
    # Testa l'esecuzione di un'operazione di vendita
    success = agent.execute_sell('AAPL', 5, 155.0)
    if not success:
        logger.error("Errore nell'esecuzione della vendita")
        return False
    
    logger.info(f"Vendita eseguita, nuovo cash: {agent.cash}, portafoglio: {agent.portfolio}")
    
    # Testa il calcolo delle performance
    performance = agent.get_performance_metrics(market_data)
    logger.info(f"Performance: {performance}")
    
    return True

def test_simulation_manager():
    """Testa il gestore di simulazione"""
    logger.info("Test del SimulationManager...")
    
    # Crea una configurazione di test
    config = setup_test_config()
    
    # Crea il gestore di simulazione
    sim_manager = SimulationManager(config)
    
    # Inizializza la simulazione
    if not sim_manager.initialize_simulation():
        logger.error("Errore nell'inizializzazione della simulazione")
        return False
    
    logger.info("Simulazione inizializzata")
    
    # Crea gli agenti
    if not sim_manager.create_agents(num_agents=3):
        logger.error("Errore nella creazione degli agenti")
        return False
    
    logger.info("Agenti creati")
    
    # Esegui la simulazione
    transactions = sim_manager.run_simulation()
    if transactions is None:
        logger.error("Errore nell'esecuzione della simulazione")
        return False
    
    logger.info(f"Simulazione completata: {len(transactions)} transazioni")
    
    # Ottieni il riepilogo delle transazioni
    summary = sim_manager.get_transactions_summary()
    logger.info(f"Riepilogo transazioni: {summary}")
    
    # Ottieni le performance degli agenti
    performances = sim_manager.get_agents_performance()
    logger.info(f"Performance agenti: {performances}")
    
    # Salva i risultati
    if not sim_manager.save_results():
        logger.error("Errore nel salvataggio dei risultati")
        return False
    
    logger.info("Risultati salvati")
    
    return True

def run_all_tests():
    """Esegue tutti i test di integrazione"""
    logger.info("Avvio dei test di integrazione...")
    
    tests = [
        test_data_collector,
        test_market_environment,
        test_trading_strategies,
        test_trading_agent,
        test_simulation_manager
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            logger.info(f"Test {test_func.__name__}: {'SUCCESSO' if result else 'FALLITO'}")
        except Exception as e:
            logger.error(f"Errore nel test {test_func.__name__}: {e}")
            results.append(False)
    
    success_count = sum(1 for r in results if r)
    logger.info(f"Test completati: {success_count}/{len(tests)} successi")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 