#!/usr/bin/env python3
"""
Main Script per il sistema di simulazione del mercato.

Questo script avvia il sistema di simulazione del mercato azionario.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/main_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Main')

# Aggiungi le directory necessarie al percorso di importazione
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, update_config
from market_simulator.simulation import SimulationManager
from interface.wizard import run_wizard
from data.collector import DataCollector
from neural_network.model_trainer import ModelTrainer

def main():
    """Funzione principale del programma."""
    # Crea il parser degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Sistema di simulazione del mercato azionario')
    parser.add_argument('--mode', choices=['simulation', 'training', 'self_play', 'data_collection', 'wizard'], 
                        default='simulation', help='Modalità di esecuzione')
    parser.add_argument('--num_agents', type=int, default=5, help='Numero di agenti (per modalità simulation)')
    parser.add_argument('--strategy', choices=CONFIG['strategies']['available_strategies'], 
                        default='random', help='Strategia di trading (per modalità simulation)')
    parser.add_argument('--symbols', nargs='+', default=None, 
                        help='Simboli di trading (lascia vuoto per usare quelli in configurazione)')
    parser.add_argument('--start_date', default=None, 
                        help='Data di inizio in formato YYYY-MM-DD (lascia vuoto per usare quella in configurazione)')
    parser.add_argument('--end_date', default=None, 
                        help='Data di fine in formato YYYY-MM-DD (lascia vuoto per usare quella in configurazione)')
    parser.add_argument('--initial_capital', type=float, default=None, 
                        help='Capitale iniziale (lascia vuoto per usare quello in configurazione)')
    parser.add_argument('--save_report', action='store_true', help='Salva il report della simulazione')
    
    # Analizza gli argomenti da linea di comando
    args = parser.parse_args()
    
    # Crea le directory di base se non esistono
    for dir_path in ['data', 'logs', 'reports', 'models']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Aggiorna la configurazione in base agli argomenti
    config = CONFIG.copy()
    
    if args.symbols:
        config['market']['symbols'] = args.symbols
    if args.start_date:
        config['market']['start_date'] = args.start_date
    if args.end_date:
        config['market']['end_date'] = args.end_date
    if args.initial_capital:
        config['trading']['initial_capital'] = args.initial_capital
    if args.strategy:
        config['strategies']['active_strategy'] = args.strategy
    
    # Esecuzione in base alla modalità
    if args.mode == 'wizard':
        # Modalità wizard: avvia la configurazione guidata
        run_wizard()
    elif args.mode == 'data_collection':
        # Modalità raccolta dati: scarica i dati storici
        data_collection(config)
    elif args.mode == 'simulation':
        # Modalità simulazione: esegui la simulazione
        run_simulation(config, args.num_agents, args.save_report)
    elif args.mode == 'training':
        # Modalità addestramento: addestra il modello di rete neurale
        run_training(config)
    elif args.mode == 'self_play':
        # Modalità self-play: esegui il self-play per la rete neurale
        run_self_play(config)
    else:
        logger.error(f"Modalità non valida: {args.mode}")
        return False
    
    return True

def data_collection(config):
    """
    Esegue la raccolta dei dati.
    
    Args:
        config: Configurazione del sistema
    """
    logger.info("Avvio raccolta dati...")
    
    # Crea il collector di dati
    collector = DataCollector()
    
    symbols = config['market']['symbols']
    start_date = config['market']['start_date']
    end_date = config['market']['end_date']
    
    # Scarica i dati per ogni simbolo
    for symbol in symbols:
        logger.info(f"Scaricamento dati per {symbol}...")
        try:
            data = collector.get_stock_data(symbol, start_date, end_date, force_download=True)
            if data is not None:
                logger.info(f"Dati per {symbol} scaricati con successo: {len(data)} righe")
            else:
                logger.error(f"Errore nel download dei dati per {symbol}")
        except Exception as e:
            logger.error(f"Errore nel download dei dati per {symbol}: {e}")
    
    logger.info("Raccolta dati completata")

def run_simulation(config, num_agents, save_report=True):
    """
    Esegue la simulazione.
    
    Args:
        config: Configurazione del sistema
        num_agents: Numero di agenti da creare
        save_report: Salva il report della simulazione
    """
    logger.info("Avvio simulazione...")
    
    # Assicurati che i dati necessari siano disponibili
    symbols = config['market']['symbols']
    start_date = config['market']['start_date']
    end_date = config['market']['end_date']
    
    # Verifica la disponibilità dei dati
    collector = DataCollector()
    for symbol in symbols:
        if not collector.is_data_available(symbol, start_date, end_date):
            logger.warning(f"Dati mancanti per {symbol}. Avvio download...")
            collector.get_stock_data(symbol, start_date, end_date)
    
    # Crea il simulation manager
    sim_manager = SimulationManager(config)
    
    # Inizializza la simulazione
    if not sim_manager.initialize_simulation():
        logger.error("Errore nell'inizializzazione della simulazione")
        return False
    
    # Crea gli agenti
    if not sim_manager.create_agents(num_agents=num_agents):
        logger.error("Errore nella creazione degli agenti")
        return False
    
    # Esegui la simulazione
    logger.info("Esecuzione simulazione...")
    results = sim_manager.run_simulation()
    
    if results is None:
        logger.error("Errore nell'esecuzione della simulazione")
        return False
    
    # Ottieni il riepilogo delle transazioni
    summary = sim_manager.get_transactions_summary()
    if summary:
        logger.info("\nRiepilogo delle transazioni:")
        logger.info(f"- Transazioni totali: {summary['total_transactions']}")
        logger.info(f"- Acquisti: {summary['buy_transactions']}")
        logger.info(f"- Vendite: {summary['sell_transactions']}")
        logger.info(f"- Valore totale acquisti: {summary['total_buy_value']:.2f} USD")
        logger.info(f"- Valore totale vendite: {summary['total_sell_value']:.2f} USD")
        logger.info(f"- Valore netto: {summary['net_value']:.2f} USD")
    
    # Ottieni le performance degli agenti
    performances = sim_manager.get_agents_performance()
    if performances:
        logger.info("\nPerformance degli agenti:")
        for perf in performances:
            logger.info(f"- Agente {perf['id']}: Rendimento {perf['percentage_return']:.2f}%")
    
    # Salva i risultati
    if save_report:
        sim_manager.save_results()
    
    logger.info("Simulazione completata con successo")
    return True

def run_training(config):
    """
    Addestra un modello di rete neurale.
    
    Args:
        config: Configurazione del sistema
    """
    logger.info("Avvio addestramento modello...")
    
    # Crea il trainer
    trainer = ModelTrainer(config)
    
    # Carica i dati
    if not trainer.load_data():
        logger.error("Errore nel caricamento dei dati")
        return False
    
    # Prepara i dati
    if not trainer.prepare_data():
        logger.error("Errore nella preparazione dei dati")
        return False
    
    # Crea il modello
    if not trainer.create_model():
        logger.error("Errore nella creazione del modello")
        return False
    
    # Addestra il modello
    if not trainer.train_model():
        logger.error("Errore nell'addestramento del modello")
        return False
    
    # Valuta il modello
    metrics = trainer.evaluate_model()
    logger.info(f"Metriche di valutazione: {metrics}")
    
    # Salva il modello
    if not trainer.save_model():
        logger.error("Errore nel salvataggio del modello")
        return False
    
    logger.info("Addestramento completato con successo")
    return True

def run_self_play(config):
    """
    Esegue il self-play per la rete neurale.
    
    Args:
        config: Configurazione del sistema
    """
    logger.info("Avvio self-play...")
    
    # Verifica che il self-play sia abilitato
    if not config.get('self_play', {}).get('enabled', False):
        logger.warning("Self-play non abilitato nella configurazione")
        logger.info("Abilita il self-play nella configurazione con: python -m interface.wizard")
        return False
    
    # Crea il trainer
    trainer = ModelTrainer(config)
    
    # Esegui il self-play
    generations = config.get('self_play', {}).get('generations', 10)
    population_size = config.get('self_play', {}).get('population_size', 20)
    
    logger.info(f"Avvio self-play con {population_size} modelli per {generations} generazioni")
    
    for generation in range(generations):
        logger.info(f"Generazione {generation+1}/{generations}")
        
        # Evolvi la popolazione
        trainer.evolve_population(population_size)
        
        # Valuta i modelli
        fitness_scores = trainer.evaluate_population()
        
        # Seleziona i migliori
        best_models = trainer.select_best_models(fitness_scores)
        
        logger.info(f"Migliore rendimento nella generazione {generation+1}: {max(fitness_scores):.2f}%")
    
    # Salva il miglior modello dell'ultima generazione
    trainer.save_best_model()
    
    logger.info("Self-play completato con successo")
    return True

if __name__ == '__main__':
    main() 