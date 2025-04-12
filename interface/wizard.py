#!/usr/bin/env python3
"""
Wizard Module.

Questo modulo fornisce un'interfaccia guidata per gli utenti non tecnici.
Permette di configurare il sistema di trading algoritmico passo-passo.
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta

# Aggiungiamo il percorso principale per poter importare altri moduli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import update_config, get_config, CONFIG
from market_simulator.simulation import SimulationManager
from data.collector import DataCollector

# Configurazione del logger
logger = logging.getLogger('Wizard')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class SetupWizard:
    """Wizard per la configurazione guidata del sistema."""
    
    def __init__(self):
        """Inizializza il wizard."""
        self.config = get_config()
        self.collector = DataCollector()
        
        # Assicurati che le directory necessarie esistano
        for dir_path in ['data', 'logs', 'reports', 'models']:
            os.makedirs(dir_path, exist_ok=True)
    
    def run(self):
        """Esegue il wizard passo-passo."""
        print("=" * 80)
        print("  WIZARD DI CONFIGURAZIONE DEL SISTEMA DI TRADING ALGORITMICO")
        print("=" * 80)
        print("\nBenvenuto nel wizard di configurazione del sistema di trading algoritmico.")
        print("Questo wizard ti guiderà nella configurazione del sistema passo dopo passo.\n")
        
        # Step 1: Configurazione dei simboli di trading
        self._step_configure_symbols()
        
        # Step 2: Configurazione del periodo di tempo
        self._step_configure_time_period()
        
        # Step 3: Configurazione del capitale iniziale
        self._step_configure_initial_capital()
        
        # Step 4: Configurazione della strategia di trading
        self._step_configure_strategy()
        
        # Step 5: Configurazione del numero di agenti
        self._step_configure_agents()
        
        # Step 6: Salvataggio della configurazione
        self._step_save_configuration()
        
        # Step 7: Scaricamento dei dati
        if self._confirm("Vuoi scaricare i dati storici per i simboli configurati?"):
            self._step_download_data()
        
        # Step 8: Esecuzione della simulazione
        if self._confirm("Vuoi eseguire una simulazione con la configurazione impostata?"):
            self._step_run_simulation()
        
        print("\nConfigurazione completata con successo!")
        print("Puoi ora utilizzare il sistema di trading algoritmico con la configurazione impostata.")
        print("Per avviare la dashboard web, esegui: python run_dashboard.py")
    
    def _confirm(self, question):
        """Chiede conferma all'utente."""
        while True:
            response = input(f"{question} (s/n): ").strip().lower()
            if response in ('s', 'si', 'sì', 'y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("Risposta non valida. Inserisci 's' o 'n'.")
    
    def _step_configure_symbols(self):
        """Configurazione dei simboli di trading."""
        print("\n--- STEP 1: CONFIGURAZIONE DEI SIMBOLI DI TRADING ---")
        print("Inserisci i simboli dei titoli da tradare separati da virgola (es. AAPL, MSFT, GOOGL).")
        print("Simboli correnti:", ", ".join(self.config['market']['symbols']))
        
        symbols_input = input("Nuovi simboli (lascia vuoto per mantenere i correnti): ").strip()
        
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            self.config['market']['symbols'] = symbols
            print(f"Simboli configurati: {', '.join(symbols)}")
    
    def _step_configure_time_period(self):
        """Configurazione del periodo di tempo."""
        print("\n--- STEP 2: CONFIGURAZIONE DEL PERIODO DI TEMPO ---")
        print("Inserisci la data di inizio e fine per la simulazione.")
        print(f"Periodo corrente: {self.config['market']['start_date']} - {self.config['market']['end_date']}")
        
        # Data di inizio
        while True:
            start_date = input("Data di inizio (YYYY-MM-DD, lascia vuoto per mantenere la corrente): ").strip()
            if not start_date:
                start_date = self.config['market']['start_date']
                break
            
            try:
                # Verifica che la data sia valida
                datetime.strptime(start_date, '%Y-%m-%d')
                break
            except ValueError:
                print("Formato data non valido. Usa il formato YYYY-MM-DD.")
        
        # Data di fine
        while True:
            end_date = input("Data di fine (YYYY-MM-DD, lascia vuoto per mantenere la corrente): ").strip()
            if not end_date:
                end_date = self.config['market']['end_date']
                break
            
            try:
                # Verifica che la data sia valida
                datetime.strptime(end_date, '%Y-%m-%d')
                
                # Verifica che la data di fine sia successiva alla data di inizio
                if end_date <= start_date:
                    print("La data di fine deve essere successiva alla data di inizio.")
                    continue
                
                break
            except ValueError:
                print("Formato data non valido. Usa il formato YYYY-MM-DD.")
        
        self.config['market']['start_date'] = start_date
        self.config['market']['end_date'] = end_date
        print(f"Periodo configurato: {start_date} - {end_date}")
    
    def _step_configure_initial_capital(self):
        """Configurazione del capitale iniziale."""
        print("\n--- STEP 3: CONFIGURAZIONE DEL CAPITALE INIZIALE ---")
        print("Inserisci il capitale iniziale per gli agenti di trading.")
        print(f"Capitale corrente: {self.config['trading']['initial_capital']} USD")
        
        while True:
            capital_input = input("Capitale iniziale (USD, lascia vuoto per mantenere il corrente): ").strip()
            if not capital_input:
                break
            
            try:
                capital = float(capital_input)
                if capital <= 0:
                    print("Il capitale deve essere positivo.")
                    continue
                
                self.config['trading']['initial_capital'] = capital
                break
            except ValueError:
                print("Valore non valido. Inserisci un numero.")
        
        print(f"Capitale configurato: {self.config['trading']['initial_capital']} USD")
    
    def _step_configure_strategy(self):
        """Configurazione della strategia di trading."""
        print("\n--- STEP 4: CONFIGURAZIONE DELLA STRATEGIA DI TRADING ---")
        print("Seleziona la strategia di trading da utilizzare.")
        print(f"Strategia corrente: {self.config['strategies']['active_strategy']}")
        
        strategies = self.config['strategies']['available_strategies']
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
        
        while True:
            strategy_input = input(f"Seleziona una strategia (1-{len(strategies)}, lascia vuoto per mantenere la corrente): ").strip()
            if not strategy_input:
                break
            
            try:
                strategy_index = int(strategy_input) - 1
                if 0 <= strategy_index < len(strategies):
                    self.config['strategies']['active_strategy'] = strategies[strategy_index]
                    break
                else:
                    print(f"Indice non valido. Inserisci un numero tra 1 e {len(strategies)}.")
            except ValueError:
                print("Valore non valido. Inserisci un numero.")
        
        print(f"Strategia configurata: {self.config['strategies']['active_strategy']}")
        
        # Configurazione dei parametri specifici della strategia
        strategy = self.config['strategies']['active_strategy']
        if strategy == 'mean_reversion':
            self._configure_mean_reversion_params()
        elif strategy == 'trend_following':
            self._configure_trend_following_params()
        elif strategy == 'neural_network':
            self._configure_neural_network_params()
    
    def _configure_mean_reversion_params(self):
        """Configurazione dei parametri per la strategia di mean reversion."""
        params = self.config['strategies']['strategy_params']['mean_reversion']
        print("\nConfigurazione dei parametri per la strategia Mean Reversion:")
        
        window = params.get('window', 20)
        window_input = input(f"Dimensione della finestra per la media mobile (corrente: {window}): ").strip()
        if window_input:
            try:
                window = int(window_input)
                if window <= 0:
                    print("La dimensione della finestra deve essere positiva. Mantengo il valore corrente.")
                else:
                    params['window'] = window
            except ValueError:
                print("Valore non valido. Mantengo il valore corrente.")
        
        print(f"Parametri configurati: window={params['window']}")
    
    def _configure_trend_following_params(self):
        """Configurazione dei parametri per la strategia di trend following."""
        params = self.config['strategies']['strategy_params']['trend_following']
        print("\nConfigurazione dei parametri per la strategia Trend Following:")
        
        short_window = params.get('short_window', 10)
        short_window_input = input(f"Dimensione della finestra breve (corrente: {short_window}): ").strip()
        if short_window_input:
            try:
                short_window = int(short_window_input)
                if short_window <= 0:
                    print("La dimensione della finestra deve essere positiva. Mantengo il valore corrente.")
                else:
                    params['short_window'] = short_window
            except ValueError:
                print("Valore non valido. Mantengo il valore corrente.")
        
        long_window = params.get('long_window', 50)
        long_window_input = input(f"Dimensione della finestra lunga (corrente: {long_window}): ").strip()
        if long_window_input:
            try:
                long_window = int(long_window_input)
                if long_window <= 0:
                    print("La dimensione della finestra deve essere positiva. Mantengo il valore corrente.")
                elif long_window <= short_window:
                    print("La finestra lunga deve essere maggiore della finestra breve. Mantengo il valore corrente.")
                else:
                    params['long_window'] = long_window
            except ValueError:
                print("Valore non valido. Mantengo il valore corrente.")
        
        print(f"Parametri configurati: short_window={params['short_window']}, long_window={params['long_window']}")
    
    def _configure_neural_network_params(self):
        """Configurazione dei parametri per la strategia basata su rete neurale."""
        params = self.config['strategies']['strategy_params']['neural_network']
        nn_config = self.config.get('neural_network', {})
        print("\nConfigurazione dei parametri per la strategia Neural Network:")
        
        # Tipo di modello
        model_types = ['lstm', 'cnn', 'transformer']
        model_type = params.get('model_type', 'lstm')
        print("Tipi di modello disponibili:")
        for i, mt in enumerate(model_types, 1):
            print(f"{i}. {mt}")
        
        model_input = input(f"Seleziona un tipo di modello (1-{len(model_types)}, corrente: {model_type}): ").strip()
        if model_input:
            try:
                model_index = int(model_input) - 1
                if 0 <= model_index < len(model_types):
                    params['model_type'] = model_types[model_index]
                else:
                    print(f"Indice non valido. Mantengo il valore corrente.")
            except ValueError:
                print("Valore non valido. Mantengo il valore corrente.")
        
        # Lunghezza della sequenza
        sequence_length = params.get('sequence_length', 10)
        sequence_input = input(f"Lunghezza della sequenza (corrente: {sequence_length}): ").strip()
        if sequence_input:
            try:
                sequence_length = int(sequence_input)
                if sequence_length <= 0:
                    print("La lunghezza della sequenza deve essere positiva. Mantengo il valore corrente.")
                else:
                    params['sequence_length'] = sequence_length
            except ValueError:
                print("Valore non valido. Mantengo il valore corrente.")
        
        print(f"Parametri configurati: model_type={params['model_type']}, sequence_length={params['sequence_length']}")
    
    def _step_configure_agents(self):
        """Configurazione del numero di agenti."""
        print("\n--- STEP 5: CONFIGURAZIONE DEL NUMERO DI AGENTI ---")
        print("Inserisci il numero di agenti di trading da creare.")
        print(f"Numero corrente: 5 (default)")
        
        while True:
            agents_input = input("Numero di agenti (lascia vuoto per mantenere il corrente): ").strip()
            if not agents_input:
                self.config['simulation'] = self.config.get('simulation', {})
                self.config['simulation']['num_agents'] = 5
                break
            
            try:
                num_agents = int(agents_input)
                if num_agents <= 0:
                    print("Il numero di agenti deve essere positivo.")
                    continue
                
                self.config['simulation'] = self.config.get('simulation', {})
                self.config['simulation']['num_agents'] = num_agents
                break
            except ValueError:
                print("Valore non valido. Inserisci un numero.")
        
        print(f"Numero di agenti configurato: {self.config['simulation']['num_agents']}")
    
    def _step_save_configuration(self):
        """Salvataggio della configurazione."""
        print("\n--- STEP 6: SALVATAGGIO DELLA CONFIGURAZIONE ---")
        print("Riepilogo della configurazione:")
        print(f"- Simboli: {', '.join(self.config['market']['symbols'])}")
        print(f"- Periodo: {self.config['market']['start_date']} - {self.config['market']['end_date']}")
        print(f"- Capitale iniziale: {self.config['trading']['initial_capital']} USD")
        print(f"- Strategia: {self.config['strategies']['active_strategy']}")
        print(f"- Numero di agenti: {self.config['simulation']['num_agents']}")
        
        if self._confirm("Vuoi salvare questa configurazione?"):
            # Aggiorna la configurazione
            update_config(self.config)
            print("Configurazione salvata con successo!")
        else:
            print("Configurazione non salvata.")
    
    def _step_download_data(self):
        """Scaricamento dei dati."""
        print("\n--- STEP 7: SCARICAMENTO DEI DATI ---")
        print("Scaricamento dei dati storici per i simboli configurati...")
        
        start_date = self.config['market']['start_date']
        end_date = self.config['market']['end_date']
        symbols = self.config['market']['symbols']
        
        for symbol in symbols:
            print(f"Scaricamento dei dati per {symbol}...")
            try:
                data = self.collector.get_stock_data(symbol, start_date, end_date, force_download=True)
                if data is not None:
                    print(f"Dati per {symbol} scaricati con successo: {len(data)} righe")
                else:
                    print(f"Errore nel download dei dati per {symbol}")
            except Exception as e:
                print(f"Errore nel download dei dati per {symbol}: {e}")
        
        print("Scaricamento dei dati completato!")
    
    def _step_run_simulation(self):
        """Esecuzione della simulazione."""
        print("\n--- STEP 8: ESECUZIONE DELLA SIMULAZIONE ---")
        print("Avvio della simulazione con la configurazione impostata...")
        
        try:
            # Crea il simulation manager
            sim_manager = SimulationManager(self.config)
            
            # Inizializza la simulazione
            if not sim_manager.initialize_simulation():
                print("Errore nell'inizializzazione della simulazione.")
                return
            
            # Crea gli agenti
            num_agents = self.config['simulation']['num_agents']
            if not sim_manager.create_agents(num_agents=num_agents):
                print("Errore nella creazione degli agenti.")
                return
            
            # Esegui la simulazione
            results = sim_manager.run_simulation()
            
            if results is None:
                print("Errore nell'esecuzione della simulazione.")
                return
            
            # Ottieni il riepilogo delle transazioni
            summary = sim_manager.get_transactions_summary()
            if summary:
                print("\nRiepilogo delle transazioni:")
                print(f"- Transazioni totali: {summary['total_transactions']}")
                print(f"- Acquisti: {summary['buy_transactions']}")
                print(f"- Vendite: {summary['sell_transactions']}")
                print(f"- Valore totale acquisti: {summary['total_buy_value']:.2f} USD")
                print(f"- Valore totale vendite: {summary['total_sell_value']:.2f} USD")
                print(f"- Valore netto: {summary['net_value']:.2f} USD")
            
            # Ottieni le performance degli agenti
            performances = sim_manager.get_agents_performance()
            if performances:
                print("\nPerformance degli agenti:")
                for perf in performances:
                    print(f"- Agente {perf['id']}: Rendimento {perf['percentage_return']:.2f}%")
            
            # Salva i risultati
            sim_manager.save_results()
            
            print("\nSimulazione completata con successo!")
            
        except Exception as e:
            print(f"Errore durante la simulazione: {e}")

def run_wizard():
    """Esegue il wizard."""
    wizard = SetupWizard()
    wizard.run()

if __name__ == '__main__':
    run_wizard() 