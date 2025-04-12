"""
Route del wizard per la configurazione guidata del sistema.

Questo modulo fornisce le route per il wizard di configurazione.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from flask import render_template, request, redirect, url_for, flash, jsonify

# Ottieni il percorso base del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Configura il logger
logger = logging.getLogger('wizard_routes')

def register_wizard_routes(app, CONFIG, state_manager, collector, get_available_strategies):
    """
    Registra le route del wizard nell'app Flask.
    
    Args:
        app: Istanza dell'app Flask
        CONFIG: Configurazione del sistema
        state_manager: Gestore dello stato della dashboard
        collector: Istanza del DataCollector
        get_available_strategies: Funzione per ottenere le strategie disponibili
    """
    
    @app.route('/wizard/configure/symbols', methods=['GET', 'POST'])
    def wizard_configure_symbols():
        """Pagina del wizard per configurare i simboli."""
        if request.method == 'POST':
            # Recupera i simboli dal form
            symbols_str = request.form.get('symbols', '')
            symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
            
            if not symbols:
                flash('Inserisci almeno un simbolo', 'error')
                return redirect(url_for('wizard_configure_symbols'))
            
            # Salva i simboli nella configurazione
            CONFIG['market']['symbols'] = symbols
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
            
            # Salva lo stato nel state_manager
            state_manager.update_tab_state('wizard', {
                'symbols': symbols_str,
                'step': 'symbols',
                'next_step': 'time_period'
            })
            
            # Verifica se è necessario scaricare i dati
            if request.form.get('verifySymbols') == 'on':
                missing_symbols = []
                for symbol in symbols:
                    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
                    if not os.path.exists(file_path):
                        missing_symbols.append(symbol)
                
                if missing_symbols:
                    flash(f'Dati mancanti per i simboli: {", ".join(missing_symbols)}. Scarica i dati dalla pagina di raccolta dati.', 'warning')
            
            flash('Simboli salvati con successo', 'success')
            return redirect(url_for('wizard_configure_time_period'))
        
        # Metodo GET
        # Recupera lo stato salvato
        wizard_state = state_manager.get_tab_state('wizard')
        symbols_str = wizard_state.get('symbols', '')
        current_symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        
        # Se non ci sono simboli salvati, usa quelli della configurazione
        if not current_symbols:
            current_symbols = CONFIG['market']['symbols']
        
        return render_template('wizard_symbols.html', current_symbols=current_symbols)
    
    @app.route('/wizard/configure/time_period', methods=['GET', 'POST'])
    def wizard_configure_time_period():
        """Pagina del wizard per configurare il periodo temporale."""
        if request.method == 'POST':
            # Recupera i dati dal form
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            interval = request.form.get('interval', '1d')
            
            # Validazione
            try:
                start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start_date_dt >= end_date_dt:
                    flash('La data di inizio deve essere precedente alla data di fine', 'error')
                    return redirect(url_for('wizard_configure_time_period'))
            except ValueError:
                flash('Formato data non valido', 'error')
                return redirect(url_for('wizard_configure_time_period'))
            
            # Salva i dati nella configurazione
            CONFIG['market']['start_date'] = start_date
            CONFIG['market']['end_date'] = end_date
            CONFIG['market']['interval'] = interval
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
            
            # Salva lo stato nel state_manager
            state_manager.update_tab_state('wizard', {
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval,
                'step': 'time_period',
                'next_step': 'capital'
            })
            
            flash('Periodo temporale salvato con successo', 'success')
            return redirect(url_for('wizard_configure_capital'))
        
        # Metodo GET
        # Recupera lo stato salvato
        wizard_state = state_manager.get_tab_state('wizard')
        
        # Date predefinite
        today = datetime.now()
        default_end_date = today.strftime('%Y-%m-%d')
        default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Usa i valori salvati o quelli predefiniti
        start_date = wizard_state.get('start_date', CONFIG['market'].get('start_date', default_start_date))
        end_date = wizard_state.get('end_date', CONFIG['market'].get('end_date', default_end_date))
        interval = wizard_state.get('interval', CONFIG['market'].get('interval', '1d'))
        
        return render_template('wizard_time_period.html', 
                              start_date=start_date, 
                              end_date=end_date, 
                              interval=interval,
                              default_start_date=default_start_date,
                              default_end_date=default_end_date)
    
    @app.route('/wizard/configure/capital', methods=['GET', 'POST'])
    def wizard_configure_capital():
        """Pagina del wizard per configurare il capitale iniziale."""
        if request.method == 'POST':
            # Recupera i dati dal form
            initial_capital = float(request.form.get('initial_capital', 100000))
            risk_level = int(request.form.get('risk_level', 5))
            use_stop_loss = request.form.get('use_stop_loss') == 'on'
            use_take_profit = request.form.get('use_take_profit') == 'on'
            stop_loss_percentage = float(request.form.get('stop_loss_percentage', 2.0))
            take_profit_percentage = float(request.form.get('take_profit_percentage', 5.0))
            
            # Salva i dati nella configurazione
            CONFIG['trading']['initial_capital'] = initial_capital
            
            # Assicurati che la sezione risk_management esista
            if 'risk_management' not in CONFIG['trading']:
                CONFIG['trading']['risk_management'] = {}
            
            CONFIG['trading']['risk_management']['risk_level'] = risk_level
            CONFIG['trading']['risk_management']['use_stop_loss'] = use_stop_loss
            CONFIG['trading']['risk_management']['use_take_profit'] = use_take_profit
            CONFIG['trading']['risk_management']['stop_loss_percentage'] = stop_loss_percentage
            CONFIG['trading']['risk_management']['take_profit_percentage'] = take_profit_percentage
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
            
            # Salva lo stato nel state_manager
            state_manager.update_tab_state('wizard', {
                'initial_capital': initial_capital,
                'risk_level': risk_level,
                'use_stop_loss': use_stop_loss,
                'use_take_profit': use_take_profit,
                'stop_loss_percentage': stop_loss_percentage,
                'take_profit_percentage': take_profit_percentage,
                'step': 'capital',
                'next_step': 'strategy'
            })
            
            flash('Capitale iniziale salvato con successo', 'success')
            return redirect(url_for('wizard_configure_strategy'))
        
        # Metodo GET
        # Recupera lo stato salvato
        wizard_state = state_manager.get_tab_state('wizard')
        
        # Usa i valori salvati o quelli predefiniti
        initial_capital = wizard_state.get('initial_capital', CONFIG['trading'].get('initial_capital', 100000))
        
        # Recupera i parametri di gestione del rischio
        risk_management = CONFIG['trading'].get('risk_management', {})
        risk_level = wizard_state.get('risk_level', risk_management.get('risk_level', 5))
        use_stop_loss = wizard_state.get('use_stop_loss', risk_management.get('use_stop_loss', True))
        use_take_profit = wizard_state.get('use_take_profit', risk_management.get('use_take_profit', True))
        stop_loss_percentage = wizard_state.get('stop_loss_percentage', risk_management.get('stop_loss_percentage', 2.0))
        take_profit_percentage = wizard_state.get('take_profit_percentage', risk_management.get('take_profit_percentage', 5.0))
        
        return render_template('wizard_capital.html', 
                              initial_capital=initial_capital,
                              risk_level=risk_level,
                              use_stop_loss=use_stop_loss,
                              use_take_profit=use_take_profit,
                              stop_loss_percentage=stop_loss_percentage,
                              take_profit_percentage=take_profit_percentage)
    
    @app.route('/wizard/configure/strategy', methods=['GET', 'POST'])
    def wizard_configure_strategy():
        """Pagina del wizard per configurare la strategia."""
        if request.method == 'POST':
            # Recupera i dati dal form
            strategy = request.form.get('strategy', 'random')
            
            # Parametri specifici per ogni strategia
            strategy_params = {}
            
            if strategy == 'mean_reversion':
                strategy_params['window'] = int(request.form.get('mean_reversion_window', 20))
                strategy_params['threshold'] = float(request.form.get('mean_reversion_threshold', 2.0))
            elif strategy == 'trend_following':
                strategy_params['short_window'] = int(request.form.get('trend_short_window', 10))
                strategy_params['long_window'] = int(request.form.get('trend_long_window', 50))
            elif strategy == 'value_investing':
                strategy_params['pe_threshold'] = float(request.form.get('value_pe_threshold', 15.0))
                strategy_params['pb_threshold'] = float(request.form.get('value_pb_threshold', 1.5))
            elif strategy == 'neural_network':
                strategy_params['model_type'] = request.form.get('nn_model_type', 'lstm')
                strategy_params['sequence_length'] = int(request.form.get('nn_sequence_length', 10))
                strategy_params['threshold'] = float(request.form.get('nn_threshold', 1.0))
            
            # Salva i dati nella configurazione
            CONFIG['strategies']['active_strategy'] = strategy
            
            # Assicurati che la sezione strategy_params esista
            if 'strategy_params' not in CONFIG['strategies']:
                CONFIG['strategies']['strategy_params'] = {}
            
            CONFIG['strategies']['strategy_params'][strategy] = strategy_params
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
            
            # Salva lo stato nel state_manager
            state_manager.update_tab_state('wizard', {
                'strategy': strategy,
                'strategy_params': strategy_params,
                'step': 'strategy',
                'next_step': 'agents'
            })
            
            flash('Strategia salvata con successo', 'success')
            return redirect(url_for('wizard_configure_agents'))
        
        # Metodo GET
        # Recupera lo stato salvato
        wizard_state = state_manager.get_tab_state('wizard')
        
        # Ottieni le strategie disponibili
        strategies_info = get_available_strategies()
        
        # Usa i valori salvati o quelli predefiniti
        active_strategy = wizard_state.get('strategy', CONFIG['strategies'].get('active_strategy', 'random'))
        
        # Recupera i parametri della strategia
        strategy_params = CONFIG['strategies'].get('strategy_params', {})
        
        return render_template('wizard_strategy.html', 
                              strategies_info=strategies_info,
                              active_strategy=active_strategy,
                              strategy_params=strategy_params)
    
    @app.route('/wizard/configure/agents', methods=['GET', 'POST'])
    def wizard_configure_agents():
        """Pagina del wizard per configurare gli agenti."""
        if request.method == 'POST':
            # Recupera i dati dal form
            num_agents = int(request.form.get('num_agents', 5))
            use_mixed_strategies = request.form.get('use_mixed_strategies') == 'on'
            use_variable_capital = request.form.get('use_variable_capital') == 'on'
            
            # Parametri per strategie miste
            strategy_distribution = {}
            if use_mixed_strategies:
                strategies_info = get_available_strategies()
                for strategy_id, strategy_info in strategies_info.items():
                    if strategy_info['status'] == 'implemented':
                        percentage = int(request.form.get(f'strategy_pct_{strategy_id}', 0))
                        if percentage > 0:
                            strategy_distribution[strategy_id] = percentage
            
            # Parametri per capitale variabile
            min_capital = float(request.form.get('min_capital', 50000))
            max_capital = float(request.form.get('max_capital', 150000))
            
            # Salva i dati nella configurazione
            CONFIG['trading']['max_agents'] = num_agents
            CONFIG['trading']['use_mixed_strategies'] = use_mixed_strategies
            CONFIG['trading']['use_variable_capital'] = use_variable_capital
            
            if use_mixed_strategies:
                CONFIG['trading']['strategy_distribution'] = strategy_distribution
            
            if use_variable_capital:
                CONFIG['trading']['min_capital'] = min_capital
                CONFIG['trading']['max_capital'] = max_capital
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
            
            # Salva lo stato nel state_manager
            state_manager.update_tab_state('wizard', {
                'num_agents': num_agents,
                'use_mixed_strategies': use_mixed_strategies,
                'use_variable_capital': use_variable_capital,
                'strategy_distribution': strategy_distribution,
                'min_capital': min_capital,
                'max_capital': max_capital,
                'step': 'agents',
                'next_step': 'complete'
            })
            
            flash('Configurazione agenti salvata con successo', 'success')
            return redirect(url_for('wizard_complete'))
        
        # Metodo GET
        # Recupera lo stato salvato
        wizard_state = state_manager.get_tab_state('wizard')
        
        # Ottieni le strategie disponibili
        strategies_info = get_available_strategies()
        
        # Usa i valori salvati o quelli predefiniti
        num_agents = wizard_state.get('num_agents', CONFIG['trading'].get('max_agents', 5))
        use_mixed_strategies = wizard_state.get('use_mixed_strategies', CONFIG['trading'].get('use_mixed_strategies', False))
        use_variable_capital = wizard_state.get('use_variable_capital', CONFIG['trading'].get('use_variable_capital', False))
        
        # Recupera la distribuzione delle strategie
        strategy_distribution = wizard_state.get('strategy_distribution', CONFIG['trading'].get('strategy_distribution', {}))
        
        # Recupera i parametri del capitale variabile
        min_capital = wizard_state.get('min_capital', CONFIG['trading'].get('min_capital', 50000))
        max_capital = wizard_state.get('max_capital', CONFIG['trading'].get('max_capital', 150000))
        
        # Ottieni il nome della strategia attiva
        active_strategy = CONFIG['strategies'].get('active_strategy', 'random')
        active_strategy_name = strategies_info.get(active_strategy, {}).get('name', active_strategy)
        
        return render_template('wizard_agents.html', 
                              num_agents=num_agents,
                              use_mixed_strategies=use_mixed_strategies,
                              use_variable_capital=use_variable_capital,
                              strategy_distribution=strategy_distribution,
                              min_capital=min_capital,
                              max_capital=max_capital,
                              strategies_info=strategies_info,
                              active_strategy_name=active_strategy_name)
    
    @app.route('/wizard/complete', methods=['GET'])
    def wizard_complete():
        """Pagina finale del wizard con il riepilogo della configurazione."""
        # Ottieni le strategie disponibili
        strategies_info = get_available_strategies()
        
        # Ottieni il nome della strategia attiva
        active_strategy = CONFIG['strategies'].get('active_strategy', 'random')
        strategy_name = strategies_info.get(active_strategy, {}).get('name', active_strategy)
        
        # Ottieni i parametri della strategia
        strategy_params = CONFIG['strategies'].get('strategy_params', {}).get(active_strategy, {})
        
        return render_template('wizard_complete.html', 
                              config=CONFIG,
                              strategy_name=strategy_name,
                              strategy_params=strategy_params,
                              strategies_info=strategies_info)
    
    @app.route('/wizard/save_config', methods=['POST'])
    def wizard_save_config():
        """Salva la configurazione finale e avvia la simulazione se richiesto."""
        action = request.form.get('action', 'save')
        
        # Salva la configurazione
        with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
            json.dump(CONFIG, f, indent=4)
        
        # Copia la configurazione in config_user.json
        with open(os.path.join(BASE_DIR, 'config_user.json'), 'w') as f:
            json.dump(CONFIG, f, indent=4)
        
        flash('Configurazione salvata con successo', 'success')
        
        # Se richiesto, avvia la simulazione
        if action == 'save_and_run' or request.form.get('run_simulation') == 'true':
            return redirect(url_for('simulation'))
        
        return redirect(url_for('index'))
