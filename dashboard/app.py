#!/usr/bin/env python
"""
Dashboard per la gestione del sistema di trading algoritmico.

Questo modulo fornisce un'interfaccia web per la gestione del sistema.
"""

import os
import sys
import json
import logging
import argparse
import math
import locale
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_socketio import SocketIO, emit
import threading

# Aggiungi la directory root al path di Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ora possiamo importare i moduli
from market_simulator.simulation import SimulationManager
from market_simulator.monitored_simulation import MonitoredSimulationManager
from data.collector import DataCollector
from interface.wizard import SetupWizard
from neural_network.model_trainer import ModelTrainer
from neural_network.integration import NeuralNetworkIntegration
from trading_strategy import get_available_strategies, create_strategy
from logging.handlers import RotatingFileHandler
# Importazioni assolute, non relative
from dashboard.state_manager import DashboardStateManager
from dashboard.websocket_manager import WebSocketManager
from dashboard.visualization_manager import VisualizationManager
from dashboard.real_time_monitor import RealTimeMonitor
from dashboard.cache_manager import CacheManager
from dashboard.report_generator import ReportGenerator
from dashboard.wizard_routes import register_wizard_routes
from dashboard.prediction_routes import register_prediction_routes
from dashboard.neural_network_routes import neural_network_bp, init_neural_network_routes

# Imposta la localizzazione per formattare i numeri con "." come separatore delle migliaia e "," per i decimali
locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')

# Ottieni il percorso base del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Assicurati che le directory necessarie esistano
for d in [DATA_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Configura il logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(LOGS_DIR, 'dashboard.log'),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

# Carica la configurazione dal file
try:
    with open(os.path.join(BASE_DIR, 'config_updated.json'), 'r') as f:
        CONFIG = json.load(f)
    
    # Assicurati che la chiave 'strategies' esista
    if 'strategies' not in CONFIG:
        CONFIG['strategies'] = {
            'active_strategy': CONFIG['trading'].get('strategy', 'random'),
            'strategy_params': {}
        }
    
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Errore nel caricamento della configurazione: {e}")
    # Configurazione predefinita
    CONFIG = {
        "market": {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "start_date": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            "end_date": datetime.now().strftime('%Y-%m-%d'),
            "interval": "1d"
        },
        "trading": {
            "strategy": "mean_reversion",
            "initial_capital": 10000,
            "max_agents": 5
        },
        "strategies": {
            "active_strategy": "mean_reversion",
            "strategy_params": {}
        }
    }
    # Salva la configurazione predefinita
    with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
        json.dump(CONFIG, f, indent=4)

# Inizializza Flask e i moduli
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')  # Per flash/sessioni

# Inizializza i manager
state_manager = DashboardStateManager()
websocket_manager = WebSocketManager(app)
visualization_manager = VisualizationManager()
real_time_monitor = RealTimeMonitor(websocket_manager, state_manager)
cache_manager = CacheManager(cache_dir=os.path.join(BASE_DIR, 'dashboard', 'cache'))
report_generator = ReportGenerator(output_dir=REPORTS_DIR)

# Inizializza gli altri moduli
collector = DataCollector(data_dir=DATA_DIR)
simulator = MonitoredSimulationManager(CONFIG, real_time_monitor)
wizard = SetupWizard()
model_trainer = ModelTrainer()
neural_network_integration = NeuralNetworkIntegration(model_trainer, collector)

# Registra i blueprint
app.register_blueprint(neural_network_bp)
init_neural_network_routes(collector)

# Route principali
@app.route('/')
def index():
    """Pagina principale/dashboard"""
    try:
        # Ottieni informazioni di base sul sistema
        system_info = {
            'data_symbols': len(os.listdir(DATA_DIR)),
            'models_count': len(os.listdir(os.path.join(BASE_DIR, 'models'))) if os.path.exists(os.path.join(BASE_DIR, 'models')) else 0,
            'reports_count': len(os.listdir(REPORTS_DIR)),
            'config_status': 'Configurato' if os.path.exists(os.path.join(BASE_DIR, 'config_updated.json')) else 'Non configurato',
            'last_simulation': 'N/A'
        }
        
        # Cerca l'ultima simulazione
        if os.path.exists(REPORTS_DIR):
            report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
            if report_files:
                latest_report = max(report_files, key=lambda x: os.path.getctime(os.path.join(REPORTS_DIR, x)))
                system_info['last_simulation'] = latest_report.replace('.json', '')
        
        # TODO: Ottieni informazioni sui dati disponibili
        
        return render_template('index.html', system_info=system_info, config=CONFIG)
    except Exception as e:
        logger.error(f"Errore nel rendering della pagina principale: {e}")
        return render_template('error.html', error=str(e))

@app.route('/available_data')
def available_data():
    """Visualizza i dati disponibili"""
    try:
        data = {}
        
        # Elenco dei file di dati
        for file in os.listdir(DATA_DIR):
            if file.endswith('.csv'):
                file_path = os.path.join(DATA_DIR, file)
                symbol = file.replace('.csv', '')
                
                # Informazioni sul file
                file_size = os.path.getsize(file_path)
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Carica i primi 5 record per un'anteprima
                df = pd.read_csv(file_path, nrows=5)
                preview = df.to_dict('records')
                
                data[symbol] = {
                    'file_path': file_path,
                    'size': get_file_size_str(file_size),
                    'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'columns': df.columns.tolist(),
                    'preview': preview
                }
        
        return render_template('available_data.html', data=data)
    except Exception as e:
        logger.error(f"Errore nella visualizzazione dei dati disponibili: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/data_collection', methods=['GET', 'POST'])
def data_collection():
    """Gestisce la raccolta dei dati"""
    try:
        if request.method == 'POST':
            # Recupera i dati dal form
            symbols = request.form.get('symbols', '').split(',')
            symbols = [s.strip() for s in symbols if s.strip()]
            
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            interval = request.form.get('interval', '1d')
            
            if not symbols:
                flash('Inserisci almeno un simbolo', 'error')
                return redirect(url_for('data_collection'))
            
            # Aggiorna la configurazione
            CONFIG['market']['symbols'] = symbols
            CONFIG['market']['start_date'] = start_date
            CONFIG['market']['end_date'] = end_date
            CONFIG['market']['interval'] = interval
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
            
            # Avvia il download dei dati
            try:
                collector.download_data(symbols, start_date, end_date, interval)
                flash('Download dei dati completato con successo', 'success')
            except Exception as e:
                flash(f'Errore durante il download dei dati: {str(e)}', 'error')
            
            return redirect(url_for('data_collection'))
        
        # Carica i dati disponibili
        available_data = {}
        try:
            for symbol in os.listdir(DATA_DIR):
                if symbol.endswith('.csv'):
                    symbol_name = symbol[:-4]  # Rimuove .csv
                    file_path = os.path.join(DATA_DIR, symbol)
                    
                    if os.path.exists(file_path):
                        # Ottieni informazioni sul file
                        file_stats = os.stat(file_path)
                        size_bytes = file_stats.st_size
                        modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                        
                        # Carica il dataframe per ottenere info
                        try:
                            df = pd.read_csv(file_path)
                            start_date = df['Date'].iloc[0] if 'Date' in df else 'N/A'
                            end_date = df['Date'].iloc[-1] if 'Date' in df else 'N/A'
                            rows = len(df)
                            
                            available_data[symbol_name] = {
                                'start_date': start_date,
                                'end_date': end_date,
                                'rows': rows,
                                'size': get_file_size_str(size_bytes),
                                'last_update': get_time_ago(modified_time)
                            }
                        except Exception as e:
                            logger.error(f"Errore nell'elaborazione del file {symbol}: {e}")
        except Exception as e:
            logger.error(f"Errore nel caricamento dei dati disponibili: {e}")
        
        # Genera date predefinite
        today = datetime.now()
        default_end_date = today.strftime('%Y-%m-%d')
        default_start_date = (today - timedelta(days=365*2)).strftime('%Y-%m-%d')
        logger.info(f"Date predefinite generate: {default_start_date} a {default_end_date}")
        
        # Recupera lo stato salvato
        tab_state = state_manager.get_tab_state('data_collection')
        symbols = tab_state.get('symbols', '').split(',') if 'symbols' in tab_state else []
        symbols = [s.strip() for s in symbols if s.strip()]
        
        # Prepara il contesto per il template
        context = {
            'available_data': available_data,
            'symbols': symbols,
            'start_date': tab_state.get('startDate', default_start_date),
            'end_date': tab_state.get('endDate', default_end_date),
            'interval': tab_state.get('interval', '1d'),
            'default_start_date': default_start_date,
            'default_end_date': default_end_date
        }
        
        return render_template('data_collection.html', **context)
    except Exception as e:
        logger.error(f"Errore nella pagina di raccolta dati: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/api/check_data', methods=['POST'])
def check_data():
    """API per verificare i dati disponibili"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        result = {}
        for symbol in symbols:
            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                    
                    if date_col:
                        result[symbol] = {
                            'available': True,
                            'start_date': df[date_col].iloc[0] if not df.empty else 'N/A',
                            'end_date': df[date_col].iloc[-1] if not df.empty else 'N/A',
                            'rows': len(df)
                        }
                    else:
                        result[symbol] = {
                            'available': True,
                            'start_date': 'N/A',
                            'end_date': 'N/A',
                            'rows': len(df)
                        }
                except Exception as e:
                    result[symbol] = {
                        'available': False,
                        'error': str(e)
                    }
            else:
                result[symbol] = {
                    'available': False
                }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Errore nella verifica dei dati disponibili: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Gestisce la configurazione del sistema"""
    if request.method == 'POST':
        try:
            # Aggiorna la configurazione
            CONFIG['market']['symbols'] = request.form.get('symbols', '').split(',')
            CONFIG['market']['start_date'] = request.form.get('start_date')
            CONFIG['market']['end_date'] = request.form.get('end_date')
            CONFIG['market']['interval'] = request.form.get('interval')
            CONFIG['trading']['strategy'] = request.form.get('strategy')
            CONFIG['trading']['initial_capital'] = float(request.form.get('initial_capital', 10000))
            CONFIG['trading']['max_agents'] = int(request.form.get('max_agents', 5))
            
            # Verifica che la strategia selezionata sia implementata
            strategy_name = request.form.get('active_strategy')
            available_strategies = get_available_strategies()
            if strategy_name not in available_strategies or available_strategies[strategy_name]['status'] != 'implemented':
                flash(f'La strategia selezionata ({strategy_name}) non è disponibile o non è ancora implementata.', 'warning')
                # Non blocchiamo il salvataggio, ma avvisiamo l'utente.
                # Potremmo voler reimpostare a una strategia di default se non implementata
                # CONFIG['strategies']['active_strategy'] = 'random' # Esempio
            else:
                CONFIG['strategies']['active_strategy'] = strategy_name
            
            # Salva la configurazione
            with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
                json.dump(CONFIG, f, indent=4)
                
            flash('Configurazione salvata con successo', 'success')
            return redirect(url_for('config'))
        except Exception as e:
            logger.error(f"Errore nel salvataggio della configurazione: {e}")
            flash(f'Errore: {str(e)}', 'error')
    
    # Metodo GET
    # Ottieni le strategie disponibili (con il loro stato)
    strategies_info = get_available_strategies()
    
    # Carica la configurazione corrente
    # ... (logica per caricare CONFIG)
    
    return render_template('config.html', config=CONFIG, strategies_info=strategies_info)

@app.route('/simulation')
def simulation():
    """Visualizza la pagina di simulazione"""
    try:
        # Ottieni le strategie disponibili (con il loro stato)
        strategies_info = get_available_strategies() 
        return render_template('simulation.html', config=CONFIG, strategies_info=strategies_info)
    except Exception as e:
        logger.error(f"Errore nel rendering della pagina di simulazione: {e}", exc_info=True)
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Endpoint per l'esecuzione della simulazione"""
    try:
        # Recupera i parametri dalla richiesta
        data = request.get_json() or {}
        num_agents = int(data.get('num_agents', CONFIG['trading']['max_agents']))
        strategy = data.get('strategy', CONFIG['strategies']['active_strategy'])
        initial_capital = float(data.get('initial_capital', CONFIG['trading']['initial_capital']))
        
        # ++ AGGIUNTA VERIFICA STRATEGIA IMPLEMENTATA PRIMA DI AVVIARE ++
        available_strategies = get_available_strategies()
        if strategy not in available_strategies or available_strategies[strategy]['status'] != 'implemented':
            error_msg = f'Impossibile avviare la simulazione: la strategia \'{strategy}\' non è implementata.'
            logger.error(error_msg)
            return jsonify({'status': 'error', 'error': error_msg}), 400

        # Aggiorna la configurazione
        CONFIG['trading']['max_agents'] = num_agents
        CONFIG['trading']['initial_capital'] = initial_capital
        CONFIG['strategies']['active_strategy'] = strategy
        
        # Salva la configurazione
        with open(os.path.join(BASE_DIR, 'config_updated.json'), 'w') as f:
            json.dump(CONFIG, f, indent=4)
        
        # Verifica che i dati siano disponibili per tutti i simboli
        missing_symbols = []
        for symbol in CONFIG['market']['symbols']:
            cache_file = os.path.join(DATA_DIR, f"{symbol}.csv")
            if not os.path.exists(cache_file):
                missing_symbols.append(symbol)
        
        if missing_symbols:
            return jsonify({
                'error': f'Dati mancanti per i seguenti simboli: {", ".join(missing_symbols)}'
            }), 400
        
        # Connetti il monitor in tempo reale
        simulator.set_real_time_monitor(real_time_monitor)
        
        # Inizializza e avvia la simulazione in un thread separato
        def run_simulation_task():
            try:
                # Inizializza la simulazione
                # La creazione agenti ora usa create_strategy che restituisce None se non implementata
                # -> SimulationManager.create_agents deve gestire il caso in cui strategy è None
                if not simulator.initialize_simulation():
                     raise Exception("Errore nell'inizializzazione della simulazione.")
                if not simulator.create_agents(num_agents):
                     # create_agents dovrebbe loggare l'errore specifico
                     raise Exception(f"Errore nella creazione degli agenti (strategia '{strategy}' valida?).")
                
                # Esegui la simulazione
                results = simulator.run_simulation()
                
                # Aggiorna lo stato della dashboard
                if results:
                    state_manager.update_market_simulation_state({
                        'status': 'completed',
                        'last_update': datetime.now().isoformat(),
                        'results': {
                            'timestamp': datetime.now().isoformat(),
                            'num_agents': num_agents,
                            'strategy': strategy,
                            'initial_capital': initial_capital,
                            'agents_performance': [agent for agent in results['agents']],
                            'transactions_count': len(results['transactions'])
                        }
                    })
                    
                    # Emetti l'aggiornamento finale
                    websocket_manager.emit_market_simulation_update({
                        'status': 'completed',
                        'message': 'Simulazione completata con successo',
                        'results': results
                    })
            except Exception as e:
                logger.error(f"Errore nell'esecuzione della simulazione nel thread: {e}", exc_info=True)
                websocket_manager.emit_error(str(e), 'simulation')
                # Aggiorna lo stato per riflettere l'errore
                state_manager.update_market_simulation_state({
                    'status': 'error',
                    'error': str(e),
                    'last_update': datetime.now().isoformat()
                })
        
        # Avvia la simulazione in un thread separato
        simulation_thread = threading.Thread(target=run_simulation_task)
        simulation_thread.daemon = True
        simulation_thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Simulazione avviata con successo'
        })
    except Exception as e:
        logger.error(f"Errore nell'avvio della simulazione (endpoint /run_simulation): {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/reports')
def reports():
    """Visualizza i report disponibili"""
    try:
        # Directory contenente i report
        report_dir = os.path.join(BASE_DIR, 'reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        logger.info(f"Caricamento della pagina dei report iniziato")
        logger.info(f"Esaminando i file in {report_dir}")
        
        reports_list = []
        
        # Elenca tutti i file JSON nella directory reports
        for filename in os.listdir(report_dir):
            if filename.endswith('.json'):
                logger.info(f"Elaborazione del file: {filename}")
                report_path = os.path.join(report_dir, filename)
                
                # Ottieni informazioni sul file
                file_stats = os.stat(report_path)
                size_bytes = file_stats.st_size
                size_str = get_file_size_str(size_bytes)
                modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                time_ago = get_time_ago(modified_time)
                
                try:
                    # Carica il file JSON per estrarre informazioni
                    with open(report_path, 'r') as f:
                        report_data = json.load(f)
                    
                    # Estrai informazioni rilevanti
                    report_id = os.path.splitext(filename)[0]
                    timestamp = report_data.get('timestamp', modified_time.strftime('%Y-%m-%d %H:%M:%S'))
                    symbols = report_data.get('symbols', ['N/A'])
                    start_date = report_data.get('start_date', 'N/A')
                    end_date = report_data.get('end_date', 'N/A')
                    strategy = report_data.get('strategy', 'N/A')
                    
                    # Calcola rendimento se disponibile
                    returns = report_data.get('returns', {})
                    overall_return = returns.get('overall_return', 0) * 100 if isinstance(returns, dict) else 0
                    
                    reports_list.append({
                        'id': report_id,
                        'name': filename,
                        'path': report_path,
                        'timestamp': timestamp,
                        'time_ago': time_ago,
                        'symbols': symbols,
                        'start_date': start_date,
                        'end_date': end_date,
                        'strategy': strategy,
                        'return': overall_return,
                        'size': size_str
                    })
                except Exception as e:
                    logger.error(f"Errore nell'elaborazione del file {filename}: {e}")
                    reports_list.append({
                        'id': os.path.splitext(filename)[0],
                        'name': filename,
                        'path': report_path,
                        'timestamp': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'time_ago': time_ago,
                        'symbols': ['N/A'],
                        'start_date': 'N/A',
                        'end_date': 'N/A',
                        'strategy': 'N/A',
                        'return': 0,
                        'size': size_str
                    })
        
        # Ordina i report per data (più recenti prima)
        reports_list = sorted(reports_list, key=lambda x: x['timestamp'], reverse=True)
        logger.info(f"Trovati {len(reports_list)} report")
        logger.info("Report ordinati correttamente")
        logger.info("Rendering del template reports.html")
        
        return render_template('reports.html', reports=reports_list)
    except Exception as e:
        logger.error(f"Errore nella visualizzazione dei report: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/view_report/<report_id>')
def view_report(report_id):
    """Visualizza un report specifico con grafici interattivi"""
    try:
        report_path = os.path.join(BASE_DIR, 'reports', f"{report_id}.json")
        
        if not os.path.exists(report_path):
            flash(f'Report non trovato: {report_id}', 'danger')
            return redirect(url_for('reports'))
        
        # Ottieni informazioni sul file
        file_stats = os.stat(report_path)
        size_bytes = file_stats.st_size
        size_str = get_file_size_str(size_bytes)
        modified_time = datetime.fromtimestamp(file_stats.st_mtime)
        
        # Carica il file JSON per estrarre dati per la visualizzazione
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        # Prepara informazioni generali
        report_info = {
            'description': report_data.get('description', 'Simulation Report'),
            'timestamp': report_data.get('timestamp', modified_time.strftime('%Y-%m-%d %H:%M:%S')),
            'size': size_str
        }
        
        # Prepara statistiche di trading
        report_stats = prepare_report_stats(report_data)
        
        # Prepara dati per il grafico
        simulation_data = prepare_chart_data(report_data)
        
        # Prepara dati di trading
        trades = prepare_trades_data(report_data)
        
        # Simboli disponibili
        symbols = report_data.get('symbols', [])
        
        return render_template('report_visualization.html', 
                              report_id=report_id,
                              report_info=report_info,
                              report_stats=report_stats,
                              symbols=symbols,
                              trades=trades,
                              simulation_data=simulation_data)
    except Exception as e:
        logger.error(f"Errore nella visualizzazione del report {report_id}: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('reports'))

@app.route('/download_report/<report_id>')
def download_report(report_id):
    """Scarica un report specifico"""
    try:
        report_path = os.path.join(BASE_DIR, 'reports', f"{report_id}.json")
        
        if not os.path.exists(report_path):
            flash(f'Report non trovato: {report_id}', 'danger')
            return redirect(url_for('reports'))
        
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Errore nel download del report {report_id}: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('reports'))

@app.route('/delete_report/<report_id>', methods=['GET'])
def delete_report(report_id):
    """Elimina un report specifico"""
    try:
        report_path = os.path.join(BASE_DIR, 'reports', f"{report_id}.json")
        
        if not os.path.exists(report_path):
            flash(f'Report non trovato: {report_id}', 'danger')
            return redirect(url_for('reports'))
        
        os.remove(report_path)
        flash(f'Report eliminato con successo', 'success')
        return redirect(url_for('reports'))
    except Exception as e:
        logger.error(f"Errore nell'eliminazione del report {report_id}: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('reports'))

# Funzioni di utilità per i report

def get_file_size_str(size_bytes):
    """Converte i byte in una stringa leggibile (KB, MB, etc.)"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def get_time_ago(timestamp):
    """Calcola il tempo trascorso da una data"""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 365:
        years = diff.days // 365
        return f"{years} {'anno' if years == 1 else 'anni'} fa"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} {'mese' if months == 1 else 'mesi'} fa"
    elif diff.days > 0:
        return f"{diff.days} {'giorno' if diff.days == 1 else 'giorni'} fa"
    elif diff.seconds // 3600 > 0:
        hours = diff.seconds // 3600
        return f"{hours} {'ora' if hours == 1 else 'ore'} fa"
    elif diff.seconds // 60 > 0:
        minutes = diff.seconds // 60
        return f"{minutes} {'minuto' if minutes == 1 else 'minuti'} fa"
    else:
        return "poco fa"

def prepare_report_stats(report_data):
    """Prepara le statistiche di trading dal report"""
    stats = {}
    
    # Estrai statistiche base
    returns = report_data.get('returns', {})
    if isinstance(returns, dict):
        stats['Rendimento Totale'] = f"{returns.get('overall_return', 0) * 100:.2f}%"
        stats['Rendimento Annualizzato'] = f"{returns.get('annualized_return', 0) * 100:.2f}%"
        stats['Volatilità'] = f"{returns.get('volatility', 0) * 100:.2f}%"
        stats['Sharpe Ratio'] = f"{returns.get('sharpe_ratio', 0):.2f}"
        stats['Max Drawdown'] = f"{returns.get('max_drawdown', 0) * 100:.2f}%"
    
    # Estrai metriche di trading
    metrics = report_data.get('metrics', {})
    if isinstance(metrics, dict):
        stats['N. Operazioni'] = metrics.get('total_trades', 0)
        stats['% Operazioni Vincenti'] = f"{metrics.get('win_rate', 0) * 100:.2f}%"
        stats['Rapporto Profitto/Perdita'] = f"{metrics.get('profit_loss_ratio', 0):.2f}"
        stats['Guadagno Medio'] = f"{metrics.get('avg_profit', 0):.2f}"
        stats['Perdita Media'] = f"{metrics.get('avg_loss', 0):.2f}"
    
    return stats

def prepare_trades_data(report_data):
    """Prepara i dati delle operazioni di trading"""
    trades_data = report_data.get('trades', [])
    trades = []
    
    if not trades_data:
        return []
    
    for trade in trades_data:
        trades.append({
            'date': trade.get('date', 'N/A'),
            'symbol': trade.get('symbol', 'N/A'),
            'type': trade.get('type', 'N/A'),
            'price': f"{trade.get('price', 0):.2f}",
            'quantity': trade.get('quantity', 0)
        })
    
    return trades

def prepare_chart_data(report_data):
    """Prepara i dati per i grafici"""
    chart_data = {}
    
    # Ottieni dati OHLCV per ogni simbolo
    market_data = report_data.get('market_data', {})
    symbols = report_data.get('symbols', [])
    
    for symbol in symbols:
        symbol_data = market_data.get(symbol, {})
        if not symbol_data:
            continue
            
        dates = symbol_data.get('dates', [])
        open_prices = symbol_data.get('open', [])
        high_prices = symbol_data.get('high', [])
        low_prices = symbol_data.get('low', [])
        close_prices = symbol_data.get('close', [])
        volumes = symbol_data.get('volume', [])
        
        # Calcola medie mobili
        ma20 = calculate_moving_average(close_prices, 20)
        ma50 = calculate_moving_average(close_prices, 50)
        ma200 = calculate_moving_average(close_prices, 200)
        
        chart_data[symbol] = {
            'dates': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200
        }
    
    # Dati del portafoglio
    portfolio = report_data.get('portfolio', {})
    if portfolio:
        chart_data['portfolio'] = {
            'dates': portfolio.get('dates', []),
            'equity': portfolio.get('equity', []),
            'benchmark': portfolio.get('benchmark', [])
        }
    
    return chart_data

def calculate_moving_average(data, window):
    """Calcola la media mobile"""
    if not data or len(data) < window:
        return []
        
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(None)
        else:
            window_sum = sum(data[i-(window-1):i+1])
            result.append(window_sum / window)
            
    return result

# Aggiungi le nuove route per la gestione dello stato e WebSocket
@app.route('/api/state/<tab_name>')
def get_tab_state(tab_name):
    """Endpoint per recuperare lo stato di una tab"""
    return jsonify(state_manager.get_tab_state(tab_name))

@app.route('/api/state/<tab_name>', methods=['POST'])
def update_tab_state(tab_name):
    """Endpoint per aggiornare lo stato di una tab"""
    data = request.get_json()
    state_manager.update_tab_state(tab_name, data)
    return jsonify({'status': 'success'})

@app.route('/api/data_preview/<symbol>')
def data_preview(symbol):
    """Restituisce un'anteprima dei dati per un simbolo specifico"""
    try:
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path):
            return jsonify({'error': f'File non trovato per il simbolo {symbol}'})
        
        # Carica i primi 20 record
        df = pd.read_csv(file_path, nrows=20)
        
        # Prepara i dati per la risposta JSON
        return jsonify({
            'columns': df.columns.tolist(),
            'data': df.values.tolist()
        })
    except Exception as e:
        logger.error(f"Errore nella preview dei dati per {symbol}: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/delete_data/<symbol>', methods=['DELETE'])
def delete_data(symbol):
    """API per eliminare un file di dati"""
    try:
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path):
            return jsonify({'error': f'File non trovato per il simbolo {symbol}'}), 404
        
        # Elimina il file
        os.remove(file_path)
        logger.info(f"Dati eliminati per il simbolo {symbol}")
        
        return jsonify({'message': f'Dati eliminati con successo per {symbol}'})
    except Exception as e:
        logger.error(f"Errore nell'eliminazione dei dati per {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/market_data/<symbol>')
def market_data(symbol):
    """Visualizza i dati di mercato per un simbolo specifico"""
    try:
        # Prova a recuperare i dati dalla cache
        cache_key = f"market_data_{symbol}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            return jsonify(cached_data)
        
        # Se non in cache, carica dal file
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path):
            return jsonify({'error': f'Dati non disponibili per il simbolo {symbol}'}), 404
        
        # Carica i dati
        df = pd.read_csv(file_path)
        
        # Formatta i dati per la visualizzazione
        data = {
            'symbol': symbol,
            'dates': df['Date'].tolist() if 'Date' in df.columns else [],
            'open': df['Open'].tolist() if 'Open' in df.columns else [],
            'high': df['High'].tolist() if 'High' in df.columns else [],
            'low': df['Low'].tolist() if 'Low' in df.columns else [],
            'close': df['Close'].tolist() if 'Close' in df.columns else [],
            'volume': df['Volume'].tolist() if 'Volume' in df.columns else []
        }
        
        # Salva in cache per 15 minuti (900 secondi)
        cache_manager.set(cache_key, data, ttl=900)
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Errore nel recupero dei dati di mercato per {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/market_data_visualization/<symbol>')
def market_data_visualization(symbol):
    """Visualizza grafici avanzati per i dati di mercato"""
    try:
        # Verifica che il file esista
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path):
            flash(f'Dati per il simbolo {symbol} non trovati', 'error')
            return redirect(url_for('data_collection'))
        
        # Carica i dati
        df = pd.read_csv(file_path)
        
        # Converti la colonna Data in datetime e imposta come indice
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            # Prova a convertire l'indice
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
        
        # Prepara i dati aggiuntivi con indicatori tecnici
        if df is not None and 'Close' in df.columns:
            try:
                # Calcola medie mobili
                df['SMA_5'] = df['Close'].rolling(window=5).mean()
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # Calcola MACD
                df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
                
                # Calcola RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Calcola Bollinger Bands
                df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                df['BB_Std'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
                df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
                
                # Calcola la volatilità
                df['Returns'] = df['Close'].pct_change()
                df['Volatility_20'] = df['Returns'].rolling(window=20).std()
                
            except Exception as e:
                logger.error(f"Errore nel calcolo degli indicatori tecnici: {e}")
        
        # Crea i grafici usando il visualization_manager
        charts = {}
        
        # 1. Grafico a candele
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            charts['candlestick'] = visualization_manager.create_candlestick_chart(
                df, title=f"{symbol} - Grafico a Candele"
            )
        
        # 2. Grafico dei volumi
        if 'Volume' in df.columns:
            charts['volume'] = visualization_manager.create_volume_chart(
                df, title=f"{symbol} - Volume"
            )
        
        # 3. Grafico con medie mobili
        if 'Close' in df.columns and 'SMA_5' in df.columns:
            sma_columns = [col for col in df.columns if col.startswith('SMA_')]
            price_data = df[['Close'] + sma_columns].dropna()
            
            performance_data = pd.DataFrame()
            performance_data['Close'] = price_data['Close']
            for col in sma_columns:
                performance_data[col] = price_data[col]
            
            charts['moving_averages'] = visualization_manager.create_performance_chart(
                performance_data, title=f"{symbol} - Medie Mobili"
            )
        
        # 4. Grafico MACD
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            macd_data = df[['MACD', 'MACD_Signal', 'MACD_Hist']].dropna()
            
            charts['macd'] = visualization_manager.create_performance_chart(
                macd_data, title=f"{symbol} - MACD"
            )
        
        # 5. Grafico RSI
        if 'RSI' in df.columns:
            rsi_data = df[['RSI']].dropna()
            
            charts['rsi'] = visualization_manager.create_performance_chart(
                rsi_data, title=f"{symbol} - RSI"
            )
        
        # 6. Grafico Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            bb_data = df[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']].dropna()
            
            charts['bollinger'] = visualization_manager.create_performance_chart(
                bb_data, title=f"{symbol} - Bollinger Bands"
            )
        
        # Prepara alcune statistiche di base
        stats = {}
        if not df.empty and 'Close' in df.columns:
            try:
                stats['current_price'] = df['Close'].iloc[-1]
                stats['change_percent'] = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
                stats['max_price'] = df['Close'].max()
                stats['min_price'] = df['Close'].min()
                stats['avg_volume'] = df['Volume'].mean() if 'Volume' in df.columns else 'N/A'
                stats['period_start'] = df.index[0].strftime('%Y-%m-%d')
                stats['period_end'] = df.index[-1].strftime('%Y-%m-%d')
                stats['trading_days'] = len(df)
            except Exception as e:
                logger.error(f"Errore nel calcolo delle statistiche: {e}")
                stats = {}
        
        return render_template(
            'market_data_visualization.html',
            symbol=symbol,
            charts=charts,
            stats=stats,
            df_head=df.head().to_html(classes='table table-sm table-striped'),
            df_tail=df.tail().to_html(classes='table table-sm table-striped')
        )
    except Exception as e:
        logger.error(f"Errore nella visualizzazione dei dati di mercato: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('data_collection'))

@app.route('/user_guide')
def user_guide():
    """Mostra la guida utente"""
    try:
        return render_template('user_guide.html')
    except Exception as e:
        logger.error(f"Errore nel caricamento della guida utente: {e}")
        flash(f'Si è verificato un errore: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/cache/stats')
def cache_stats():
    """Visualizza le statistiche della cache"""
    try:
        stats = cache_manager.get_cache_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Errore nel recupero delle statistiche della cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Svuota la cache"""
    try:
        success = cache_manager.clear()
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Errore nello svuotamento della cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache')
def cache_view():
    """Pagina per visualizzare le statistiche della cache"""
    try:
        return render_template('cache_stats.html')
    except Exception as e:
        logger.error(f"Errore nel rendering della pagina di cache: {e}")
        return render_template('error.html', error=str(e))

@app.route('/correlation')
def correlation_view():
    """Pagina per l'analisi delle correlazioni tra titoli"""
    try:
        # Ottieni la lista dei simboli disponibili
        available_symbols = []
        if os.path.exists(DATA_DIR):
            for file_name in os.listdir(DATA_DIR):
                if file_name.endswith('.csv'):
                    symbol = file_name.replace('.csv', '')
                    available_symbols.append(symbol)
        
        # Ottieni i parametri dalla query string
        symbols = request.args.get('symbols', '').split(',')
        symbols = [s.strip() for s in symbols if s.strip()]
        
        # Date predefinite
        today = datetime.now()
        end_date = today.strftime('%Y-%m-%d')
        start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
        
        return render_template('correlation_view.html', 
                              available_symbols=available_symbols,
                              symbols=symbols,
                              start_date=start_date,
                              end_date=end_date)
    except Exception as e:
        logger.error(f"Errore nel rendering della pagina di correlazione: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/correlation')
def api_correlation():
    """API per l'analisi delle correlazioni tra titoli"""
    try:
        # Ottieni i parametri dalla query string
        symbols_str = request.args.get('symbols', '')
        symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Verifica che ci siano almeno 2 simboli
        if len(symbols) < 2:
            return jsonify({'error': 'Servono almeno 2 simboli per l\'analisi di correlazione'}), 400
        
        # Prova a recuperare i dati dalla cache
        cache_key = f"correlation_{symbols_str}_{start_date}_{end_date}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            return jsonify(cached_data)
        
        # Carica i dati per ciascun simbolo
        market_data = {}
        for symbol in symbols:
            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            if not os.path.exists(file_path):
                continue
            
            df = pd.read_csv(file_path)
            
            # Filtra per data
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            market_data[symbol] = {
                'dates': df['Date'].astype(str).tolist() if 'Date' in df.columns else [],
                'close': df['Close'].tolist() if 'Close' in df.columns else []
            }
        
        # Verifica che ci siano dati sufficienti
        valid_symbols = [s for s, data in market_data.items() if len(data.get('close', [])) > 0]
        if len(valid_symbols) < 2:
            return jsonify({'error': 'Dati insufficienti per l\'analisi di correlazione'}), 400
        
        # Crea la matrice di correlazione
        correlation_matrix = visualization_manager.create_correlation_matrix(
            market_data, 
            symbols=valid_symbols,
            title='Matrice di Correlazione'
        )
        
        # Crea la scatter matrix
        scatter_matrix = visualization_manager.create_scatter_matrix(
            market_data,
            symbols=valid_symbols,
            title='Scatter Matrix'
        )
        
        # Prepara i dati per il client
        result = {
            'correlation_matrix': correlation_matrix.to_dict() if correlation_matrix else None,
            'scatter_matrix': scatter_matrix.to_dict() if scatter_matrix else None,
            'symbols': valid_symbols,
            'period': {
                'start_date': start_date,
                'end_date': end_date
            }
        }
        
        # Salva in cache per 30 minuti (1800 secondi)
        cache_manager.set(cache_key, result, ttl=1800)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Errore nell'analisi delle correlazioni: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report/<report_id>/<format>')
def generate_report(report_id, format):
    """
    Genera ed esporta un report in PDF o Excel
    
    Args:
        report_id: ID del report da generare
        format: Formato del report (pdf o excel)
    """
    try:
        # Verifica che il formato sia supportato
        if format not in ['pdf', 'excel']:
            flash(f'Formato non supportato: {format}', 'error')
            return redirect(url_for('view_report', report_id=report_id))
        
        # Carica i dati della simulazione
        report_file = os.path.join(REPORTS_DIR, f"{report_id}.json")
        if not os.path.exists(report_file):
            flash(f'Report non trovato: {report_id}', 'error')
            return redirect(url_for('reports'))
        
        with open(report_file, 'r') as f:
            simulation_data = json.load(f)
        
        # Genera il report nel formato richiesto
        file_path = ""
        if format == 'pdf':
            file_path = report_generator.generate_pdf_report(simulation_data, report_id)
        elif format == 'excel':
            file_path = report_generator.generate_excel_report(simulation_data, report_id)
        
        if not file_path or not os.path.exists(file_path):
            flash(f'Errore nella generazione del report', 'error')
            return redirect(url_for('view_report', report_id=report_id))
        
        # Restituisci il file generato per il download
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Errore nella generazione del report: {e}")
        flash(f'Errore: {str(e)}', 'error')
        return redirect(url_for('view_report', report_id=report_id))

# WebSocket events
@websocket_manager.socketio.on('connect')
def handle_connect():
    """Gestisce la connessione WebSocket"""
    logger.info("Client WebSocket connesso")
    websocket_manager.emit_dashboard_state(state_manager.get_dashboard_state())

@websocket_manager.socketio.on('disconnect')
def handle_disconnect():
    """Gestisce la disconnessione WebSocket"""
    logger.info("Client WebSocket disconnesso")

@websocket_manager.socketio.on('subscribe')
def handle_subscribe(data):
    """Gestisce la sottoscrizione agli eventi"""
    tab_name = data.get('tab')
    if tab_name:
        websocket_manager.subscribe_client(request.sid, tab_name)
        websocket_manager.emit_tab_state(request.sid, tab_name, state_manager.get_tab_state(tab_name))

@websocket_manager.socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Gestisce l'annullamento della sottoscrizione agli eventi"""
    tab_name = data.get('tab')
    if tab_name:
        websocket_manager.unsubscribe_client(request.sid, tab_name)

@websocket_manager.socketio.on('start_monitoring')
def handle_start_monitoring(data):
    """Gestisce la richiesta di avvio del monitoraggio in tempo reale"""
    try:
        client_sid = request.sid
        logger.info(f"Client {client_sid} ha richiesto l'avvio del monitoraggio")
        
        update_interval = float(data.get('update_interval', 1.0))
        
        # Avvia il monitoraggio se c'è una simulazione in corso
        if hasattr(simulator, 'simulation_state') and simulator.simulation_state.get('status') == 'running':
            real_time_monitor.start_monitoring(update_interval)
            emit('monitoring_started', {'status': 'success'})
        else:
            emit('monitoring_started', {'status': 'error', 'message': 'Nessuna simulazione in corso'})
    except Exception as e:
        logger.error(f"Errore nell'avvio del monitoraggio: {e}")
        emit('monitoring_started', {'status': 'error', 'message': str(e)})

@websocket_manager.socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Gestisce la richiesta di interruzione del monitoraggio in tempo reale"""
    try:
        client_sid = request.sid
        logger.info(f"Client {client_sid} ha richiesto l'interruzione del monitoraggio")
        
        real_time_monitor.stop_monitoring()
        emit('monitoring_stopped', {'status': 'success'})
    except Exception as e:
        logger.error(f"Errore nell'interruzione del monitoraggio: {e}")
        emit('monitoring_stopped', {'status': 'error', 'message': str(e)})

# NUOVO ENDPOINT PER INTERROMPERE LA SIMULAZIONE
@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Endpoint per richiedere l'interruzione della simulazione corrente."""
    try:
        if simulator and hasattr(simulator, 'request_stop'):
            logger.info("Ricevuta richiesta API per interrompere la simulazione.")
            simulator.request_stop() # Chiama il metodo nel Simulation Manager
            return jsonify({'status': 'stop_requested', 'message': 'Richiesta di interruzione inviata.'})
        else:
            logger.warning("Ricevuta richiesta API stop ma il simulatore non è pronto o non supporta request_stop.")
            return jsonify({'status': 'error', 'message': 'Simulatore non disponibile o non supporta interruzione.'}), 400
    except Exception as e:
        logger.error(f"Errore nell'endpoint /stop_simulation: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Registra le route del wizard e delle previsioni
register_wizard_routes(app, CONFIG, state_manager, collector, get_available_strategies)
register_prediction_routes(app, CONFIG, collector, model_trainer, neural_network_integration)

# Aggiungi la route per l'addestramento delle reti neurali
@app.route('/neural_network')
def neural_network():
    """Redirect alla pagina di addestramento delle reti neurali"""
    return redirect(url_for('neural_network.neural_network_training'))

# Aggiungi alias per le route esistenti
@app.route('/training')
def training():
    """Alias per la route train_model"""
    return redirect(url_for('train_model'))

@app.route('/self_play')
def self_play():
    """Alias per la route self_play"""
    return render_template('self_play.html')

@app.route('/wizard')
def wizard_redirect():
    """Redirect alla prima pagina del wizard"""
    return redirect(url_for('wizard_configure_symbols'))

@app.route('/wizard/data_collection', methods=['GET', 'POST'])
def wizard_data_collection():
    """Interfaccia guidata per la raccolta dati"""
    # Utilizza la stessa logica di data_collection ma con un template diverso
    return render_template('wizard_data_collection.html', config=CONFIG)

@app.route('/wizard/run_simulation', methods=['GET', 'POST'])
def wizard_run_simulation():
    """Interfaccia guidata per l'esecuzione della simulazione"""
    # Ottieni le strategie disponibili
    strategies_info = get_available_strategies()
    return render_template('wizard_run_simulation.html', config=CONFIG, strategies_info=strategies_info)

# Route per supportare i vecchi url senza prefisso /wizard
@app.route('/wizard_data_collection', methods=['GET', 'POST'])
def wizard_data_collection_redirect():
    """Implementa direttamente l'interfaccia guidata per la raccolta dati"""
    # Utilizza la stessa logica di data_collection ma con un template diverso
    return render_template('wizard_data_collection.html', config=CONFIG)

@app.route('/wizard_run_simulation', methods=['GET', 'POST'])
def wizard_run_simulation_redirect():
    """Implementa direttamente l'interfaccia guidata per l'esecuzione della simulazione"""
    # Ottieni le strategie disponibili
    strategies_info = get_available_strategies()
    return render_template('wizard_run_simulation.html', config=CONFIG, strategies_info=strategies_info)

# Gestione errori generali dell'app
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Errore interno del server: {e}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Analizza gli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Dashboard del sistema di trading algoritmico')
    parser.add_argument('--port', type=int, default=8081, help='Porta su cui avviare la dashboard')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host su cui avviare la dashboard')
    parser.add_argument('--debug', action='store_true', help='Attiva la modalità debug')
    
    args = parser.parse_args()
    
    # Avvia la dashboard
    logger.info(f"Avvio della dashboard su {args.host}:{args.port} (debug: {args.debug})")
    websocket_manager.socketio.run(app, host=args.host, port=args.port, debug=args.debug)
