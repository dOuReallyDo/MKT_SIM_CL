"""
Neural Network Routes Module.

Questo modulo fornisce le route per l'addestramento e l'utilizzo di reti neurali.
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

from data.collector import DataCollector
from neural_network.model_trainer import ModelTrainer
from trading_strategy.neural_network_bridge import NeuralNetworkBridge

# Configura il logger
logger = logging.getLogger('dashboard.neural_network_routes')

# Crea il blueprint
neural_network_bp = Blueprint('neural_network', __name__)

# Istanze globali
data_collector = None
model_trainer = None
neural_network_bridge = None

# Stato dell'addestramento
training_status = {
    'active': False,
    'symbol': None,
    'model_type': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'current_test_loss': 0.0,
    'start_time': None,
    'stop_requested': False
}

def init_neural_network_routes(app_data_collector):
    """
    Inizializza le route per le reti neurali.
    
    Args:
        app_data_collector: Istanza del DataCollector
    """
    global data_collector, model_trainer, neural_network_bridge
    
    data_collector = app_data_collector
    model_trainer = ModelTrainer()
    neural_network_bridge = NeuralNetworkBridge(model_trainer, data_collector)
    
    logger.info("Route per le reti neurali inizializzate")

@neural_network_bp.route('/neural_network_training')
def neural_network_training():
    """
    Pagina per l'addestramento delle reti neurali.
    """
    # Ottieni i simboli disponibili
    available_symbols = data_collector.get_available_symbols() if data_collector else []
    
    # Date predefinite
    today = datetime.now()
    default_end_date = today.strftime('%Y-%m-%d')
    default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Ottieni i modelli disponibili
    available_models = _get_available_models()
    
    return render_template(
        'neural_network_training.html',
        available_symbols=available_symbols,
        default_start_date=default_start_date,
        default_end_date=default_end_date,
        available_models=available_models
    )

@neural_network_bp.route('/train_model', methods=['POST'])
def train_model():
    """
    Avvia l'addestramento di un modello.
    """
    global training_status
    
    # Verifica che non ci sia già un addestramento in corso
    if training_status['active']:
        return jsonify({
            'status': 'error',
            'error': 'Addestramento già in corso'
        })
    
    try:
        # Ottieni i parametri dalla richiesta
        data = request.json
        symbol = data.get('symbol')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        model_type = data.get('model_type', 'lstm')
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        
        # Verifica che i parametri siano validi
        if not symbol or not start_date or not end_date:
            return jsonify({
                'status': 'error',
                'error': 'Parametri mancanti'
            })
        
        # Aggiorna lo stato dell'addestramento
        training_status = {
            'active': True,
            'symbol': symbol,
            'model_type': model_type,
            'current_epoch': 0,
            'total_epochs': epochs,
            'current_loss': 0.0,
            'current_test_loss': 0.0,
            'start_time': datetime.now(),
            'stop_requested': False
        }
        
        # Avvia l'addestramento in un thread separato
        import threading
        thread = threading.Thread(
            target=_train_model_thread,
            args=(symbol, start_date, end_date, model_type, epochs, batch_size)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Addestramento avviato per {symbol}'
        })
    
    except Exception as e:
        logger.error(f"Errore nell'avvio dell'addestramento: {e}")
        logger.error(traceback.format_exc())
        
        # Resetta lo stato dell'addestramento
        training_status = {
            'active': False,
            'symbol': None,
            'model_type': None,
            'current_epoch': 0,
            'total_epochs': 0,
            'current_loss': 0.0,
            'current_test_loss': 0.0,
            'start_time': None,
            'stop_requested': False
        }
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@neural_network_bp.route('/stop_training', methods=['POST'])
def stop_training():
    """
    Interrompe l'addestramento in corso.
    """
    global training_status
    
    # Verifica che ci sia un addestramento in corso
    if not training_status['active']:
        return jsonify({
            'status': 'error',
            'message': 'Nessun addestramento in corso'
        })
    
    # Richiedi l'interruzione
    training_status['stop_requested'] = True
    
    return jsonify({
        'status': 'stop_requested',
        'message': 'Richiesta di interruzione inviata'
    })

@neural_network_bp.route('/training_status')
def get_training_status():
    """
    Restituisce lo stato dell'addestramento.
    """
    global training_status
    
    # Calcola il tempo rimanente
    remaining_time = None
    if training_status['active'] and training_status['current_epoch'] > 0:
        elapsed_time = (datetime.now() - training_status['start_time']).total_seconds()
        time_per_epoch = elapsed_time / training_status['current_epoch']
        remaining_epochs = training_status['total_epochs'] - training_status['current_epoch']
        remaining_seconds = time_per_epoch * remaining_epochs
        
        # Formatta il tempo rimanente
        if remaining_seconds < 60:
            remaining_time = f"{int(remaining_seconds)} sec"
        elif remaining_seconds < 3600:
            remaining_time = f"{int(remaining_seconds / 60)} min"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            remaining_time = f"{hours}h {minutes}m"
    
    return jsonify({
        'active': training_status['active'],
        'symbol': training_status['symbol'],
        'model_type': training_status['model_type'],
        'current_epoch': training_status['current_epoch'],
        'total_epochs': training_status['total_epochs'],
        'progress': int(training_status['current_epoch'] / training_status['total_epochs'] * 100) if training_status['total_epochs'] > 0 else 0,
        'current_loss': training_status['current_loss'],
        'current_test_loss': training_status['current_test_loss'],
        'remaining_time': remaining_time,
        'stop_requested': training_status['stop_requested']
    })

@neural_network_bp.route('/generate_predictions', methods=['POST'])
def generate_predictions():
    """
    Genera previsioni per un simbolo.
    """
    try:
        # Ottieni i parametri dalla richiesta
        data = request.json
        symbol = data.get('symbol')
        days = data.get('days', 5)
        
        # Verifica che i parametri siano validi
        if not symbol:
            return jsonify({
                'status': 'error',
                'error': 'Parametri mancanti'
            })
        
        # Genera le previsioni
        result = neural_network_bridge.generate_predictions(symbol, days)
        
        if not result:
            return jsonify({
                'status': 'error',
                'error': f'Impossibile generare previsioni per {symbol}'
            })
        
        # Ottieni i dati storici
        historical_df = data_collector.get_stock_data(symbol)
        
        if historical_df is None or historical_df.empty:
            return jsonify({
                'status': 'error',
                'error': f'Nessun dato storico disponibile per {symbol}'
            })
        
        # Prepara i dati per il frontend
        predictions_df = result['predictions']
        signals = result['signals']
        
        # Converti i dati storici in un formato serializzabile
        historical_dates = [date.strftime('%Y-%m-%d') for date in historical_df.index[-30:]]
        historical_prices = historical_df['Close'].values[-30:].tolist()
        
        # Converti le previsioni in un formato serializzabile
        prediction_dates = [date.strftime('%Y-%m-%d') for date in predictions_df.index]
        prediction_prices = predictions_df['Close'].values.tolist()
        
        return jsonify({
            'status': 'success',
            'historical': {
                'dates': historical_dates,
                'prices': historical_prices
            },
            'predictions': {
                'dates': prediction_dates,
                'prices': prediction_prices
            },
            'signals': signals
        })
    
    except Exception as e:
        logger.error(f"Errore nella generazione delle previsioni: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@neural_network_bp.route('/use_model_in_simulation', methods=['POST'])
def use_model_in_simulation():
    """
    Configura un modello per l'utilizzo in una simulazione.
    """
    try:
        # Ottieni i parametri dalla richiesta
        data = request.json
        symbol = data.get('symbol')
        sequence_length = data.get('sequence_length', 10)
        threshold = data.get('threshold', 0.01)
        
        # Verifica che i parametri siano validi
        if not symbol:
            return jsonify({
                'status': 'error',
                'error': 'Parametri mancanti'
            })
        
        # Crea la strategia
        strategy = neural_network_bridge.create_strategy(symbol, sequence_length, threshold)
        
        if not strategy:
            return jsonify({
                'status': 'error',
                'error': f'Impossibile creare la strategia per {symbol}'
            })
        
        # Salva la strategia nella sessione
        session['neural_network_strategy'] = {
            'symbol': symbol,
            'sequence_length': sequence_length,
            'threshold': threshold
        }
        
        return jsonify({
            'status': 'success',
            'message': f'Strategia creata per {symbol}'
        })
    
    except Exception as e:
        logger.error(f"Errore nella configurazione del modello: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

def _train_model_thread(symbol, start_date, end_date, model_type, epochs, batch_size):
    """
    Thread per l'addestramento del modello.
    
    Args:
        symbol: Simbolo dell'asset
        start_date: Data di inizio
        end_date: Data di fine
        model_type: Tipo di modello
        epochs: Numero di epoche
        batch_size: Dimensione del batch
    """
    global training_status
    
    try:
        logger.info(f"Avvio addestramento per {symbol} ({model_type})")
        
        # Configura il model trainer con callback per aggiornare lo stato
        config = {
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'sequence_length': 10,
            'callbacks': {
                'on_epoch_end': _on_epoch_end_callback
            }
        }
        
        # Crea un nuovo model trainer
        model_trainer = ModelTrainer(model_type=model_type, config=config)
        
        # Addestra il modello
        success = neural_network_bridge.train_model(
            symbol, start_date, end_date, model_type, epochs, batch_size
        )
        
        # Aggiorna lo stato dell'addestramento
        training_status['active'] = False
        
        logger.info(f"Addestramento completato per {symbol}: {success}")
    
    except Exception as e:
        logger.error(f"Errore nell'addestramento: {e}")
        logger.error(traceback.format_exc())
        
        # Aggiorna lo stato dell'addestramento
        training_status['active'] = False

def _on_epoch_end_callback(epoch, logs):
    """
    Callback chiamato alla fine di ogni epoca.
    
    Args:
        epoch: Numero dell'epoca
        logs: Log dell'epoca
    """
    global training_status
    
    # Aggiorna lo stato dell'addestramento
    training_status['current_epoch'] = epoch + 1
    training_status['current_loss'] = logs.get('loss', 0.0)
    training_status['current_test_loss'] = logs.get('val_loss', 0.0)
    
    # Verifica se è stata richiesta l'interruzione
    return not training_status['stop_requested']

def _get_available_models():
    """
    Restituisce la lista dei modelli disponibili.
    
    Returns:
        list: Lista dei modelli disponibili
    """
    models = []
    
    try:
        # Directory dei modelli
        models_dir = './models'
        
        if not os.path.exists(models_dir):
            return models
        
        # Trova tutti i file dei modelli
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        
        for file in model_files:
            try:
                # Estrai le informazioni dal nome del file
                parts = file.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    model_type = parts[1]
                    
                    # Estrai la data di creazione
                    date_str = parts[2].split('.')[0]
                    created_at = datetime.strptime(date_str, '%Y%m%d%H%M%S').strftime('%d/%m/%Y %H:%M')
                    
                    # Carica le metriche se disponibili
                    metrics_file = os.path.join(models_dir, f"{symbol}_{model_type}_{date_str}_metrics.json")
                    mse = 0.0
                    rmse = 0.0
                    
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            mse = metrics.get('mse', 0.0)
                            rmse = metrics.get('rmse', 0.0)
                    
                    models.append({
                        'symbol': symbol,
                        'type': model_type,
                        'created_at': created_at,
                        'mse': f"{mse:.6f}",
                        'rmse': f"{rmse:.6f}"
                    })
            
            except Exception as e:
                logger.error(f"Errore nell'elaborazione del file {file}: {e}")
        
        # Ordina i modelli per data di creazione (più recente prima)
        models.sort(key=lambda x: x['created_at'], reverse=True)
    
    except Exception as e:
        logger.error(f"Errore nel recupero dei modelli disponibili: {e}")
    
    return models
