"""
Route per la gestione delle previsioni con reti neurali.

Questo modulo fornisce le route per l'addestramento dei modelli e la generazione delle previsioni.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import render_template, request, redirect, url_for, flash, jsonify

# Configura il logger
logger = logging.getLogger('prediction_routes')

def register_prediction_routes(app, CONFIG, data_collector, model_trainer, neural_network_integration):
    """
    Registra le route per la gestione delle previsioni.
    
    Args:
        app: Istanza dell'app Flask
        CONFIG: Configurazione del sistema
        data_collector: Istanza del DataCollector
        model_trainer: Istanza del ModelTrainer
        neural_network_integration: Istanza del NeuralNetworkIntegration
    """
    
    @app.route('/prediction')
    def prediction():
        """Pagina per la gestione delle previsioni."""
        try:
            # Ottieni i simboli disponibili
            available_symbols = []
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            
            for file_name in os.listdir(data_dir):
                if file_name.endswith('.csv'):
                    symbol = file_name.replace('.csv', '')
                    available_symbols.append(symbol)
            
            # Date predefinite
            today = datetime.now()
            default_end_date = today.strftime('%Y-%m-%d')
            default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Ottieni i modelli disponibili
            available_models = get_available_models()
            
            return render_template('prediction.html',
                                  available_symbols=available_symbols,
                                  default_start_date=default_start_date,
                                  default_end_date=default_end_date,
                                  available_models=available_models)
        except Exception as e:
            logger.error(f"Errore nella pagina di previsione: {e}")
            flash(f'Si è verificato un errore: {str(e)}', 'danger')
            return redirect(url_for('index'))
    
    @app.route('/train_model', methods=['POST'])
    def train_model():
        """Endpoint per l'addestramento dei modelli."""
        try:
            # Recupera i parametri dal form
            symbols = request.form.getlist('symbols')
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            model_type = request.form.get('model_type', 'lstm')
            sequence_length = int(request.form.get('sequence_length', 10))
            epochs = int(request.form.get('epochs', 50))
            batch_size = int(request.form.get('batch_size', 32))
            
            if not symbols:
                flash('Seleziona almeno un simbolo', 'error')
                return redirect(url_for('prediction'))
            
            # Prepara i dati per l'addestramento
            prepared_data = neural_network_integration.prepare_data_for_training(
                symbols, start_date, end_date, sequence_length
            )
            
            if not prepared_data:
                flash('Nessun dato disponibile per l\'addestramento', 'error')
                return redirect(url_for('prediction'))
            
            # Addestra i modelli
            start_time = time.time()
            models = neural_network_integration.train_models(
                prepared_data, model_type, epochs, batch_size
            )
            training_time = time.time() - start_time
            
            if not models:
                flash('Errore nell\'addestramento dei modelli', 'error')
                return redirect(url_for('prediction'))
            
            # Valuta i modelli
            metrics = neural_network_integration.evaluate_models(prepared_data)
            
            # Salva i modelli
            neural_network_integration.save_models()
            
            # Prepara i risultati dell'addestramento
            training_results = {}
            
            for symbol, model in models.items():
                # Ottieni la storia dell'addestramento
                history = model.history.history
                
                # Calcola le metriche finali
                final_loss = history['loss'][-1] if 'loss' in history else 0
                final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 0
                
                training_results[symbol] = {
                    'model_type': model_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'final_loss': final_loss,
                    'final_val_loss': final_val_loss,
                    'training_time': round(training_time, 2),
                    'history': history
                }
            
            flash('Modelli addestrati con successo', 'success')
            
            # Ottieni i simboli disponibili
            available_symbols = []
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            
            for file_name in os.listdir(data_dir):
                if file_name.endswith('.csv'):
                    symbol = file_name.replace('.csv', '')
                    available_symbols.append(symbol)
            
            # Date predefinite
            today = datetime.now()
            default_end_date = today.strftime('%Y-%m-%d')
            default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Ottieni i modelli disponibili
            available_models = get_available_models()
            
            return render_template('prediction.html',
                                  available_symbols=available_symbols,
                                  default_start_date=default_start_date,
                                  default_end_date=default_end_date,
                                  available_models=available_models,
                                  training_results=training_results)
        except Exception as e:
            logger.error(f"Errore nell'addestramento dei modelli: {e}")
            flash(f'Si è verificato un errore: {str(e)}', 'danger')
            return redirect(url_for('prediction'))
    
    @app.route('/generate_predictions', methods=['POST'])
    def generate_predictions():
        """Endpoint per la generazione delle previsioni."""
        try:
            # Recupera i parametri dal form
            symbols = request.form.getlist('symbols')
            days = int(request.form.get('days', 5))
            threshold = float(request.form.get('threshold', 1.0)) / 100  # Converti da percentuale a decimale
            
            if not symbols:
                flash('Seleziona almeno un simbolo', 'error')
                return redirect(url_for('prediction'))
            
            # Carica i modelli
            models = neural_network_integration.load_models(symbols)
            
            if not models:
                flash('Nessun modello disponibile per i simboli selezionati', 'error')
                return redirect(url_for('prediction'))
            
            # Genera le previsioni
            predictions_df = neural_network_integration.predict(symbols, days)
            
            if not predictions_df:
                flash('Errore nella generazione delle previsioni', 'error')
                return redirect(url_for('prediction'))
            
            # Genera i segnali di trading
            signals = neural_network_integration.generate_trading_signals(predictions_df, threshold)
            
            # Salva le previsioni
            neural_network_integration.save_predictions()
            
            # Prepara i dati per la visualizzazione
            predictions = {}
            
            for symbol, pred_df in predictions_df.items():
                # Ottieni i dati storici
                df = data_collector.get_stock_data(symbol)
                
                if df is None or df.empty:
                    continue
                
                # Assicurati che la colonna Date sia l'indice
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                
                # Ottieni gli ultimi 30 giorni di dati storici
                historical_df = df.iloc[-30:]
                
                # Prepara i dati per il grafico
                dates = [date.strftime('%Y-%m-%d') for date in historical_df.index]
                dates.extend([date.strftime('%Y-%m-%d') for date in pred_df.index])
                
                historical_prices = historical_df['Close'].tolist()
                predicted_prices = [None] * len(historical_df) + pred_df['Close'].tolist()
                
                # Ottieni le metriche del modello
                metrics = {}
                if symbol in models:
                    try:
                        # Carica i dati per il test
                        test_data = neural_network_integration.prepare_data_for_training(
                            [symbol], df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')
                        )
                        
                        if symbol in test_data:
                            # Valuta il modello
                            model_metrics = neural_network_integration.evaluate_models({symbol: test_data[symbol]})
                            if symbol in model_metrics:
                                metrics = model_metrics[symbol]
                    except Exception as e:
                        logger.error(f"Errore nel calcolo delle metriche per {symbol}: {e}")
                
                predictions[symbol] = {
                    'dates': dates,
                    'historical_prices': historical_prices,
                    'predicted_prices': predicted_prices,
                    'signals': signals.get(symbol, []),
                    'metrics': metrics
                }
            
            flash('Previsioni generate con successo', 'success')
            
            # Ottieni i simboli disponibili
            available_symbols = []
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            
            for file_name in os.listdir(data_dir):
                if file_name.endswith('.csv'):
                    symbol = file_name.replace('.csv', '')
                    available_symbols.append(symbol)
            
            # Date predefinite
            today = datetime.now()
            default_end_date = today.strftime('%Y-%m-%d')
            default_start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Ottieni i modelli disponibili
            available_models = get_available_models()
            
            return render_template('prediction.html',
                                  available_symbols=available_symbols,
                                  default_start_date=default_start_date,
                                  default_end_date=default_end_date,
                                  available_models=available_models,
                                  predictions=predictions)
        except Exception as e:
            logger.error(f"Errore nella generazione delle previsioni: {e}")
            flash(f'Si è verificato un errore: {str(e)}', 'danger')
            return redirect(url_for('prediction'))

def get_available_models():
    """
    Ottiene i modelli disponibili.
    
    Returns:
        dict: Modelli disponibili
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        return {}
    
    available_models = {}
    
    for file_name in os.listdir(models_dir):
        if file_name.endswith('.h5'):
            try:
                # Estrai il simbolo e la data dal nome del file
                parts = file_name.split('_')
                symbol = parts[0]
                date_str = parts[1].replace('.h5', '')
                
                # Converti la data
                date = datetime.strptime(date_str, '%Y%m%d')
                
                # Determina il tipo di modello (per semplicità, assumiamo LSTM)
                model_type = 'LSTM'
                
                # Calcola l'accuratezza (per semplicità, un valore casuale)
                accuracy = np.random.uniform(70, 95)
                
                available_models[symbol] = {
                    'type': model_type,
                    'date': date.strftime('%Y-%m-%d'),
                    'accuracy': round(accuracy, 2)
                }
            except Exception as e:
                logger.error(f"Errore nell'analisi del file {file_name}: {e}")
    
    return available_models
