"""
Modulo di integrazione tra la simulazione e le reti neurali.

Questo modulo fornisce funzioni per integrare le reti neurali con la simulazione di mercato.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configura il logger
logger = logging.getLogger('neural_network.integration')

class NeuralNetworkIntegration:
    """
    Classe per l'integrazione tra la simulazione e le reti neurali.
    """
    
    def __init__(self, model_trainer, data_collector):
        """
        Inizializza l'integrazione.
        
        Args:
            model_trainer: Istanza del ModelTrainer
            data_collector: Istanza del DataCollector
        """
        self.model_trainer = model_trainer
        self.data_collector = data_collector
        self.models = {}
        self.predictions = {}
        
    def prepare_data_for_training(self, symbols, start_date, end_date, sequence_length=10, test_size=0.2):
        """
        Prepara i dati per l'addestramento dei modelli.
        
        Args:
            symbols: Lista di simboli
            start_date: Data di inizio
            end_date: Data di fine
            sequence_length: Lunghezza della sequenza per i modelli RNN
            test_size: Percentuale di dati da utilizzare per il test
            
        Returns:
            dict: Dati preparati per l'addestramento
        """
        logger.info(f"Preparazione dei dati per l'addestramento dei modelli: {symbols}")
        
        prepared_data = {}
        
        for symbol in symbols:
            try:
                # Ottieni i dati dal collector
                df = self.data_collector.get_stock_data(symbol, start_date, end_date)
                
                if df is None or df.empty:
                    logger.warning(f"Nessun dato disponibile per {symbol}")
                    continue
                
                # Assicurati che la colonna Date sia l'indice
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                
                # Calcola feature aggiuntive
                df = self._calculate_features(df)
                
                # Normalizza i dati
                df_norm, scaler = self._normalize_data(df)
                
                # Crea sequenze per l'addestramento
                X, y = self._create_sequences(df_norm, sequence_length)
                
                # Dividi in training e test
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                prepared_data[symbol] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'scaler': scaler,
                    'df': df,
                    'df_norm': df_norm
                }
                
                logger.info(f"Dati preparati per {symbol}: {len(X_train)} sequenze di training, {len(X_test)} sequenze di test")
            
            except Exception as e:
                logger.error(f"Errore nella preparazione dei dati per {symbol}: {e}")
        
        return prepared_data
    
    def _calculate_features(self, df):
        """
        Calcola feature aggiuntive per l'addestramento.
        
        Args:
            df: DataFrame con i dati
            
        Returns:
            DataFrame: DataFrame con le feature aggiuntive
        """
        # Copia il DataFrame per evitare modifiche indesiderate
        df = df.copy()
        
        # Calcola i rendimenti
        df['Returns'] = df['Close'].pct_change()
        
        # Calcola medie mobili
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Calcola la volatilità
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Calcola il momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        # Calcola RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Rimuovi le righe con NaN
        df.dropna(inplace=True)
        
        return df
    
    def _normalize_data(self, df):
        """
        Normalizza i dati.
        
        Args:
            df: DataFrame con i dati
            
        Returns:
            tuple: DataFrame normalizzato e scaler
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Crea lo scaler
        scaler = MinMaxScaler()
        
        # Normalizza i dati
        df_norm = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return df_norm, scaler
    
    def _create_sequences(self, df, sequence_length):
        """
        Crea sequenze per l'addestramento.
        
        Args:
            df: DataFrame normalizzato
            sequence_length: Lunghezza della sequenza
            
        Returns:
            tuple: X e y per l'addestramento
        """
        X = []
        y = []
        
        # Ottieni i valori come array
        data = df.values
        
        # Crea le sequenze
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length, 0])  # Predici il prezzo di chiusura
        
        return np.array(X), np.array(y)
    
    def train_models(self, prepared_data, model_type='lstm', epochs=50, batch_size=32):
        """
        Addestra i modelli per i simboli.
        
        Args:
            prepared_data: Dati preparati per l'addestramento
            model_type: Tipo di modello ('lstm', 'cnn', 'transformer')
            epochs: Numero di epoche
            batch_size: Dimensione del batch
            
        Returns:
            dict: Modelli addestrati
        """
        logger.info(f"Addestramento dei modelli di tipo {model_type}")
        
        for symbol, data in prepared_data.items():
            try:
                logger.info(f"Addestramento del modello per {symbol}")
                
                # Ottieni i dati di addestramento
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']
                
                # Addestra il modello
                model = self.model_trainer.train_model(
                    X_train, y_train, X_test, y_test,
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                # Salva il modello
                self.models[symbol] = model
                
                logger.info(f"Modello per {symbol} addestrato con successo")
            
            except Exception as e:
                logger.error(f"Errore nell'addestramento del modello per {symbol}: {e}")
        
        return self.models
    
    def save_models(self, output_dir='models'):
        """
        Salva i modelli addestrati.
        
        Args:
            output_dir: Directory di output
            
        Returns:
            bool: True se i modelli sono stati salvati con successo
        """
        logger.info(f"Salvataggio dei modelli in {output_dir}")
        
        # Assicurati che la directory esista
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, model in self.models.items():
            try:
                # Crea il percorso del file
                model_path = os.path.join(output_dir, f"{symbol}_{datetime.now().strftime('%Y%m%d')}.h5")
                
                # Salva il modello
                model.save(model_path)
                
                logger.info(f"Modello per {symbol} salvato in {model_path}")
            
            except Exception as e:
                logger.error(f"Errore nel salvataggio del modello per {symbol}: {e}")
                return False
        
        return True
    
    def load_models(self, symbols, input_dir='models'):
        """
        Carica i modelli salvati.
        
        Args:
            symbols: Lista di simboli
            input_dir: Directory di input
            
        Returns:
            dict: Modelli caricati
        """
        logger.info(f"Caricamento dei modelli da {input_dir}")
        
        from tensorflow.keras.models import load_model
        
        for symbol in symbols:
            try:
                # Trova il modello più recente per il simbolo
                model_files = [f for f in os.listdir(input_dir) if f.startswith(f"{symbol}_") and f.endswith('.h5')]
                
                if not model_files:
                    logger.warning(f"Nessun modello trovato per {symbol}")
                    continue
                
                # Ordina per data (più recente prima)
                model_files.sort(reverse=True)
                
                # Carica il modello
                model_path = os.path.join(input_dir, model_files[0])
                model = load_model(model_path)
                
                # Salva il modello
                self.models[symbol] = model
                
                logger.info(f"Modello per {symbol} caricato da {model_path}")
            
            except Exception as e:
                logger.error(f"Errore nel caricamento del modello per {symbol}: {e}")
        
        return self.models
    
    def predict(self, symbols, days=5, sequence_length=10):
        """
        Genera previsioni per i simboli.
        
        Args:
            symbols: Lista di simboli
            days: Numero di giorni da prevedere
            sequence_length: Lunghezza della sequenza
            
        Returns:
            dict: Previsioni per i simboli
        """
        logger.info(f"Generazione delle previsioni per {symbols} ({days} giorni)")
        
        predictions = {}
        
        for symbol in symbols:
            try:
                # Verifica che il modello sia stato addestrato
                if symbol not in self.models:
                    logger.warning(f"Nessun modello disponibile per {symbol}")
                    continue
                
                # Ottieni i dati più recenti
                df = self.data_collector.get_stock_data(symbol)
                
                if df is None or df.empty:
                    logger.warning(f"Nessun dato disponibile per {symbol}")
                    continue
                
                # Assicurati che la colonna Date sia l'indice
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                
                # Calcola feature aggiuntive
                df = self._calculate_features(df)
                
                # Normalizza i dati
                df_norm, scaler = self._normalize_data(df)
                
                # Ottieni l'ultima sequenza
                last_sequence = df_norm.values[-sequence_length:]
                
                # Genera le previsioni
                model = self.models[symbol]
                
                # Prepara le date future
                last_date = df.index[-1]
                future_dates = []
                
                for i in range(1, days + 1):
                    future_date = last_date + timedelta(days=i)
                    # Salta i weekend
                    while future_date.weekday() >= 5:  # 5 = Sabato, 6 = Domenica
                        future_date += timedelta(days=1)
                    future_dates.append(future_date)
                
                # Genera le previsioni
                predicted_values = []
                current_sequence = last_sequence.copy()
                
                for _ in range(days):
                    # Reshape per il modello
                    X = current_sequence.reshape(1, sequence_length, df_norm.shape[1])
                    
                    # Genera la previsione
                    prediction = model.predict(X)[0]
                    
                    # Aggiungi la previsione alla sequenza
                    new_row = current_sequence[-1].copy()
                    new_row[0] = prediction  # Sostituisci il prezzo di chiusura
                    
                    # Aggiorna la sequenza
                    current_sequence = np.vstack([current_sequence[1:], new_row])
                    
                    # Salva la previsione
                    predicted_values.append(prediction)
                
                # Denormalizza le previsioni
                close_idx = df.columns.get_loc('Close')
                predicted_closes = scaler.inverse_transform(np.array([predicted_values] * df.shape[1]).T)[:, close_idx]
                
                # Crea il DataFrame delle previsioni
                predictions_df = pd.DataFrame({
                    'Date': future_dates,
                    'Close': predicted_closes
                })
                predictions_df.set_index('Date', inplace=True)
                
                # Salva le previsioni
                predictions[symbol] = predictions_df
                
                logger.info(f"Previsioni generate per {symbol}: {len(predictions_df)} giorni")
            
            except Exception as prediction_error:
                logger.error(f"Errore nella generazione delle previsioni per {symbol}: {prediction_error}")
                # Continua con il prossimo simbolo
        
        # Salva le previsioni
        self.predictions = predictions
        
        return predictions
    
    def evaluate_models(self, prepared_data):
        """
        Valuta le performance dei modelli.
        
        Args:
            prepared_data: Dati preparati per l'addestramento
            
        Returns:
            dict: Metriche di valutazione
        """
        logger.info("Valutazione delle performance dei modelli")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {}
        
        for symbol, data in prepared_data.items():
            try:
                # Verifica che il modello sia stato addestrato
                if symbol not in self.models:
                    logger.warning(f"Nessun modello disponibile per {symbol}")
                    continue
                
                # Ottieni i dati di test
                X_test = data['X_test']
                y_test = data['y_test']
                
                # Genera le previsioni
                model = self.models[symbol]
                y_pred = model.predict(X_test)
                
                # Calcola le metriche
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Salva le metriche
                metrics[symbol] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                logger.info(f"Metriche per {symbol}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
            except Exception as e:
                logger.error(f"Errore nella valutazione del modello per {symbol}: {e}")
        
        return metrics
    
    def generate_trading_signals(self, predictions, threshold=0.01):
        """
        Genera segnali di trading basati sulle previsioni.
        
        Args:
            predictions: Previsioni generate
            threshold: Soglia per i segnali (percentuale)
            
        Returns:
            dict: Segnali di trading
        """
        logger.info(f"Generazione dei segnali di trading (soglia: {threshold})")
        
        signals = {}
        
        for symbol, pred_df in predictions.items():
            try:
                # Ottieni i dati storici
                df = self.data_collector.get_stock_data(symbol)
                
                if df is None or df.empty:
                    logger.warning(f"Nessun dato disponibile per {symbol}")
                    continue
                
                # Assicurati che la colonna Date sia l'indice
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                
                # Ottieni l'ultimo prezzo di chiusura
                last_close = df['Close'].iloc[-1]
                
                # Calcola le variazioni percentuali
                pred_df['Change'] = (pred_df['Close'] - last_close) / last_close
                
                # Genera i segnali
                signals[symbol] = []
                
                for date, row in pred_df.iterrows():
                    change = row['Change']
                    
                    if change > threshold:
                        signal = 'BUY'
                    elif change < -threshold:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    signals[symbol].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'predicted_close': row['Close'],
                        'change': change,
                        'signal': signal
                    })
                
                logger.info(f"Segnali generati per {symbol}: {len(signals[symbol])} giorni")
            
            except Exception as e:
                logger.error(f"Errore nella generazione dei segnali per {symbol}: {e}")
        
        return signals
    
    def save_predictions(self, output_dir='predictions'):
        """
        Salva le previsioni generate.
        
        Args:
            output_dir: Directory di output
            
        Returns:
            bool: True se le previsioni sono state salvate con successo
        """
        logger.info(f"Salvataggio delle previsioni in {output_dir}")
        
        # Assicurati che la directory esista
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Crea il percorso del file
            predictions_path = os.path.join(output_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Converti le previsioni in un formato serializzabile
            serializable_predictions = {}
            
            for symbol, pred_df in self.predictions.items():
                serializable_predictions[symbol] = {
                    'dates': [date.strftime('%Y-%m-%d') for date in pred_df.index],
                    'closes': pred_df['Close'].tolist()
                }
            
            # Salva le previsioni
            with open(predictions_path, 'w') as f:
                json.dump(serializable_predictions, f, indent=4)
            
            logger.info(f"Previsioni salvate in {predictions_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Errore nel salvataggio delle previsioni: {e}")
            return False
    
    def load_predictions(self, input_path):
        """
        Carica le previsioni salvate.
        
        Args:
            input_path: Percorso del file
            
        Returns:
            dict: Previsioni caricate
        """
        logger.info(f"Caricamento delle previsioni da {input_path}")
        
        try:
            # Carica le previsioni
            with open(input_path, 'r') as f:
                serializable_predictions = json.load(f)
            
            # Converti le previsioni in DataFrame
            predictions = {}
            
            for symbol, pred_data in serializable_predictions.items():
                dates = [datetime.strptime(date, '%Y-%m-%d') for date in pred_data['dates']]
                closes = pred_data['closes']
                
                predictions[symbol] = pd.DataFrame({
                    'Close': closes
                }, index=dates)
            
            # Salva le previsioni
            self.predictions = predictions
            
            logger.info(f"Previsioni caricate da {input_path}")
            
            return predictions
        
        except Exception as e:
            logger.error(f"Errore nel caricamento delle previsioni: {e}")
            return {}
