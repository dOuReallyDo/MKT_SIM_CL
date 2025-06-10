"""
Modulo di integrazione tra la simulazione e le reti neurali.

Questo modulo fornisce funzioni per integrare le reti neurali con la simulazione di mercato.
"""

import os
import sys
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch # Added import
from datetime import datetime, timedelta
from typing import Optional # Added for type hinting

# Import ModelTrainer
from neural_network.model_trainer import ModelTrainer 

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
        self.model_trainer = model_trainer # This might be the initial trainer instance passed in
        self.data_collector = data_collector
        self.models = {} # Dictionary to store loaded ModelTrainer instances per symbol
        self.predictions = {}
        
    def _find_model_path(self, symbol: str, input_dir: str = 'models') -> Optional[str]:
        """
        Trova il percorso del modello .pt più recente per un simbolo.
        Helper function adapted from NeuralNetworkBridge.
        
        Args:
            symbol: Simbolo dell'asset
            input_dir: Directory dove cercare i modelli
            
        Returns:
            Percorso del modello o None se non trovato
        """
        try:
            if not os.path.exists(input_dir):
                logger.warning(f"Directory dei modelli non trovata: {input_dir}")
                return None
            
            # Trova tutti i file dei modelli .pt per il simbolo
            model_files = [
                f for f in os.listdir(input_dir)
                if f.startswith(f"{symbol}_") and f.endswith('.pt')
            ]
            
            if not model_files:
                return None
            
            # Ordina per data (più recente prima, basato sul timestamp nel nome)
            # Assumendo formato nome file: {symbol}_{type}_{timestamp}.pt
            def get_timestamp(filename):
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # Estrai timestamp YYYYMMDDHHMMSS da es. lstm_model_20231027_103000.pt
                        ts_str = parts[-1].split('.')[0] 
                        # Gestisce sia YYYYMMDD_HHMMSS che YYYYMMDDHHMMSS
                        if len(ts_str) == 15 and ts_str[8] == '_': # YYYYMMDD_HHMMSS
                             return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                        elif len(ts_str) == 14: # YYYYMMDDHHMMSS
                             return datetime.strptime(ts_str, '%Y%m%d%H%M%S')
                    return datetime.min # Ritorna data minima se il formato non è valido
                except:
                    return datetime.min

            model_files.sort(key=get_timestamp, reverse=True)
            
            # Restituisci il percorso del modello più recente
            return os.path.join(input_dir, model_files[0])
        
        except Exception as e:
            logger.error(f"Errore nella ricerca del modello per {symbol}: {e}")
            return None

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
    
    # Removed train_models function as it duplicates/conflicts with NeuralNetworkBridge logic
    
    def save_models(self, output_dir='models'):
        """
        Salva i modelli addestrati.
        
        Args:
            output_dir: Directory di output
            
        Returns:
            bool: True se i modelli sono stati salvati con successo
        """
        """
        Salva i modelli addestrati usando il ModelTrainer associato.
        Nota: Questo assume che self.models contenga istanze di ModelTrainer.
        """
        logger.info(f"Salvataggio dei modelli PyTorch in {output_dir}")
        
        all_saved = True
        for symbol, model_trainer_instance in self.models.items():
            if isinstance(model_trainer_instance, ModelTrainer) and model_trainer_instance.model is not None:
                try:
                    # Usa il metodo interno del trainer per salvare
                    # Nota: _save_model in ModelTrainer non prende argomenti e genera il nome file
                    model_trainer_instance._save_model() 
                    logger.info(f"Modello per {symbol} salvato tramite ModelTrainer.")
                except Exception as e:
                    logger.error(f"Errore nel salvataggio del modello per {symbol} tramite ModelTrainer: {e}")
                    all_saved = False
            else:
                logger.warning(f"Nessun ModelTrainer valido o modello addestrato trovato per {symbol} in self.models. Impossibile salvare.")
                all_saved = False
        
        return all_saved
    
    def load_models(self, symbols, input_dir='models'):
        """
        Carica i modelli salvati.
        
        Args:
            symbols: Lista di simboli
            input_dir: Directory di input
            
        Returns:
            dict: Modelli caricati
        """
        logger.info(f"Caricamento dei modelli PyTorch da {input_dir}")
        
        # Utilizza il ModelTrainer per caricare i modelli
        if not hasattr(self, 'model_trainer') or self.model_trainer is None:
            logger.error("ModelTrainer non inizializzato in NeuralNetworkIntegration")
            return {}
            
        loaded_models = {}
        for symbol in symbols:
            try:
                # Trova il modello .pt più recente per il simbolo
                model_files = [f for f in os.listdir(input_dir) if f.startswith(f"{symbol}_") and f.endswith('.pt')]
                
                if not model_files:
                    logger.warning(f"Nessun modello .pt trovato per {symbol} in {input_dir}")
                    continue
                
                # Ordina per data (più recente prima, basato sul timestamp nel nome)
                model_files.sort(reverse=True)
                
                # Carica il modello utilizzando ModelTrainer
                model_path = os.path.join(input_dir, model_files[0])
                
                # Creiamo un trainer temporaneo per caricare il modello
                # Nota: Questo potrebbe essere migliorato se ModelTrainer potesse caricare senza addestrare prima
                temp_trainer = ModelTrainer() 
                success = temp_trainer.load_model(model_path)
                
                if success:
                    # Salviamo il trainer (che contiene il modello caricato e la configurazione)
                    # Usiamo il simbolo come chiave per coerenza con il resto della classe
                    loaded_models[symbol] = temp_trainer 
                    self.models[symbol] = temp_trainer # Aggiorna anche l'attributo della classe
                    logger.info(f"Modello per {symbol} caricato da {model_path} usando ModelTrainer")
                else:
                    logger.error(f"Errore nel caricamento del modello {model_path} usando ModelTrainer")

            except Exception as e:
                logger.error(f"Errore nel caricamento del modello per {symbol}: {e}")
        
        return loaded_models # Restituisce i trainer caricati
    
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
        
        # Utilizza il ModelTrainer per caricare i modelli
        # Import spostato all'inizio del file
            
        loaded_models = {}
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
                
                # Ottieni il ModelTrainer caricato per il simbolo
                # self.models[symbol] ora contiene un'istanza di ModelTrainer
                model_trainer = self.models.get(symbol)
                if not model_trainer or not model_trainer.model:
                    # Se il modello non è stato caricato correttamente o non è presente nel trainer
                    # Prova a caricarlo di nuovo (potrebbe essere necessario se l'app è stata riavviata)
                    model_path = self._find_model_path(symbol, input_dir='models') # Assumendo che _find_model_path sia aggiornato per .pt
                    if model_path:
                        temp_trainer = ModelTrainer()
                        if temp_trainer.load_model(model_path):
                             # Assicurati che il modello sia effettivamente caricato nello state_dict
                             if hasattr(temp_trainer, 'model_state_dict'):
                                 # Ricrea il modello con la dimensione corretta prima di caricare lo state_dict
                                 # Calcola input_dim dai dati normalizzati
                                 input_dim = df_norm.shape[1] 
                                 temp_trainer.create_model(input_dim) # Crea il modello con la giusta dimensione
                                 temp_trainer.model.load_state_dict(temp_trainer.model_state_dict)
                                 temp_trainer.model.to(temp_trainer.device) # Sposta sul dispositivo corretto
                                 self.models[symbol] = temp_trainer
                                 model_trainer = temp_trainer
                                 logger.info(f"Modello per {symbol} ricaricato e inizializzato on-the-fly.")
                             else:
                                logger.warning(f"Model state_dict non trovato nel trainer caricato per {symbol}")
                                continue # Salta questo simbolo se non si può caricare
                        else:
                             logger.warning(f"Impossibile ricaricare il modello per {symbol} da {model_path}")
                             continue # Salta questo simbolo se non si può caricare
                    else:
                        logger.warning(f"Nessun modello .pt trovato per {symbol} per la previsione.")
                        continue # Salta questo simbolo se non c'è modello

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
                    # Usa il metodo predict del ModelTrainer (che usa PyTorch)
                    # Il metodo predict del trainer si aspetta l'input normalizzato
                    # e restituisce la previsione normalizzata
                    
                    # Prepara l'input per il modello PyTorch (batch_size=1)
                    input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(model_trainer.device)
                    
                    # Esegui la previsione con il modello PyTorch
                    model_trainer.model.eval() # Assicurati che sia in modalità valutazione
                    with torch.no_grad():
                         # Verifica se il modello è stato creato (potrebbe essere None dopo il caricamento)
                         if model_trainer.model is None:
                             if hasattr(model_trainer, 'model_state_dict'):
                                 # Ricrea il modello se abbiamo lo state_dict
                                 input_dim = current_sequence.shape[1]
                                 model_trainer.create_model(input_dim)
                                 model_trainer.model.load_state_dict(model_trainer.model_state_dict)
                                 model_trainer.model.to(model_trainer.device)
                                 logger.info(f"Modello per {symbol} ricreato per la previsione.")
                             else:
                                 logger.error(f"Modello per {symbol} non è inizializzato e manca state_dict.")
                                 raise ValueError(f"Modello per {symbol} non inizializzato.")
                                 
                         prediction_norm = model_trainer.model(input_tensor).item()

                    # Aggiungi la previsione normalizzata alla sequenza per la prossima iterazione
                    # Dobbiamo ricostruire l'intera riga della sequenza successiva
                    # Assumiamo che la previsione sia per la prima colonna (es. 'Close')
                    new_row = current_sequence[-1].copy() 
                    new_row[0] = prediction_norm # Aggiorna la colonna predetta (assumendo sia la prima)
                    # Nota: Le altre feature nella new_row rimangono quelle dell'ultimo timestep reale.
                    # Questo è un approccio comune ma potrebbe essere migliorato prevedendo tutte le feature.
                    
                    # Aggiorna la sequenza
                    current_sequence = np.vstack([current_sequence[1:], new_row])
                    
                    # Salva la previsione normalizzata
                    predicted_values.append(prediction_norm)
                
                # Denormalizza le previsioni usando il DataPreprocessor del ModelTrainer
                # Il preprocessor del trainer ha lo scaler corretto per i prezzi
                predicted_closes = [model_trainer.data_preprocessor.inverse_transform_price(p) for p in predicted_values]
                
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
                    # Verifica che un ModelTrainer sia stato caricato per il simbolo
                    model_trainer = self.models.get(symbol)
                    if not isinstance(model_trainer, ModelTrainer) or model_trainer.model is None:
                         # Prova a ricaricare se necessario
                         model_path = self._find_model_path(symbol, input_dir='models')
                         if model_path:
                             temp_trainer = ModelTrainer()
                             if temp_trainer.load_model(model_path):
                                 if hasattr(temp_trainer, 'model_state_dict'):
                                     input_dim = data['X_test'].shape[2] # Ottieni input_dim dai dati di test
                                     temp_trainer.create_model(input_dim)
                                     temp_trainer.model.load_state_dict(temp_trainer.model_state_dict)
                                     temp_trainer.model.to(temp_trainer.device)
                                     self.models[symbol] = temp_trainer
                                     model_trainer = temp_trainer
                                     logger.info(f"Modello per {symbol} ricaricato per valutazione.")
                                 else:
                                     logger.warning(f"State_dict mancante per {symbol} durante ricaricamento per valutazione.")
                                     continue
                             else:
                                 logger.warning(f"Impossibile ricaricare modello per {symbol} per valutazione.")
                                 continue
                         else:
                             logger.warning(f"Nessun modello .pt trovato per {symbol} per valutazione.")
                             continue

                    # Ottieni i dati di test e convertili in tensori PyTorch
                    X_test = torch.FloatTensor(data['X_test']).to(model_trainer.device)
                    y_test = torch.FloatTensor(data['y_test']).to(model_trainer.device)
                    
                    # Genera le previsioni usando il modello PyTorch
                    model_trainer.model.eval() # Modalità valutazione
                    with torch.no_grad():
                        y_pred_tensor = model_trainer.model(X_test).squeeze()
                    
                    # Converte i tensori in array numpy per le metriche sklearn
                    y_pred = y_pred_tensor.cpu().numpy()
                    y_true = y_test.cpu().numpy()
                    
                    # Calcola le metriche
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred) # Corrected indentation
                    r2 = r2_score(y_true, y_pred) # Corrected indentation
                
                    # Salva le metriche
                    metrics[symbol] = { # Corrected indentation
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                
                    logger.info(f"Metriche per {symbol}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}") # Corrected indentation
            
            except Exception as e: # Added missing except block from previous edit
                logger.error(f"Errore nella valutazione del modello per {symbol}: {e}")
        
        return metrics # Corrected indentation
    
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
