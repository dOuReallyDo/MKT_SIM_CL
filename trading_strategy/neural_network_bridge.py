"""
Neural Network Bridge Module.

Questo modulo fornisce un ponte tra le strategie di trading basate su reti neurali
e il modulo di addestramento delle reti neurali.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

from neural_network.model_trainer import ModelTrainer
from neural_network.integration import NeuralNetworkIntegration
from trading_strategy.strategies import NeuralNetworkStrategy

# Configura il logger
logger = logging.getLogger('trading_strategy.neural_network_bridge')

class NeuralNetworkBridge:
    """
    Classe ponte tra le strategie di trading basate su reti neurali
    e il modulo di addestramento delle reti neurali.
    """
    
    def __init__(self, model_trainer=None, data_collector=None):
        """
        Inizializza il ponte.
        
        Args:
            model_trainer: Istanza di ModelTrainer
            data_collector: Istanza di DataCollector
        """
        self.model_trainer = model_trainer or ModelTrainer()
        self.data_collector = data_collector
        self.integration = None
        
        if data_collector:
            self.integration = NeuralNetworkIntegration(self.model_trainer, data_collector)
        
        self.models = {}
        self.strategies = {}
    
    def create_strategy(self, symbol: str, sequence_length: int = 10, threshold: float = 0.01) -> Optional[NeuralNetworkStrategy]:
        """
        Crea una strategia di trading basata su rete neurale per un simbolo.
        
        Args:
            symbol: Simbolo dell'asset
            sequence_length: Lunghezza della sequenza per la previsione
            threshold: Soglia per la generazione dei segnali (in percentuale)
            
        Returns:
            Strategia di trading basata su rete neurale o None se non è possibile crearla
        """
        try:
            # Verifica che il modello sia stato addestrato
            if symbol not in self.models:
                # Cerca un modello salvato
                model_path = self._find_model_path(symbol)
                if model_path:
                    self.load_model(symbol, model_path)
                else:
                    logger.warning(f"Nessun modello trovato per {symbol}")
                    return None
            
            # Crea la strategia
            strategy = NeuralNetworkStrategy(
                model_trainer=self.models[symbol],
                sequence_length=sequence_length,
                threshold=threshold
            )
            
            # Salva la strategia
            self.strategies[symbol] = strategy
            
            logger.info(f"Strategia creata per {symbol}")
            
            return strategy
        
        except Exception as e:
            logger.error(f"Errore nella creazione della strategia per {symbol}: {e}")
            return None
    
    def train_model(self, symbol: str, start_date: str, end_date: str, model_type: str = 'lstm', epochs: int = 50, batch_size: int = 32) -> bool:
        """
        Addestra un modello per un simbolo.
        
        Args:
            symbol: Simbolo dell'asset
            start_date: Data di inizio
            end_date: Data di fine
            model_type: Tipo di modello ('lstm', 'cnn', 'transformer')
            epochs: Numero di epoche
            batch_size: Dimensione del batch
            
        Returns:
            True se l'addestramento è riuscito, False altrimenti
        """
        try:
            if not self.integration:
                logger.error("Integrazione non inizializzata")
                return False
            
            # Prepara i dati per l'addestramento
            prepared_data = self.integration.prepare_data_for_training(
                [symbol], start_date, end_date
            )
            
            if not prepared_data or symbol not in prepared_data:
                logger.error(f"Impossibile preparare i dati per {symbol}")
                return False
            
            # Configura il model trainer
            config = {
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': 0.001,
                'sequence_length': 10
            }
            
            # Crea un nuovo model trainer
            model_trainer = ModelTrainer(model_type=model_type, config=config)
            
            # Addestra il modello
            data = prepared_data[symbol]
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            
            # Crea il modello
            input_dim = X_train.shape[2]
            model = model_trainer.create_model(input_dim)
            
            try:
                # Addestra il modello
                result = model_trainer.train(
                    X_train, y_train, X_test, y_test,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                # Salva il modello
                self.models[symbol] = model_trainer
                
                logger.info(f"Modello addestrato per {symbol}: MSE={result['mse']:.6f}, RMSE={result['rmse']:.6f}")
                
                return True
            
            except Exception as train_error:
                logger.error(f"Errore nell'addestramento del modello per {symbol}: {train_error}")
                return False
        
        except Exception as e:
            logger.error(f"Errore nella preparazione o nell'addestramento del modello per {symbol}: {e}")
            return False
    
    def load_model(self, symbol: str, model_path: str) -> bool:
        """
        Carica un modello salvato.
        
        Args:
            symbol: Simbolo dell'asset
            model_path: Percorso del file del modello
            
        Returns:
            True se il caricamento è riuscito, False altrimenti
        """
        try:
            # Crea un nuovo model trainer
            model_trainer = ModelTrainer()
            
            # Carica il modello
            success = model_trainer.load_model(model_path)
            
            if not success:
                logger.error(f"Impossibile caricare il modello da {model_path}")
                return False
            
            # Salva il model trainer
            self.models[symbol] = model_trainer
            
            logger.info(f"Modello caricato per {symbol} da {model_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello per {symbol}: {e}")
            return False
    
    def _find_model_path(self, symbol: str) -> Optional[str]:
        """
        Trova il percorso del modello più recente per un simbolo.
        
        Args:
            symbol: Simbolo dell'asset
            
        Returns:
            Percorso del modello o None se non trovato
        """
        try:
            # Directory dei modelli
            models_dir = './models'
            
            if not os.path.exists(models_dir):
                return None
            
            # Trova tutti i file dei modelli per il simbolo
            model_files = [
                f for f in os.listdir(models_dir)
                if f.startswith(f"{symbol}_") and f.endswith('.pt')
            ]
            
            if not model_files:
                return None
            
            # Ordina per data (più recente prima)
            model_files.sort(reverse=True)
            
            # Restituisci il percorso del modello più recente
            return os.path.join(models_dir, model_files[0])
        
        except Exception as e:
            logger.error(f"Errore nella ricerca del modello per {symbol}: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """
        Restituisce la lista dei simboli per cui sono disponibili modelli.
        
        Returns:
            Lista di simboli
        """
        try:
            # Directory dei modelli
            models_dir = './models'
            
            if not os.path.exists(models_dir):
                return []
            
            # Trova tutti i file dei modelli
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            
            # Estrai i simboli
            symbols = set()
            for file in model_files:
                parts = file.split('_')
                if len(parts) > 0:
                    symbols.add(parts[0])
            
            return list(symbols)
        
        except Exception as e:
            logger.error(f"Errore nel recupero dei modelli disponibili: {e}")
            return []
    
    def generate_predictions(self, symbol: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """
        Genera previsioni per un simbolo.
        
        Args:
            symbol: Simbolo dell'asset
            days: Numero di giorni da prevedere
            
        Returns:
            Dizionario con le previsioni o None se non è possibile generarle
        """
        try:
            if not self.integration:
                logger.error("Integrazione non inizializzata")
                return None
            
            # Verifica che il modello sia stato addestrato
            if symbol not in self.models:
                # Cerca un modello salvato
                model_path = self._find_model_path(symbol)
                if model_path:
                    self.load_model(symbol, model_path)
                else:
                    logger.warning(f"Nessun modello trovato per {symbol}")
                    return None
            
            # Genera le previsioni
            predictions = self.integration.predict([symbol], days=days)
            
            if not predictions or symbol not in predictions:
                logger.error(f"Impossibile generare previsioni per {symbol}")
                return None
            
            # Genera i segnali di trading
            signals = self.integration.generate_trading_signals(predictions)
            
            if not signals or symbol not in signals:
                logger.error(f"Impossibile generare segnali per {symbol}")
                return None
            
            return {
                'predictions': predictions[symbol],
                'signals': signals[symbol]
            }
        
        except Exception as e:
            logger.error(f"Errore nella generazione delle previsioni per {symbol}: {e}")
            return None
