"""
Trading Strategies Module.

Questo modulo contiene le implementazioni delle diverse strategie di trading.
"""

import random
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional

class TradingStrategy(ABC):
    """
    Classe base astratta per le strategie di trading.
    
    Tutte le strategie di trading devono implementare il metodo generate_signal.
    """
    
    def __init__(self):
        """Inizializza la strategia di trading."""
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
    
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Genera un segnale di trading basato sui dati di mercato.
        
        Args:
            market_data: Dizionario con i dati di mercato, strutturato come:
                        {
                            'AAPL': {
                                'open': 150.0,
                                'high': 152.0,
                                'low': 149.0,
                                'close': 151.0,
                                'volume': 100000
                            },
                            'MSFT': { ... }
                        }
        
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali:
                        {
                            'symbol': 'AAPL',
                            'action': 'buy', # o 'sell'
                            'quantity': 10,
                            'price': 151.0,
                            'confidence': 0.8 # opzionale
                        }
        """
        pass

class RandomStrategy(TradingStrategy):
    """
    Strategia di trading casuale.
    
    Genera segnali di trading in modo casuale.
    """
    
    def __init__(self, buy_probability: float = 0.1, sell_probability: float = 0.1):
        """
        Inizializza la strategia casuale.
        
        Args:
            buy_probability: Probabilità di generare un segnale di acquisto
            sell_probability: Probabilità di generare un segnale di vendita
        """
        super().__init__()
        self.buy_probability = buy_probability
        self.sell_probability = sell_probability
    
    def generate_signal(self, market_data: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Genera un segnale di trading casuale.
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            # Se non ci sono dati di mercato, non generare segnali
            if not market_data:
                return None
            
            # Scegli un simbolo casuale
            symbol = random.choice(list(market_data.keys()))
            
            # Probabilità di generare un segnale
            p = random.random()
            
            if p < self.buy_probability:
                # Genera un segnale di acquisto
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': random.randint(1, 10),
                    'price': market_data[symbol]['close']
                }
            elif p < self.buy_probability + self.sell_probability:
                # Genera un segnale di vendita
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': random.randint(1, 10),
                    'price': market_data[symbol]['close']
                }
            
            # Nessun segnale generato
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale casuale: {e}")
            return None

class MeanReversionStrategy(TradingStrategy):
    """
    Strategia di Mean Reversion.
    
    Acquista quando il prezzo è significativamente al di sotto della media mobile
    e vende quando è significativamente al di sopra.
    """
    
    def __init__(self, window: int = 20, threshold: float = 0.02):
        """
        Inizializza la strategia di Mean Reversion.
        
        Args:
            window: Dimensione della finestra per la media mobile
            threshold: Soglia oltre la quale generare segnali (in percentuale)
        """
        super().__init__()
        self.window = window
        self.threshold = threshold
        self.historical_prices = {}  # {symbol: [prezzi]}
    
    def generate_signal(self, market_data: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Genera un segnale di trading basato sulla Mean Reversion.
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            if not market_data:
                return None
            
            for symbol, data in market_data.items():
                # Aggiorna lo storico dei prezzi
                if symbol not in self.historical_prices:
                    self.historical_prices[symbol] = []
                
                self.historical_prices[symbol].append(data['close'])
                
                # Mantieni solo gli ultimi 'window' valori
                if len(self.historical_prices[symbol]) > self.window * 2:
                    self.historical_prices[symbol] = self.historical_prices[symbol][-self.window * 2:]
                
                # Calcola la media solo se ci sono abbastanza dati
                if len(self.historical_prices[symbol]) >= self.window:
                    # Calcola la media mobile
                    moving_avg = sum(self.historical_prices[symbol][-self.window:]) / self.window
                    current_price = data['close']
                    
                    # Calcola la deviazione dalla media
                    deviation = (current_price - moving_avg) / moving_avg
                    
                    # Genera segnali
                    if deviation < -self.threshold:
                        # Prezzo significativamente al di sotto della media: Buy
                        quantity = max(1, int(abs(deviation) * 10))  # Più forte la deviazione, più si compra
                        return {
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': quantity,
                            'price': current_price,
                            'confidence': abs(deviation)
                        }
                    elif deviation > self.threshold:
                        # Prezzo significativamente al di sopra della media: Sell
                        quantity = max(1, int(abs(deviation) * 10))  # Più forte la deviazione, più si vende
                        return {
                            'symbol': symbol,
                            'action': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'confidence': abs(deviation)
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale di Mean Reversion: {e}")
            return None

class TrendFollowingStrategy(TradingStrategy):
    """
    Strategia di Trend Following.
    
    Acquista quando il prezzo è in trend rialzista (media mobile breve sopra
    la media mobile lunga) e vende quando è in trend ribassista.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 50):
        """
        Inizializza la strategia di Trend Following.
        
        Args:
            short_window: Dimensione della finestra breve per la media mobile
            long_window: Dimensione della finestra lunga per la media mobile
        """
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.historical_prices = {}  # {symbol: [prezzi]}
        self.previous_trend = {}  # {symbol: 'up' o 'down'}
    
    def generate_signal(self, market_data: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Genera un segnale di trading basato sul Trend Following.
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            if not market_data:
                return None
            
            for symbol, data in market_data.items():
                # Aggiorna lo storico dei prezzi
                if symbol not in self.historical_prices:
                    self.historical_prices[symbol] = []
                
                self.historical_prices[symbol].append(data['close'])
                
                # Mantieni solo gli ultimi 'long_window' valori + alcuni extra
                if len(self.historical_prices[symbol]) > self.long_window * 2:
                    self.historical_prices[symbol] = self.historical_prices[symbol][-self.long_window * 2:]
                
                # Calcola la media solo se ci sono abbastanza dati
                if len(self.historical_prices[symbol]) >= self.long_window:
                    # Calcola le medie mobili
                    short_ma = sum(self.historical_prices[symbol][-self.short_window:]) / self.short_window
                    long_ma = sum(self.historical_prices[symbol][-self.long_window:]) / self.long_window
                    
                    # Determina il trend attuale
                    current_trend = 'up' if short_ma > long_ma else 'down'
                    
                    # Verifica se c'è un cambio di trend
                    previous_trend = self.previous_trend.get(symbol)
                    if previous_trend and current_trend != previous_trend:
                        # C'è un cambio di trend, genera un segnale
                        if current_trend == 'up':
                            # Trend rialzista: Buy
                            return {
                                'symbol': symbol,
                                'action': 'buy',
                                'quantity': 5,  # Quantità fissa o proporzionale alla forza del segnale
                                'price': data['close'],
                                'trend': current_trend
                            }
                        else:
                            # Trend ribassista: Sell
                            return {
                                'symbol': symbol,
                                'action': 'sell',
                                'quantity': 5,  # Quantità fissa o proporzionale alla forza del segnale
                                'price': data['close'],
                                'trend': current_trend
                            }
                    
                    # Aggiorna il trend precedente
                    self.previous_trend[symbol] = current_trend
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale di Trend Following: {e}")
            return None

class ValueInvestingStrategy(TradingStrategy):
    """
    Strategia di Value Investing.
    
    Acquista quando il prezzo è al di sotto del valore "intrinseco" stimato
    e vende quando è al di sopra.
    """
    
    def __init__(self, pe_threshold: float = 15.0, pb_threshold: float = 1.5):
        """
        Inizializza la strategia di Value Investing.
        
        Args:
            pe_threshold: Soglia del rapporto prezzo/utili
            pb_threshold: Soglia del rapporto prezzo/valore contabile
        """
        super().__init__()
        self.pe_threshold = pe_threshold
        self.pb_threshold = pb_threshold
        self.fundamental_data = {}  # Dati fondamentali per ogni simbolo
    
    def update_fundamental_data(self, symbol: str, pe_ratio: float, pb_ratio: float):
        """
        Aggiorna i dati fondamentali per un simbolo.
        
        Args:
            symbol: Simbolo dell'asset
            pe_ratio: Rapporto prezzo/utili
            pb_ratio: Rapporto prezzo/valore contabile
        """
        self.fundamental_data[symbol] = {
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio
        }
    
    def generate_signal(self, market_data: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Genera un segnale di trading basato sul Value Investing.
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            if not market_data or not self.fundamental_data:
                return None
            
            for symbol, data in market_data.items():
                # Verifica se ci sono dati fondamentali per questo simbolo
                if symbol not in self.fundamental_data:
                    continue
                
                fund_data = self.fundamental_data[symbol]
                
                # Valuta se il titolo è sottovalutato o sopravvalutato
                pe_ratio = fund_data.get('pe_ratio')
                pb_ratio = fund_data.get('pb_ratio')
                
                if pe_ratio and pb_ratio:
                    if pe_ratio < self.pe_threshold and pb_ratio < self.pb_threshold:
                        # Titolo sottovalutato: Buy
                        return {
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': 10,
                            'price': data['close'],
                            'reason': 'undervalued'
                        }
                    elif pe_ratio > self.pe_threshold * 1.5 and pb_ratio > self.pb_threshold * 1.5:
                        # Titolo sopravvalutato: Sell
                        return {
                            'symbol': symbol,
                            'action': 'sell',
                            'quantity': 10,
                            'price': data['close'],
                            'reason': 'overvalued'
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale di Value Investing: {e}")
            return None

class NeuralNetworkStrategy(TradingStrategy):
    """
    Strategia basata su una rete neurale.
    
    Utilizza un modello di rete neurale per generare previsioni e segnali di trading.
    """
    
    def __init__(self, model_trainer=None, sequence_length: int = 10, threshold: float = 0.01):
        """
        Inizializza la strategia basata su rete neurale.
        
        Args:
            model_trainer: Istanza di ModelTrainer con il modello addestrato
            sequence_length: Lunghezza della sequenza per la previsione
            threshold: Soglia per la generazione dei segnali (in percentuale)
        """
        super().__init__()
        self.model_trainer = model_trainer
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.historical_prices = {}  # {symbol: [prezzi]}
    
    def generate_signal(self, market_data: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Genera un segnale di trading basato sulle previsioni della rete neurale.
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            if not market_data or not self.model_trainer:
                return None
            
            for symbol, data in market_data.items():
                # Aggiorna lo storico dei prezzi
                if symbol not in self.historical_prices:
                    self.historical_prices[symbol] = []
                
                self.historical_prices[symbol].append(data['close'])
                
                # Mantieni solo gli ultimi valori necessari
                if len(self.historical_prices[symbol]) > self.sequence_length * 2:
                    self.historical_prices[symbol] = self.historical_prices[symbol][-self.sequence_length * 2:]
                
                # Verifica se ci sono abbastanza dati per la previsione
                if len(self.historical_prices[symbol]) >= self.sequence_length:
                    # Prepara i dati per la previsione
                    sequence = self.historical_prices[symbol][-self.sequence_length:]
                    current_price = data['close']
                    
                    # Normalizziamo i dati se necessario
                    normalized_sequence = self._normalize_sequence(sequence)
                    
                    # Esegui la previsione
                    predicted_price = self._predict_price(normalized_sequence, symbol)
                    
                    if predicted_price is not None:
                        # Calcola la variazione percentuale prevista
                        expected_change_pct = (predicted_price - current_price) / current_price
                        
                        # Genera segnali in base alla previsione
                        if expected_change_pct > self.threshold:
                            # Ci si aspetta un aumento del prezzo: Buy
                            quantity = max(1, int(abs(expected_change_pct) * 100))  # Quantità proporzionale alla variazione prevista
                            return {
                                'symbol': symbol,
                                'action': 'buy',
                                'quantity': min(quantity, 10),  # Limita la quantità a un massimo di 10
                                'price': current_price,
                                'expected_price': predicted_price,
                                'expected_change_pct': expected_change_pct
                            }
                        elif expected_change_pct < -self.threshold:
                            # Ci si aspetta una diminuzione del prezzo: Sell
                            quantity = max(1, int(abs(expected_change_pct) * 100))  # Quantità proporzionale alla variazione prevista
                            return {
                                'symbol': symbol,
                                'action': 'sell',
                                'quantity': min(quantity, 10),  # Limita la quantità a un massimo di 10
                                'price': current_price,
                                'expected_price': predicted_price,
                                'expected_change_pct': expected_change_pct
                            }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale neurale: {e}")
            return None
    
    def _normalize_sequence(self, sequence: List[float]) -> List[float]:
        """
        Normalizza una sequenza di prezzi.
        
        Args:
            sequence: Lista di prezzi
            
        Returns:
            Lista normalizzata
        """
        min_val = min(sequence)
        max_val = max(sequence)
        
        if max_val == min_val:
            return [0.5] * len(sequence)
        
        return [(price - min_val) / (max_val - min_val) for price in sequence]
    
    def _predict_price(self, normalized_sequence: List[float], symbol: str) -> Optional[float]:
        """
        Esegue la previsione del prezzo.
        
        Args:
            normalized_sequence: Sequenza normalizzata
            symbol: Simbolo dell'asset
            
        Returns:
            Prezzo previsto o None in caso di errore
        """
        try:
            if not self.model_trainer:
                return None
            
            try:
                # Esegui la previsione direttamente con ModelTrainer
                prediction = self.model_trainer.predict(np.array(normalized_sequence).reshape(1, self.sequence_length, 1))
                return prediction
            except Exception as prediction_error:
                self.logger.error(f"Errore nella previsione: {prediction_error}")
                return None
            
        except Exception as e:
            self.logger.error(f"Errore nella previsione del prezzo: {e}")
            return None
