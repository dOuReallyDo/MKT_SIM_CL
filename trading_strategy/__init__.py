"""
Trading Strategy Package.

Questo pacchetto contiene le implementazioni delle strategie di trading.
"""

from .strategies import (
    TradingStrategy,
    RandomStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    ValueInvestingStrategy,
    NeuralNetworkStrategy
)
from .neural_network_bridge import NeuralNetworkBridge
import logging

def get_available_strategies():
    """
    Restituisce un dizionario con le strategie disponibili e il loro stato.
    
    Returns:
        dict: Dizionario con nome strategia come chiave e un altro dizionario
              come valore, contenente 'class' e 'status'.
              Es: {'random': {'class': RandomStrategy, 'status': 'implemented'}}
    """
    return {
        'random': {'class': RandomStrategy, 'status': 'implemented'},
        'mean_reversion': {'class': MeanReversionStrategy, 'status': 'implemented'},
        'trend_following': {'class': TrendFollowingStrategy, 'status': 'implemented'},
        'value_investing': {'class': ValueInvestingStrategy, 'status': 'not_implemented'}, # Marcata come non implementata
        'neural_network': {'class': NeuralNetworkStrategy, 'status': 'implemented'}  # Ora è implementata
    }

def create_strategy(strategy_name, **kwargs):
    """
    Crea un'istanza di una strategia basata sul nome.
    Solo se lo stato è 'implemented'.
    
    Args:
        strategy_name: Nome della strategia
        **kwargs: Parametri per l'inizializzazione della strategia
        
    Returns:
        Istanza della strategia richiesta o None se non trovata o non implementata.
    """
    strategies_info = get_available_strategies()
    
    if strategy_name in strategies_info and strategies_info[strategy_name]['status'] == 'implemented':
        strategy_class = strategies_info[strategy_name]['class']
        return strategy_class(**kwargs)
    
    # Fallback a None se non implementata o non trovata
    logging.warning(f"Tentativo di creare strategia non implementata o sconosciuta: {strategy_name}. Restituito None.")
    return None # Modificato fallback da RandomStrategy a None

__all__ = [
    'TradingStrategy',
    'RandomStrategy',
    'MeanReversionStrategy',
    'TrendFollowingStrategy',
    'ValueInvestingStrategy',
    'NeuralNetworkStrategy',
    'NeuralNetworkBridge',
    'get_available_strategies',
    'create_strategy'
]
