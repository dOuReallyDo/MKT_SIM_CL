"""
Trading Agent Module.

Questo modulo contiene la classe TradingAgent per la gestione degli agenti di trading.
"""

import logging

class TradingAgent:
    def __init__(self, id, initial_capital, strategy):
        """
        Inizializza un agente di trading
        
        Args:
            id: Identificatore univoco dell'agente
            initial_capital: Capitale iniziale
            strategy: Strategia di trading da utilizzare
        """
        self.id = id
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.portfolio = {}  # {symbol: quantity}
        self.strategy = strategy
        self.transactions = []
        
        # Configurazione del logging
        self.logger = logging.getLogger(f'TradingAgent_{id}')
    
    def generate_signal(self, market_data):
        """
        Genera un segnale di trading basato sulla strategia
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con il segnale di trading o None se non ci sono segnali
        """
        try:
            return self.strategy.generate_signal(market_data)
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale: {e}")
            return None
    
    def get_portfolio_value(self, market_data):
        """
        Calcola il valore totale del portafoglio
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Valore totale del portafoglio
        """
        portfolio_value = self.cash
        
        for symbol, quantity in self.portfolio.items():
            if symbol in market_data:
                price = market_data[symbol]['close']
                portfolio_value += price * quantity
        
        return portfolio_value
    
    def get_performance_metrics(self, market_data):
        """
        Calcola le metriche di performance dell'agente
        
        Args:
            market_data: Dizionario con i dati di mercato
            
        Returns:
            Dizionario con le metriche di performance
        """
        current_value = self.get_portfolio_value(market_data)
        absolute_return = current_value - self.initial_capital
        percentage_return = (absolute_return / self.initial_capital) * 100
        
        return {
            'id': self.id,
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'absolute_return': absolute_return,
            'percentage_return': percentage_return
        }
    
    def execute_buy(self, symbol, quantity, price):
        """
        Esegue un acquisto
        
        Args:
            symbol: Simbolo dell'asset
            quantity: Quantità da acquistare
            price: Prezzo di acquisto
            
        Returns:
            True se l'acquisto è stato eseguito, False altrimenti
        """
        cost = quantity * price
        
        if cost > self.cash:
            self.logger.warning(f"Fondi insufficienti per acquistare {quantity} {symbol}")
            return False
        
        self.cash -= cost
        if symbol in self.portfolio:
            self.portfolio[symbol] += quantity
        else:
            self.portfolio[symbol] = quantity
        
        transaction = {
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'total': cost
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Acquisto eseguito: {quantity} {symbol} a {price}")
        return True
    
    def execute_sell(self, symbol, quantity, price):
        """
        Esegue una vendita
        
        Args:
            symbol: Simbolo dell'asset
            quantity: Quantità da vendere
            price: Prezzo di vendita
            
        Returns:
            True se la vendita è stata eseguita, False altrimenti
        """
        if symbol not in self.portfolio or self.portfolio[symbol] < quantity:
            self.logger.warning(f"Portfolio insufficiente per vendere {quantity} {symbol}")
            return False
        
        revenue = quantity * price
        self.cash += revenue
        self.portfolio[symbol] -= quantity
        
        # Rimuovi il simbolo dal portfolio se la quantità è 0
        if self.portfolio[symbol] == 0:
            del self.portfolio[symbol]
        
        transaction = {
            'symbol': symbol,
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'total': revenue
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Vendita eseguita: {quantity} {symbol} a {price}")
        return True 