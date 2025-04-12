#!/usr/bin/env python3
"""
Script per generare dati di esempio per testing.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_stock_data(symbol, start_date, end_date, output_dir='./tests/test_data'):
    """
    Genera dati di prezzo simulati per un simbolo
    
    Args:
        symbol: Simbolo del titolo
        start_date: Data di inizio (str o datetime)
        end_date: Data di fine (str o datetime)
        output_dir: Directory di output
    
    Returns:
        DataFrame con i dati generati
    """
    # Converti le date in datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Genera date di trading (giorni feriali)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Inizializza il prezzo di base
    base_price = np.random.uniform(50.0, 200.0)
    
    # Genera prezzo di chiusura con random walk
    np.random.seed(hash(symbol) % 10000)  # Seed per riproducibilità ma diverso per ogni simbolo
    price_changes = np.random.normal(0, 0.02, len(date_range))  # 2% di volatilità
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Crea i dati
    data = []
    for i, date in enumerate(date_range):
        close = prices[i]
        daily_volatility = close * np.random.uniform(0.01, 0.03)  # 1-3% di volatilità giornaliera
        
        # Crea valori con variazioni realistiche
        open_price = close * (1 + np.random.normal(0, 0.01))  # Apertura vicina alla chiusura precedente
        high = max(close, open_price) + np.random.uniform(0, daily_volatility)
        low = min(close, open_price) - np.random.uniform(0, daily_volatility)
        volume = int(np.random.uniform(500000, 5000000))  # Volume tra 500k e 5M
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    # Crea DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # Salva il file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}.csv")
    df.to_csv(output_file, date_format='%Y-%m-%d')
    
    print(f"Dati di esempio generati per {symbol}: {len(df)} righe salvate in {output_file}")
    return df

def main():
    """Funzione principale"""
    # Genera dati per alcuni simboli comuni
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Data di inizio 60 giorni fa, fino a oggi
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    # Directory di output - sia per test che per data
    for output_dir in ['./tests/test_data', './data']:
        for symbol in symbols:
            generate_sample_stock_data(symbol, start_date, end_date, output_dir)

if __name__ == "__main__":
    main() 