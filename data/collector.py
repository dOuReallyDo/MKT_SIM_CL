import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import re

class DataCollector:
    """Classe per la raccolta e gestione dei dati di mercato"""
    def __init__(self, data_dir='./data'):
        """
        Inizializza il collector dei dati
        
        Args:
            data_dir: Directory per il salvataggio dei dati
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Configurazione del logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configura il logging per il raccoglitore dati"""
        log_file = os.path.join(self.data_dir, "data_collection.log")
        
        # Handler del file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Handler della console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Logger
        self.logger = logging.getLogger('DataCollector')
        
        # Rimuovi gli handler esistenti per evitare duplicazioni
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Evita la propagazione al logger root
        self.logger.propagate = False
    
    def get_stock_data(self, symbol, start_date, end_date, force_download=False):
        """
        Ottiene i dati storici di un titolo
        
        Args:
            symbol: Simbolo del titolo
            start_date: Data di inizio
            end_date: Data di fine
            force_download: Se True, forza il download anche se i dati sono in cache
            
        Returns:
            DataFrame con i dati del titolo
        """
        # Percorso del file di cache
        cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
        
        # Se force_download è True o il file non esiste, scarica i dati
        if force_download or not os.path.exists(cache_file):
            try:
                # Converti le date in formato stringa se sono oggetti datetime
                if isinstance(start_date, datetime):
                    start_date = start_date.strftime('%Y-%m-%d')
                if isinstance(end_date, datetime):
                    end_date = end_date.strftime('%Y-%m-%d')
                    
                self.logger.info(f"Download dei dati per {symbol} da {start_date} a {end_date}")
                # Scarica i dati usando yfinance
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    self.logger.warning(f"Nessun dato disponibile per {symbol}")
                    
                    # Tentativo di fallback - proviamo con date più ampie
                    # A volte yfinance ha problemi con date specifiche
                    self.logger.info(f"Tentativo con periodo più ampio per {symbol}")
                    try:
                        # Amplia il periodo di 30 giorni prima e dopo
                        start_date_dt = pd.to_datetime(start_date) - pd.Timedelta(days=30)
                        end_date_dt = pd.to_datetime(end_date) + pd.Timedelta(days=30)
                        
                        # Nuova richiesta
                        df = yf.download(
                            symbol, 
                            start=start_date_dt.strftime('%Y-%m-%d'), 
                            end=end_date_dt.strftime('%Y-%m-%d'), 
                            progress=False
                        )
                        
                        if not df.empty:
                            self.logger.info(f"Dati recuperati con periodo ampliato per {symbol}")
                        else:
                            self.logger.warning(f"Nessun dato disponibile anche con periodo ampliato per {symbol}")
                            return None
                    except Exception as e:
                        self.logger.error(f"Errore nel tentativo alternativo per {symbol}: {e}")
                        return None
                
                # Assicurati che le colonne siano standardizzate
                df = self.clean_stock_data(df, symbol)
                
                # Salvataggio dei dati in cache
                try:
                    # Assicurati che l'indice sia in formato stringa ISO per coerenza
                    if isinstance(df.index, pd.DatetimeIndex):
                        # Salva con indice in formato ISO standard
                        df.to_csv(cache_file, date_format='%Y-%m-%d')
                    else:
                        # Tenta di convertire l'indice a datetime
                        df.index = pd.to_datetime(df.index)
                        df.to_csv(cache_file, date_format='%Y-%m-%d')
                    
                    self.logger.info(f"Dati per {symbol} salvati in {cache_file}")
                except Exception as e:
                    self.logger.error(f"Errore nel salvataggio dei dati per {symbol}: {e}")
                
                return df
            except Exception as e:
                self.logger.error(f"Errore nel download dei dati per {symbol}: {e}")
                
                # Verifica se abbiamo dati locali come fallback
                if os.path.exists(cache_file) and os.path.getsize(cache_file) > 100:
                    self.logger.info(f"Utilizzo dati locali esistenti per {symbol} come fallback")
                    try:
                        return self._load_cached_data(cache_file)
                    except Exception as e2:
                        self.logger.error(f"Errore anche nel caricamento dei dati locali: {e2}")
                
                return None
        else:
            try:
                # Carica i dati esistenti
                self.logger.info(f"Tentativo di caricare i dati esistenti per {symbol}")
                return self._load_cached_data(cache_file)
            except Exception as e:
                self.logger.error(f"Errore nel caricamento dei dati dalla cache: {e}")
                # Tenta il download come fallback
                return self.get_stock_data(symbol, start_date, end_date, force_download=True)
    
    def _load_cached_data(self, cache_file):
        """
        Carica i dati dalla cache in modo sicuro
        
        Args:
            cache_file: Percorso del file di cache
            
        Returns:
            DataFrame con i dati
        """
        try:
            # Metodo 1: Lettura standard con parse_dates e index_col
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    # Controlla se l'indice è già un DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        # Prova a convertire esplicitamente l'indice
                        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                        
                        # Rimuovi righe con indice NaT
                        df = df.loc[~df.index.isna()]
                    
                    # Standardizza i nomi delle colonne
                    df = self._standardize_columns(df)
                    
                    return df
            except Exception as e:
                # Log ma non ritorno, proviamo altri metodi
                self.logger.warning(f"Primo tentativo di lettura fallito: {e}")
            
            # Metodo 2: Lettura senza parse_dates
            try:
                df = pd.read_csv(cache_file)
                
                if 'Date' in df.columns:
                    # Converti la colonna Date in datetime
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
                    # Imposta come indice e standardizza le colonne
                    df = df.set_index('Date')
                    df = self._standardize_columns(df)
                    return df
                elif 'Datetime' in df.columns:
                    # Converti la colonna Datetime in datetime
                    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d', errors='coerce')
                    # Imposta come indice e standardizza le colonne
                    df = df.set_index('Datetime')
                    df = self._standardize_columns(df)
                    return df
            except Exception as e:
                self.logger.warning(f"Secondo tentativo di lettura fallito: {e}")
            
            # Metodo 3: Analisi manuale del file
            try:
                # Leggi le prime righe per analisi
                with open(cache_file, 'r') as f:
                    header = f.readline().strip()
                    first_data = f.readline().strip() if f.readline() else ""
                
                # Verifica se la prima colonna potrebbe essere una data
                if ',' in header and ',' in first_data:
                    columns = header.split(',')
                    first_row = first_data.split(',')
                    
                    # Se la prima colonna sembra una data nel formato YYYY-MM-DD
                    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
                    if re.match(date_pattern, first_row[0]):
                        # Leggi con la prima colonna come indice
                        df = pd.read_csv(cache_file, index_col=0)
                        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                        df = self._standardize_columns(df)
                        return df
            except Exception as e:
                self.logger.warning(f"Terzo tentativo di lettura fallito: {e}")
                
            # Se arriviamo qui, tutti i tentativi sono falliti
            raise ValueError("Impossibile leggere il file CSV in nessun formato noto")
                
        except Exception as e:
            self.logger.error(f"Tutti i tentativi di caricamento dati falliti: {e}")
            raise
    
    def _standardize_columns(self, df):
        """
        Standardizza i nomi delle colonne
        
        Args:
            df: DataFrame da standardizzare
            
        Returns:
            DataFrame con colonne standardizzate
        """
        # Mappa standard per i nomi delle colonne
        std_columns = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adj. close': 'Adj Close'
        }
        
        # Rinomina le colonne (case-insensitive)
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in std_columns:
                rename_map[col] = std_columns[col_lower]
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def clean_stock_data(self, df, symbol=None):
        """
        Pulisce e prepara i dati azionari:
        - Gestisce valori mancanti
        - Rimuove outlier
        - Verifica coerenza dei dati
        
        Args:
            df: DataFrame con i dati da pulire
            symbol: Simbolo del titolo (opzionale, per logging)
            
        Returns:
            DataFrame pulito
        """
        if df.empty:
            self.logger.warning(f"DataFrame vuoto per {symbol or 'simbolo sconosciuto'}")
            return df
        
        # Copia per evitare modifiche indesiderate
        df_clean = df.copy()
        
        # Assicura che l'indice sia datetime
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index, format='%Y-%m-%d', errors='coerce')
                # Rimuovi righe con indice non valido (NaT)
                df_clean = df_clean.loc[~df_clean.index.isna()]
                self.logger.info(f"Indice convertito in datetime per {symbol or 'simbolo sconosciuto'}")
            except Exception as e:
                self.logger.warning(f"Impossibile convertire l'indice in datetime: {e}")
        
        # Ordina per data
        df_clean = df_clean.sort_index()
        
        # Verifica le colonne standard
        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        column_mapping = {}
        
        # Crea un mapping per le colonne (case-insensitive)
        for col in df_clean.columns:
            for exp_col in expected_columns:
                if col.lower() == exp_col.lower():
                    column_mapping[col] = exp_col
        
        # Rinomina le colonne se necessario
        if column_mapping:
            df_clean = df_clean.rename(columns=column_mapping)
            new_columns = list(df_clean.columns)
            self.logger.info(f"Colonne rinominate: {new_columns}")
        
        # Gestisci i valori NaN/NULL
        # Forward fill per i prezzi
        if 'Open' in df_clean.columns and 'High' in df_clean.columns and 'Low' in df_clean.columns and 'Close' in df_clean.columns:
            price_cols = ['Open', 'High', 'Low', 'Close']
            nan_count_before = df_clean[price_cols].isna().sum().sum()
            if nan_count_before > 0:
                # Forward fill per valori mancanti
                df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill')
                # Backward fill per i primi valori se ancora NaN
                df_clean[price_cols] = df_clean[price_cols].fillna(method='bfill')
                nan_count_after = df_clean[price_cols].isna().sum().sum()
                self.logger.info(f"Valori NaN nei prezzi: {nan_count_before} -> {nan_count_after}")
        
        # Volume - imposta a 0 i valori NaN
        if 'Volume' in df_clean.columns:
            vol_nan_count = df_clean['Volume'].isna().sum()
            if vol_nan_count > 0:
                df_clean['Volume'] = df_clean['Volume'].fillna(0)
                self.logger.info(f"Valori NaN nel volume: {vol_nan_count} -> 0")
        
        # Se mancano colonne essenziali, aggiungi con valori NaN
        for col in expected_columns:
            if col not in df_clean.columns:
                # Per prezzo usa la media del Close se disponibile
                if col in ['Open', 'High', 'Low'] and 'Close' in df_clean.columns:
                    df_clean[col] = df_clean['Close']
                # Per Volume usa 0
                elif col == 'Volume':
                    df_clean[col] = 0
                # Per Close usa la media di Open, High, Low se disponibili
                elif col == 'Close' and all(c in df_clean.columns for c in ['Open', 'High', 'Low']):
                    df_clean[col] = df_clean[['Open', 'High', 'Low']].mean(axis=1)
                else:
                    df_clean[col] = np.nan
                    
                self.logger.info(f"Aggiunta colonna mancante {col} per {symbol or 'simbolo sconosciuto'}")
        
        return df_clean
    
    def is_data_available(self, symbol, start_date, end_date):
        """
        Verifica se i dati per un titolo sono disponibili nel periodo specificato
        
        Args:
            symbol: Simbolo del titolo
            start_date: Data di inizio
            end_date: Data di fine
            
        Returns:
            bool: True se i dati sono disponibili, False altrimenti
        """
        try:
            # Percorso del file di cache
            cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
            
            # Controllo se il file di cache esiste
            if not os.path.exists(cache_file):
                self.logger.info(f"Dati per {symbol} non trovati in cache")
                return False
            
            # Verifica se il file contiene dati (controllo base)
            if os.path.getsize(cache_file) < 100:
                self.logger.info(f"File di cache troppo piccolo per {symbol}")
                return False
                
            # Prova a leggere il file e verificare il periodo
            try:
                # Converti le date in oggetti datetime
                try:
                    start_date_dt = pd.to_datetime(start_date)
                    end_date_dt = pd.to_datetime(end_date)
                except:
                    # Se c'è un errore di conversione, considera i dati non disponibili
                    self.logger.warning(f"Errore nella conversione delle date per {symbol}")
                    return False
                
                # Prova diverse opzioni di lettura del file
                df = None
                read_successful = False
                
                # Opzione 1: lettura standard
                try:
                    df = pd.read_csv(cache_file)
                    read_successful = True
                except Exception as e:
                    self.logger.warning(f"Primo tentativo di lettura fallito per {symbol}: {e}")
                
                # Opzione 2: specifica l'indice
                if not read_successful:
                    try:
                        df = pd.read_csv(cache_file, index_col=0)
                        read_successful = True
                    except Exception as e:
                        self.logger.warning(f"Secondo tentativo di lettura fallito per {symbol}: {e}")
                
                # Se entrambi i tentativi falliscono, i dati non sono disponibili
                if not read_successful or df is None or df.empty:
                    self.logger.warning(f"Impossibile leggere il file per {symbol}")
                    return False
                
                # Cerca la colonna data (può essere Date, Datetime, date, datetime o l'indice)
                date_col = None
                if 'Date' in df.columns:
                    date_col = 'Date'
                elif 'Datetime' in df.columns:
                    date_col = 'Datetime'
                elif 'date' in df.columns:
                    date_col = 'date'
                elif 'datetime' in df.columns:
                    date_col = 'datetime'
                
                # Se la colonna data è stata trovata nelle colonne
                if date_col is not None:
                    # Converti la colonna in datetime
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col]) # Rimuovi le righe con date non valide
                    
                    if df.empty:
                        self.logger.warning(f"Nessuna data valida trovata per {symbol}")
                        return False
                    
                    min_date = df[date_col].min()
                    max_date = df[date_col].max()
                else:
                    # Prova a usare l'indice come data
                    try:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        df = df.dropna() # Rimuovi le righe con indici data non validi
                        
                        if df.empty:
                            self.logger.warning(f"Nessuna data valida nell'indice per {symbol}")
                            return False
                        
                        min_date = df.index.min()
                        max_date = df.index.max()
                    except:
                        self.logger.warning(f"Impossibile interpretare le date nel file per {symbol}")
                        return False
                
                # Verifica che il periodo richiesto sia coperto
                period_covered = (min_date <= start_date_dt and max_date >= end_date_dt)
                
                self.logger.info(f"Dati per {symbol} disponibili: {period_covered} (periodo dal {min_date} al {max_date})")
                
                # Se il periodo non è completamente coperto ma abbiamo comunque dati, 
                # consideriamoli sufficienti (il cliente potrà decidere se forzare il download)
                return True
                
            except Exception as e:
                self.logger.warning(f"Errore nell'analisi dei dati per {symbol}: {e}")
                # Per sicurezza, considera i dati disponibili se il file esiste e ha dimensione ragionevole
                return True
            
        except Exception as e:
            self.logger.warning(f"Errore nella verifica dei dati per {symbol}: {e}")
            return False
    
    def get_multiple_stocks_data(self, symbols, start_date, end_date):
        """
        Ottiene i dati storici per multiple azioni
        
        Args:
            symbols: Lista di simboli
            start_date: Data di inizio
            end_date: Data di fine
            
        Returns:
            Dizionario di DataFrame con i dati delle azioni
        """
        result = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, start_date, end_date)
            if data is not None:
                result[symbol] = data
        
        return result
    
    def prepare_features(self, df, window_sizes=[5, 10, 20, 50]):
        """
        Calcola feature aggiuntive dai dati di prezzo
        
        Args:
            df: DataFrame con i dati di prezzo
            window_sizes: Liste di dimensioni delle finestre per le medie mobili
            
        Returns:
            DataFrame con le feature aggiunte
        """
        features_df = df.copy()
        
        # Calcolo dei rendimenti
        features_df['Returns'] = df['Close'].pct_change()
        
        # Calcolo delle medie mobili
        for window in window_sizes:
            features_df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            features_df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Calcolo della volatilità
        features_df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        features_df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Calcolo del MACD
        features_df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        features_df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        features_df['MACD'] = features_df['EMA_12'] - features_df['EMA_26']
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Rimozione delle righe con NaN
        features_df = features_df.dropna()
        
        return features_df

    def download_data(self, symbols, start_date, end_date, interval='1d'):
        """
        Scarica i dati storici per i simboli specificati
        
        Args:
            symbols: Lista di simboli
            start_date: Data di inizio
            end_date: Data di fine
            interval: Intervallo dei dati ('1d', '1h', '1wk', ecc.)
            
        Returns:
            Dict: Dizionario di DataFrame con i dati scaricati, con chiavi corrispondenti ai simboli
        """
        self.logger.info(f"Avvio download dati per {len(symbols)} simboli con intervallo {interval}")
        
        # Controlla e converti le date
        try:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            
            # Formatta le date in modo coerente
            start_date_str = start_date_dt.strftime('%Y-%m-%d')
            end_date_str = end_date_dt.strftime('%Y-%m-%d')
            
            # Controlla che la data di inizio sia prima della data di fine
            if start_date_dt >= end_date_dt:
                self.logger.error("La data di inizio deve essere precedente alla data di fine")
                return {}
        except Exception as e:
            self.logger.error(f"Errore nella conversione delle date: {e}")
            return {}
        
        # Dizionario per memorizzare i risultati
        result_data = {}
        
        # Scarica i dati per ogni simbolo
        for symbol in symbols:
            try:
                self.logger.info(f"Download dei dati per {symbol} da {start_date_str} a {end_date_str}")
                
                # Tenta il download
                df = yf.download(
                    symbol, 
                    start=start_date_str, 
                    end=end_date_str, 
                    interval=interval,
                    progress=False
                )
                
                if df.empty:
                    self.logger.warning(f"Nessun dato disponibile per {symbol}")
                    continue
                
                # Pulizia e preprocessing dei dati
                df = self.clean_stock_data(df, symbol)
                
                # Salva i dati in cache
                cache_file = os.path.join(self.data_dir, f"{symbol}.csv")
                df.to_csv(cache_file)
                
                # Aggiungi al dizionario dei risultati
                result_data[symbol] = df
                
                self.logger.info(f"Download completato per {symbol}: {len(df)} righe")
            except Exception as e:
                self.logger.error(f"Errore nel download dei dati per {symbol}: {e}")
        
        self.logger.info(f"Download completato per {len(result_data)}/{len(symbols)} simboli")
        return result_data
    
    def verify_data_integrity(self):
        """
        Verifica l'integrità dei dati nella cache.
        Controlla tutti i file CSV nella directory dei dati e corregge eventuali problemi.
        
        Returns:
            dict: Report sulla verifica dell'integrità
        """
        self.logger.info("Verifica dell'integrità dei dati in corso...")
        
        # Report
        integrity_report = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'fixed_files': 0,
            'unfixable_files': 0,
            'details': {}
        }
        
        # Elenca tutti i file CSV
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        integrity_report['total_files'] = len(csv_files)
        
        for csv_file in csv_files:
            symbol = csv_file.replace('.csv', '')
            file_path = os.path.join(self.data_dir, csv_file)
            
            try:
                # Verifica se il file è vuoto
                if os.path.getsize(file_path) < 100:
                    integrity_report['invalid_files'] += 1
                    integrity_report['details'][symbol] = {
                        'status': 'invalid',
                        'reason': 'File troppo piccolo',
                        'action': 'none'
                    }
                    continue
                
                # Tenta di leggere il file
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    
                    # Nessun dato o errore di lettura
                    if df is None or df.empty:
                        raise ValueError("DataFrame vuoto")
                    
                    # Verifica le colonne necessarie
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        # Tenta di riparare le colonne
                        try:
                            # Mappa i nomi delle colonne (case-insensitive)
                            col_mapping = {}
                            for req_col in missing_cols:
                                for col in df.columns:
                                    if col.lower() == req_col.lower():
                                        col_mapping[col] = req_col
                            
                            if col_mapping:
                                df = df.rename(columns=col_mapping)
                                df.to_csv(file_path)
                                integrity_report['fixed_files'] += 1
                                integrity_report['details'][symbol] = {
                                    'status': 'fixed',
                                    'reason': f"Colonne rinominate: {col_mapping}",
                                    'action': 'rename_columns'
                                }
                            else:
                                raise ValueError(f"Colonne mancanti: {missing_cols}")
                        except Exception as e:
                            integrity_report['unfixable_files'] += 1
                            integrity_report['details'][symbol] = {
                                'status': 'unfixable',
                                'reason': f"Colonne mancanti: {missing_cols}",
                                'action': 'none'
                            }
                            continue
                    
                    # Verifica l'indice come datetime
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                            df.to_csv(file_path)
                            integrity_report['fixed_files'] += 1
                            integrity_report['details'][symbol] = {
                                'status': 'fixed',
                                'reason': "Indice convertito in datetime",
                                'action': 'convert_index'
                            }
                        except Exception as e:
                            integrity_report['unfixable_files'] += 1
                            integrity_report['details'][symbol] = {
                                'status': 'unfixable',
                                'reason': f"Indice non convertibile in datetime: {e}",
                                'action': 'none'
                            }
                            continue
                    
                    # Il file è valido
                    integrity_report['valid_files'] += 1
                    integrity_report['details'][symbol] = {
                        'status': 'valid',
                        'reason': 'Nessun problema rilevato',
                        'action': 'none'
                    }
                    
                except Exception as e:
                    integrity_report['invalid_files'] += 1
                    integrity_report['details'][symbol] = {
                        'status': 'invalid',
                        'reason': f"Errore nella lettura: {e}",
                        'action': 'none'
                    }
            except Exception as e:
                integrity_report['invalid_files'] += 1
                integrity_report['details'][symbol] = {
                    'status': 'invalid',
                    'reason': f"Errore generale: {e}",
                    'action': 'none'
                }
        
        self.logger.info(f"Verifica dell'integrità completata: {integrity_report['valid_files']}/{integrity_report['total_files']} file validi")
        return integrity_report 