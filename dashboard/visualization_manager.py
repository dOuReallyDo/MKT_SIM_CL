import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

class VisualizationManager:
    """Gestisce le visualizzazioni in tempo reale per la dashboard"""
    
    def __init__(self):
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura il logging per il gestore delle visualizzazioni"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/visualization_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('VisualizationManager')
    
    def create_candlestick_chart(self, df: pd.DataFrame, title: str = "Candlestick Chart") -> Dict[str, Any]:
        """Crea un grafico candlestick per i dati di mercato"""
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            
            fig.update_layout(
                title=title,
                yaxis_title='Prezzo',
                xaxis_title='Data',
                template='plotly_dark'
            )
            
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico candlestick: {e}")
            return {}
    
    def create_portfolio_chart(self, portfolio_data: Dict[str, float], title: str = "Portfolio Composition") -> Dict[str, Any]:
        """Crea un grafico a torta per la composizione del portafoglio"""
        try:
            fig = px.pie(
                values=list(portfolio_data.values()),
                names=list(portfolio_data.keys()),
                title=title
            )
            
            fig.update_layout(template='plotly_dark')
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico del portafoglio: {e}")
            return {}
    
    def create_performance_chart(self, performance_data: pd.DataFrame, title: str = "Performance") -> Dict[str, Any]:
        """Crea un grafico lineare per le performance"""
        try:
            fig = px.line(
                performance_data,
                x=performance_data.index,
                y=performance_data.columns,
                title=title
            )
            
            fig.update_layout(template='plotly_dark')
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico delle performance: {e}")
            return {}
    
    def create_volume_chart(self, df: pd.DataFrame, title: str = "Volume") -> Dict[str, Any]:
        """Crea un grafico a barre per i volumi"""
        try:
            fig = go.Figure(data=[go.Bar(
                x=df.index,
                y=df['Volume']
            )])
            
            fig.update_layout(
                title=title,
                yaxis_title='Volume',
                xaxis_title='Data',
                template='plotly_dark'
            )
            
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico dei volumi: {e}")
            return {}
    
    def create_neural_network_metrics(self, metrics: Dict[str, List[float]], title: str = "Neural Network Metrics") -> Dict[str, Any]:
        """Crea un grafico per le metriche della rete neurale"""
        try:
            fig = go.Figure()
            
            for metric_name, values in metrics.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    name=metric_name,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title=title,
                yaxis_title='Value',
                xaxis_title='Epoch',
                template='plotly_dark'
            )
            
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico delle metriche: {e}")
            return {}
    
    def create_prediction_comparison(self, actual: pd.Series, predicted: pd.Series, title: str = "Prediction vs Actual") -> Dict[str, Any]:
        """Crea un grafico di confronto tra previsioni e valori reali"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=actual,
                name='Actual',
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                y=predicted,
                name='Predicted',
                mode='lines'
            ))
            
            fig.update_layout(
                title=title,
                yaxis_title='Value',
                xaxis_title='Time',
                template='plotly_dark'
            )
            
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico di confronto: {e}")
            return {}
    
    def create_market_depth_chart(self, bids: pd.DataFrame, asks: pd.DataFrame, title: str = "Market Depth") -> Dict[str, Any]:
        """Crea un grafico di profondità di mercato"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=bids['price'],
                y=bids['cumulative_volume'],
                name='Bids',
                fill='tozeroy'
            ))
            
            fig.add_trace(go.Scatter(
                x=asks['price'],
                y=asks['cumulative_volume'],
                name='Asks',
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title=title,
                yaxis_title='Cumulative Volume',
                xaxis_title='Price',
                template='plotly_dark'
            )
            
            return fig.to_dict()
        except Exception as e:
            self.logger.error(f"Errore nella creazione del grafico di profondità: {e}")
            return {}
    
    def create_heatmap(self, data, x_labels=None, y_labels=None, title='Mappa di Calore', colorscale='Viridis'):
        """
        Crea una mappa di calore
        
        Args:
            data: Matrice dei dati da visualizzare
            x_labels: Etichette per l'asse x
            y_labels: Etichette per l'asse y
            title: Titolo del grafico
            colorscale: Scala di colori da utilizzare
            
        Returns:
            Grafico in formato plotly
        """
        try:
            import plotly.graph_objects as go
            
            # Se non sono specificate etichette, usa indici numerici
            if x_labels is None:
                x_labels = [f"X{i}" for i in range(len(data[0]) if data else 0)]
            if y_labels is None:
                y_labels = [f"Y{i}" for i in range(len(data) if data else 0)]
            
            # Crea la figura
            fig = go.Figure(data=go.Heatmap(
                z=data,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                hoverongaps=False
            ))
            
            # Imposta il layout
            fig.update_layout(
                title=title,
                xaxis_title="X",
                yaxis_title="Y",
                height=600,
                width=800
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Errore nella creazione della mappa di calore: {e}")
            return None
    
    def create_correlation_matrix(self, data, symbols=None, title='Matrice di Correlazione'):
        """
        Crea una matrice di correlazione tra i simboli
        
        Args:
            data: Dizionario con i dati dei prezzi {symbol: [prices]}
            symbols: Lista dei simboli da includere
            title: Titolo del grafico
            
        Returns:
            Grafico in formato plotly
        """
        try:
            import pandas as pd
            import numpy as np
            import plotly.graph_objects as go
            
            # Se non sono specificati simboli, usa tutti quelli nei dati
            if symbols is None:
                symbols = list(data.keys())
            
            # Crea un DataFrame con i prezzi
            df = pd.DataFrame()
            for symbol in symbols:
                if symbol in data:
                    # Utilizza i prezzi di chiusura
                    if isinstance(data[symbol], dict) and 'close' in data[symbol]:
                        df[symbol] = data[symbol]['close']
                    # Oppure utilizza direttamente la lista dei prezzi
                    elif isinstance(data[symbol], list):
                        df[symbol] = data[symbol]
            
            # Calcola la matrice di correlazione
            corr_matrix = df.corr()
            
            # Crea la figura
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',  # Red-Blue scale (rosso per correlazione negativa, blu per positiva)
                zmin=-1,
                zmax=1,
                hoverongaps=False
            ))
            
            # Imposta il layout
            fig.update_layout(
                title=title,
                height=600,
                width=800
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Errore nella creazione della matrice di correlazione: {e}")
            return None
    
    def create_scatter_matrix(self, data, symbols=None, title='Scatter Matrix'):
        """
        Crea una matrice di scatter plot per confronto visivo
        
        Args:
            data: Dizionario con i dati dei prezzi {symbol: [prices]}
            symbols: Lista dei simboli da includere (max 4 consigliati)
            title: Titolo del grafico
            
        Returns:
            Grafico in formato plotly
        """
        try:
            import pandas as pd
            import plotly.express as px
            
            # Se non sono specificati simboli, usa tutti quelli nei dati
            if symbols is None:
                symbols = list(data.keys())
            
            # Limita il numero di simboli per evitare grafici troppo complessi
            symbols = symbols[:6]  # Massimo 6 simboli
            
            # Crea un DataFrame con i prezzi
            df = pd.DataFrame()
            for symbol in symbols:
                if symbol in data:
                    # Utilizza i prezzi di chiusura
                    if isinstance(data[symbol], dict) and 'close' in data[symbol]:
                        df[symbol] = data[symbol]['close']
                    # Oppure utilizza direttamente la lista dei prezzi
                    elif isinstance(data[symbol], list):
                        df[symbol] = data[symbol]
            
            # Crea la scatter matrix
            fig = px.scatter_matrix(
                df,
                dimensions=symbols,
                title=title
            )
            
            # Imposta il layout
            fig.update_layout(
                height=800,
                width=900
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Errore nella creazione della scatter matrix: {e}")
            return None 