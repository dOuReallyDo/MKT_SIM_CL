"""
Report Generator Module.

Questo modulo fornisce funzionalità per la generazione di report automatici 
in diversi formati (PDF, Excel) a partire dai dati delle simulazioni.
"""

import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non interattivo, necessario per ambienti server
import seaborn as sns
from fpdf import FPDF
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union

class ReportGenerator:
    """Generatore di report automatici"""
    
    def __init__(self, output_dir='./reports'):
        """
        Inizializza il generatore di report
        
        Args:
            output_dir: Directory di output per i report generati
        """
        self.output_dir = output_dir
        
        # Configura il logging
        self.logger = logging.getLogger('ReportGenerator')
        self.logger.setLevel(logging.INFO)
        
        # Se non ci sono handler, aggiungine uno
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Crea la directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Report Generator inizializzato in {output_dir}")
    
    def generate_pdf_report(self, simulation_data: Dict[str, Any], report_id: str) -> str:
        """
        Genera un report PDF dalla simulazione
        
        Args:
            simulation_data: Dati della simulazione
            report_id: Identificatore del report
            
        Returns:
            Percorso del file PDF generato
        """
        try:
            # Crea un nome file basato sull'ID del report e sulla data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{report_id}_{timestamp}.pdf"
            file_path = os.path.join(self.output_dir, file_name)
            
            # Crea il PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Titolo e intestazione
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f"Report di Simulazione: {report_id}", 0, 1, 'C')
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f"Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
            pdf.ln(5)
            
            # Riepilogo della simulazione
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Riepilogo della Simulazione", 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f"Periodo: {simulation_data.get('start_date', 'N/A')} - {simulation_data.get('end_date', 'N/A')}", 0, 1, 'L')
            pdf.cell(0, 10, f"Simboli: {', '.join(simulation_data.get('symbols', []))}", 0, 1, 'L')
            pdf.cell(0, 10, f"Numero di agenti: {len(simulation_data.get('agents', []))}", 0, 1, 'L')
            pdf.cell(0, 10, f"Numero di transazioni: {len(simulation_data.get('transactions', []))}", 0, 1, 'L')
            pdf.ln(5)
            
            # Performance degli agenti
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Performance degli Agenti", 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            # Crea una tabella per gli agenti
            if simulation_data.get('agents'):
                # Intestazione
                pdf.set_fill_color(200, 220, 255)
                pdf.cell(20, 7, "ID", 1, 0, 'C', True)
                pdf.cell(40, 7, "Capitale Iniziale", 1, 0, 'C', True)
                pdf.cell(40, 7, "Capitale Finale", 1, 0, 'C', True)
                pdf.cell(30, 7, "Rendimento (%)", 1, 0, 'C', True)
                pdf.cell(30, 7, "Transazioni", 1, 1, 'C', True)
                
                # Dati
                for agent in simulation_data['agents']:
                    pdf.cell(20, 7, str(agent.get('id', 'N/A')), 1, 0, 'C')
                    pdf.cell(40, 7, self._format_currency(agent.get('initial_capital', 0)), 1, 0, 'C')
                    pdf.cell(40, 7, self._format_currency(agent.get('final_capital', 0)), 1, 0, 'C')
                    rendimento = agent.get('return', 0)
                    rendimento_str = f"{rendimento:+.2f}%" if isinstance(rendimento, (int, float)) else 'N/A'
                    pdf.cell(30, 7, rendimento_str, 1, 0, 'C')
                    pdf.cell(30, 7, str(agent.get('transactions', 0)), 1, 1, 'C')
            
            pdf.ln(5)
            
            # Grafici della performance
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Grafici di Performance", 0, 1, 'L')
            
            # Grafico dell'andamento del portafoglio
            if simulation_data.get('daily_data'):
                # Crea il grafico con matplotlib
                plt.figure(figsize=(10, 6))
                
                # Estrai i dati necessari
                dates = [d.get('date') for d in simulation_data['daily_data']] 
                
                # Per ogni agente, traccia l'andamento del suo portafoglio
                for agent in simulation_data['agents']:
                    agent_id = agent.get('id')
                    # Cerca i valori del portafoglio per questo agente nei dati giornalieri
                    portfolio_values = []
                    for day in simulation_data['daily_data']:
                        agent_data = next((a for a in day.get('agents', []) if a.get('id') == agent_id), None)
                        if agent_data:
                            portfolio_values.append(agent_data.get('portfolio_value', 0))
                        else:
                            portfolio_values.append(None)  # Nessun dato per questo giorno
                    
                    # Traccia la linea
                    if portfolio_values:
                        plt.plot(dates, portfolio_values, label=f"Agente {agent_id}")
                
                plt.title('Andamento del Portafoglio')
                plt.xlabel('Data')
                plt.ylabel('Valore (€)')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Salva il grafico in un buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                plt.close()
                
                # Aggiungi l'immagine al PDF
                pdf.image(img_buffer, x=pdf.get_x(), y=pdf.get_y(), w=180)
                pdf.ln(100)  # Spazio per l'immagine
            
            # Crea una semplice tabella con le transazioni più significative
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Transazioni Significative", 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            # Ordina le transazioni per valore (totale)
            if simulation_data.get('transactions'):
                sorted_transactions = sorted(
                    simulation_data['transactions'],
                    key=lambda x: abs(x.get('total', 0)),
                    reverse=True
                )
                
                # Mostra solo le prime 10
                top_transactions = sorted_transactions[:10]
                
                # Intestazione
                pdf.set_fill_color(200, 220, 255)
                pdf.cell(25, 7, "Data", 1, 0, 'C', True)
                pdf.cell(15, 7, "Agente", 1, 0, 'C', True)
                pdf.cell(20, 7, "Simbolo", 1, 0, 'C', True)
                pdf.cell(25, 7, "Azione", 1, 0, 'C', True)
                pdf.cell(15, 7, "Quantità", 1, 0, 'C', True)
                pdf.cell(25, 7, "Prezzo", 1, 0, 'C', True)
                pdf.cell(30, 7, "Totale", 1, 1, 'C', True)
                
                # Dati
                for tx in top_transactions:
                    pdf.cell(25, 7, str(tx.get('date', 'N/A')), 1, 0, 'C')
                    pdf.cell(15, 7, str(tx.get('agent_id', 'N/A')), 1, 0, 'C')
                    pdf.cell(20, 7, str(tx.get('symbol', 'N/A')), 1, 0, 'C')
                    action = "Acquisto" if tx.get('action') == 'buy' else "Vendita"
                    pdf.cell(25, 7, action, 1, 0, 'C')
                    pdf.cell(15, 7, str(tx.get('quantity', 'N/A')), 1, 0, 'C')
                    pdf.cell(25, 7, self._format_currency(tx.get('price', 0)), 1, 0, 'C')
                    pdf.cell(30, 7, self._format_currency(tx.get('total', 0)), 1, 1, 'C')
            
            # Salva il PDF
            pdf.output(file_path)
            
            self.logger.info(f"Report PDF generato con successo: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Errore nella generazione del report PDF: {e}")
            return ""
    
    def generate_excel_report(self, simulation_data: Dict[str, Any], report_id: str) -> str:
        """
        Genera un report Excel dalla simulazione
        
        Args:
            simulation_data: Dati della simulazione
            report_id: Identificatore del report
            
        Returns:
            Percorso del file Excel generato
        """
        try:
            # Crea un nome file basato sull'ID del report e sulla data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{report_id}_{timestamp}.xlsx"
            file_path = os.path.join(self.output_dir, file_name)
            
            # Crea un file Excel con pandas ExcelWriter
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Foglio di riepilogo
                summary_data = {
                    'Proprietà': [
                        'Report ID',
                        'Data Generazione',
                        'Periodo',
                        'Simboli',
                        'Numero di agenti',
                        'Numero di transazioni'
                    ],
                    'Valore': [
                        report_id,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        f"{simulation_data.get('start_date', 'N/A')} - {simulation_data.get('end_date', 'N/A')}",
                        ', '.join(simulation_data.get('symbols', [])),
                        len(simulation_data.get('agents', [])),
                        len(simulation_data.get('transactions', []))
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Riepilogo', index=False)
                
                # Foglio con i dati degli agenti
                if simulation_data.get('agents'):
                    agents_data = []
                    for agent in simulation_data['agents']:
                        agents_data.append({
                            'ID': agent.get('id', 'N/A'),
                            'Capitale Iniziale': agent.get('initial_capital', 0),
                            'Capitale Finale': agent.get('final_capital', 0),
                            'Rendimento (%)': agent.get('return', 0),
                            'Transazioni': agent.get('transactions', 0),
                            'Strategia': agent.get('strategy', 'N/A')
                        })
                    
                    agents_df = pd.DataFrame(agents_data)
                    agents_df.to_excel(writer, sheet_name='Agenti', index=False)
                
                # Foglio con le transazioni
                if simulation_data.get('transactions'):
                    transactions_data = []
                    for tx in simulation_data['transactions']:
                        transactions_data.append({
                            'Data': tx.get('date', 'N/A'),
                            'Agente': tx.get('agent_id', 'N/A'),
                            'Simbolo': tx.get('symbol', 'N/A'),
                            'Azione': "Acquisto" if tx.get('action') == 'buy' else "Vendita",
                            'Quantità': tx.get('quantity', 0),
                            'Prezzo': tx.get('price', 0),
                            'Totale': tx.get('total', 0)
                        })
                    
                    transactions_df = pd.DataFrame(transactions_data)
                    transactions_df.to_excel(writer, sheet_name='Transazioni', index=False)
                
                # Foglio con i dati giornalieri
                if simulation_data.get('daily_data'):
                    # Estrai i dati aggregati
                    daily_data = []
                    for day in simulation_data['daily_data']:
                        day_record = {
                            'Data': day.get('date', 'N/A')
                        }
                        
                        # Aggiungi i dati di ogni agente
                        for agent in day.get('agents', []):
                            agent_id = agent.get('id', 'N/A')
                            day_record[f'Agente_{agent_id}_Valore'] = agent.get('portfolio_value', 0)
                            day_record[f'Agente_{agent_id}_Cash'] = agent.get('cash', 0)
                        
                        daily_data.append(day_record)
                    
                    daily_df = pd.DataFrame(daily_data)
                    daily_df.to_excel(writer, sheet_name='Dati_Giornalieri', index=False)
                
                # Foglio con le metriche di performance
                performance_data = self._calculate_performance_metrics(simulation_data)
                if performance_data:
                    performance_df = pd.DataFrame([performance_data])
                    performance_df.to_excel(writer, sheet_name='Metriche', index=False)
            
            self.logger.info(f"Report Excel generato con successo: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Errore nella generazione del report Excel: {e}")
            return ""
    
    def _calculate_performance_metrics(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcola le metriche di performance avanzate
        
        Args:
            simulation_data: Dati della simulazione
            
        Returns:
            Dizionario con le metriche calcolate
        """
        try:
            metrics = {}
            
            # Calcola le metriche solo se ci sono dati giornalieri e agenti
            if not simulation_data.get('daily_data') or not simulation_data.get('agents'):
                return metrics
            
            # Calcola le metriche per ogni agente
            for agent in simulation_data['agents']:
                agent_id = agent.get('id')
                
                # Estrai la serie storica del valore del portafoglio
                portfolio_values = []
                for day in simulation_data['daily_data']:
                    agent_data = next((a for a in day.get('agents', []) if a.get('id') == agent_id), None)
                    if agent_data:
                        portfolio_values.append(agent_data.get('portfolio_value', 0))
                
                if not portfolio_values:
                    continue
                
                # Converti in array numpy per i calcoli
                values = np.array(portfolio_values)
                
                # Calcola i rendimenti giornalieri
                returns = np.diff(values) / values[:-1]
                
                # Rendimento medio giornaliero
                avg_daily_return = np.mean(returns) if len(returns) > 0 else 0
                
                # Volatilità (deviazione standard dei rendimenti)
                volatility = np.std(returns) if len(returns) > 0 else 0
                
                # Massimo drawdown
                cum_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cum_returns)
                drawdowns = (cum_returns / running_max) - 1
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                # Sharpe ratio (assumendo un tasso privo di rischio dello 0%)
                sharpe_ratio = (avg_daily_return / volatility) * np.sqrt(252) if volatility > 0 else 0
                
                # Aggiungi le metriche al dizionario
                metrics[f'Agente_{agent_id}_Rendimento_Medio_Giornaliero'] = avg_daily_return
                metrics[f'Agente_{agent_id}_Volatilità'] = volatility
                metrics[f'Agente_{agent_id}_Max_Drawdown'] = max_drawdown
                metrics[f'Agente_{agent_id}_Sharpe_Ratio'] = sharpe_ratio
            
            return metrics
        except Exception as e:
            self.logger.error(f"Errore nel calcolo delle metriche di performance: {e}")
            return {}
    
    def _format_currency(self, value: Union[int, float]) -> str:
        """
        Formatta un valore come valuta
        
        Args:
            value: Valore da formattare
            
        Returns:
            Stringa formattata
        """
        try:
            return f"€ {value:,.2f}".replace(',', '.').replace('.', ',')
        except:
            return str(value) 