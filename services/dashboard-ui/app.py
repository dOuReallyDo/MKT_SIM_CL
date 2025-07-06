import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import json
import time

# Configurazione della pagina
st.set_page_config(
    page_title="MKT SIM - Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Configurazione delle API
API_GATEWAY = "http://api-gateway:8000"
MARKET_SIMULATOR = "http://market-simulator:8001"
AI_ENGINE = "http://ai-engine:8002"
DATA_PIPELINE = "http://data-pipeline:8003"

# Funzioni di utilità
def get_market_data(symbol, date):
    try:
        response = requests.get(f"{MARKET_SIMULATOR}/market/data/{date}")
        return response.json()
    except:
        return None

def start_simulation(config):
    try:
        response = requests.post(f"{API_GATEWAY}/simulation", json=config)
        return response.json()
    except:
        return None

def get_model_predictions(symbol):
    try:
        response = requests.get(f"{AI_ENGINE}/predict/{symbol}")
        return response.json()
    except:
        return None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150", width=100)
    st.title("MKT SIM")
    
    selected = option_menu(
        menu_title="Menu",
        options=["Dashboard", "Simulazione", "Analisi", "Configurazione"],
        icons=["house", "play-circle", "graph-up", "gear"],
        menu_icon="cast",
        default_index=0,
    )

# Contenuto principale
if selected == "Dashboard":
    st.title("📊 Dashboard Principale")
    
    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Simulazioni Attive", value="3", delta="+1")
    with col2:
        st.metric(label="Performance Media", value="12.5%", delta="2.3%")
    with col3:
        st.metric(label="Modelli Attivi", value="5", delta="0")
    with col4:
        st.metric(label="Dati Processati", value="1.2M", delta="150K")
    
    # Grafico principale
    st.subheader("Andamento Mercato")
    chart_data = pd.DataFrame(
        {
            "Data": pd.date_range(start="2024-01-01", periods=30),
            "AAPL": pd.Series(range(30)).apply(lambda x: 150 + x * 2 + np.random.normal(0, 5)),
            "GOOGL": pd.Series(range(30)).apply(lambda x: 2800 + x * 3 + np.random.normal(0, 10)),
        }
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data["Data"], y=chart_data["AAPL"], name="AAPL"))
    fig.add_trace(go.Scatter(x=chart_data["Data"], y=chart_data["GOOGL"], name="GOOGL"))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Ultime transazioni
    st.subheader("Ultime Transazioni")
    transactions = pd.DataFrame({
        "Data": pd.date_range(start="2024-03-01", periods=5),
        "Simbolo": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
        "Tipo": ["Acquisto", "Vendita", "Acquisto", "Vendita", "Acquisto"],
        "Quantità": [100, 50, 200, 75, 150],
        "Prezzo": [175.5, 2800.0, 380.0, 175.0, 480.0]
    })
    st.dataframe(transactions, use_container_width=True)

elif selected == "Simulazione":
    st.title("🎮 Nuova Simulazione")
    
    with st.form("simulation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbols = st.multiselect(
                "Seleziona i simboli",
                ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA"],
                default=["AAPL", "GOOGL"]
            )
            
            start_date = st.date_input(
                "Data di inizio",
                datetime.now() - timedelta(days=30)
            )
        
        with col2:
            initial_capital = st.number_input(
                "Capitale iniziale",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=1000
            )
            
            end_date = st.date_input(
                "Data di fine",
                datetime.now()
            )
        
        submitted = st.form_submit_button("Avvia Simulazione")
        
        if submitted:
            config = {
                "symbols": symbols,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_capital": initial_capital
            }
            
            with st.spinner("Avvio simulazione..."):
                result = start_simulation(config)
                if result:
                    st.success("Simulazione avviata con successo!")
                    st.json(result)
                else:
                    st.error("Errore nell'avvio della simulazione")

elif selected == "Analisi":
    st.title("📈 Analisi e Previsioni")
    
    tab1, tab2, tab3 = st.tabs(["Previsioni", "Performance", "Correlazioni"])
    
    with tab1:
        st.subheader("Previsioni AI")
        symbol = st.selectbox("Seleziona simbolo", ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
        
        if st.button("Genera Previsioni"):
            with st.spinner("Generazione previsioni..."):
                predictions = get_model_predictions(symbol)
                if predictions:
                    st.success("Previsioni generate con successo!")
                    st.json(predictions)
                else:
                    st.error("Errore nella generazione delle previsioni")
    
    with tab2:
        st.subheader("Performance Modelli")
        # Implementazione grafico performance
        
    with tab3:
        st.subheader("Analisi Correlazioni")
        # Implementazione matrice correlazioni

elif selected == "Configurazione":
    st.title("⚙️ Configurazione")
    
    tab1, tab2, tab3 = st.tabs(["API", "Modelli", "Sistema"])
    
    with tab1:
        st.subheader("Configurazione API")
        api_key = st.text_input("API Key", type="password")
        if st.button("Salva Configurazione API"):
            st.success("Configurazione salvata!")
    
    with tab2:
        st.subheader("Configurazione Modelli")
        model_type = st.selectbox("Tipo Modello", ["LSTM", "Random Forest", "XGBoost"])
        training_period = st.slider("Periodo di Training (giorni)", 30, 365, 90)
        
        if st.button("Aggiorna Configurazione Modelli"):
            st.success("Configurazione modelli aggiornata!")
    
    with tab3:
        st.subheader("Configurazione Sistema")
        auto_refresh = st.checkbox("Auto-refresh dati", value=True)
        refresh_interval = st.slider("Intervallo refresh (minuti)", 1, 60, 5)
        
        if st.button("Salva Configurazione Sistema"):
            st.success("Configurazione sistema salvata!")

# Footer
st.markdown("---")
st.markdown("MKT SIM - Dashboard v1.0 | © 2024") 