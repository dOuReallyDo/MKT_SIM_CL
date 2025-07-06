FROM python:3.11-slim

WORKDIR /app

# Aggiorna e installa le dipendenze di sistema con retry e fix-missing
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    for i in 1 2 3; do \
      apt-get update -o Acquire::CompressionTypes::Order::=gz && \
      apt-get install -y --no-install-recommends --fix-missing build-essential && break || sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*

# Copia e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY . .

EXPOSE 8501

# Avvia l'applicazione Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 