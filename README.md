<<<<<<< HEAD
# MKT_SIM_CL
=======
# MKT_SIM_CL - Sistema di Simulazione del Mercato Azionario

Sistema di simulazione del mercato azionario con trading algoritmico basato su IA.

## Caratteristiche

- Simulazione realistica del mercato azionario
- Supporto per multiple strategie di trading
- Integrazione con reti neurali per le previsioni
- Dashboard interattiva per l'analisi
- Gestione sicura delle credenziali
- Sistema di caching per ottimizzare le performance
- Test automatizzati e CI/CD

## Requisiti

- Python 3.8 o superiore
- Dipendenze elencate in `requirements.txt`

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/yourusername/mkt_sim_cl.git
cd mkt_sim_cl
```

2. Crea un ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Configura le variabili d'ambiente:
```bash
cp .env.example .env
# Modifica .env con le tue credenziali
```

## Utilizzo

1. Avvia la simulazione:
```bash
python main.py
```

2. Accedi alla dashboard:
```bash
python dashboard/app.py
```

## Documentazione

La documentazione completa è disponibile nella directory `docs/`:
- [API Documentation](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Development Guide](docs/development.md)

## Testing

Esegui i test:
```bash
pytest
```

Per la copertura del codice:
```bash
pytest --cov=./ --cov-report=term-missing
```

## Contribuire

1. Fork il repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Licenza

Questo progetto è sotto licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.

## Contatti

Mario Curcio - mario.curcio@example.com

Link del progetto: [https://github.com/yourusername/mkt_sim_cl](https://github.com/yourusername/mkt_sim_cl) 
>>>>>>> 0aa259e37 (Primo commit del progetto MKT_SIM_CL)
