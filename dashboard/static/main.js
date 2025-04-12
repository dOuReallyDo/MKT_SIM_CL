// Dashboard main script
document.addEventListener('DOMContentLoaded', function() {
    console.log("DEBUG: DOM fully loaded and parsed"); // Debug log 1

    // Enable popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl)
    })

    // Enable tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // Flash messages auto close
    window.setTimeout(function() {
        $(".alert-dismissible").fadeTo(500, 0).slideUp(500, function(){
            $(this).remove();
        });
    }, 4000);

    // Add confirm dialogs
    var confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            if (!confirm(this.getAttribute('data-confirm'))) {
                e.preventDefault();
            }
        });
    });

    // Modifica gestione submit form e pulsanti (SPOSTATO QUI DENTRO DOMCONTENTLOADED)
    const simulationForm = document.getElementById('simulation-form');
    const runSimulationBtn = document.getElementById('run-simulation-btn');
    const stopSimulationBtn = document.getElementById('stop-simulation-btn');

    console.log("DEBUG: simulationForm =", simulationForm); // Debug log 2
    console.log("DEBUG: runSimulationBtn =", runSimulationBtn); // Debug log 3
    console.log("DEBUG: stopSimulationBtn =", stopSimulationBtn); // Debug log 4

    // Nascondi il pulsante stop all'inizio
    if(stopSimulationBtn) stopSimulationBtn.style.display = 'none';

    // Inizializza il grafico al caricamento della pagina (se siamo nel tab giusto)
    const simTabContent = document.getElementById('simulation-tab-content');
    if (simTabContent && simTabContent.classList.contains('active')) {
         initOrUpdatePortfolioChart();
    }

    // NUOVO APPROCCIO: Aggiungi event listener direttamente al pulsante invece che al form
    if (runSimulationBtn) {
        console.log("DEBUG: Attaching click listener to #run-simulation-btn"); // Debug log 5
        runSimulationBtn.addEventListener('click', function(e) {
            e.preventDefault(); // Previeni il comportamento predefinito del pulsante
            console.log("DEBUG: #run-simulation-btn clicked!"); // Debug log 6

            // Recupera il form
            const form = this.closest('form');
            if (!form) {
                console.error("DEBUG: Form not found!"); // Debug log 7
                return;
            }

            // Pulisci grafico precedente
            initOrUpdatePortfolioChart();

            // Recupera parametri dal form
            const numAgents = document.getElementById('num-agents').value;
            const initialCapital = document.getElementById('initial-capital').value;
            const strategy = document.getElementById('strategy').value;

            const payload = {
                num_agents: parseInt(numAgents, 10),
                initial_capital: parseFloat(initialCapital),
                strategy: strategy
            };
            console.log("DEBUG: Sending payload to /run_simulation:", payload); // Debug log 8

            // Mostra messaggio di avvio
            const statusMessageEl = document.getElementById('simulation-status-message');
            if(statusMessageEl) {
                statusMessageEl.textContent = 'Avvio simulazione in corso...';
                statusMessageEl.className = 'alert alert-warning';
            }
            // Aggiorna UI pulsanti
            if(runSimulationBtn) runSimulationBtn.disabled = true;
            if(stopSimulationBtn) stopSimulationBtn.style.display = 'inline-block';
            if(stopSimulationBtn) stopSimulationBtn.disabled = false;

            // Invia richiesta POST al backend
            fetch('/run_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            })
            .then(response => {
                console.log("DEBUG: Response received:", response); // Debug log 9
                return response.json();
            })
            .then(data => {
                console.log('DEBUG: Response data from /run_simulation:', data); // Debug log 10
                if (data.status === 'started') {
                    // La simulazione è avviata, il monitoraggio inizierà a inviare dati
                    if(statusMessageEl) {
                        statusMessageEl.textContent = 'Simulazione avviata, in attesa di dati...';
                        statusMessageEl.className = 'alert alert-info';
                    }
                } else {
                    // Errore nell'avvio
                    showError(data.error || 'Errore sconosciuto nell\'avvio della simulazione');
                    if(statusMessageEl) {
                        statusMessageEl.textContent = `Errore avvio: ${data.error}`;
                        statusMessageEl.className = 'alert alert-danger';
                    }
                     // Ripristina UI
                    if(runSimulationBtn) runSimulationBtn.disabled = false;
                    if(stopSimulationBtn) stopSimulationBtn.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('DEBUG: Error in fetch /run_simulation:', error); // Debug log 11
                showError('Errore di comunicazione con il server.');
                 if(statusMessageEl) {
                        statusMessageEl.textContent = 'Errore di comunicazione.';
                        statusMessageEl.className = 'alert alert-danger';
                    }
                 // Ripristina UI
                if(runSimulationBtn) runSimulationBtn.disabled = false;
                if(stopSimulationBtn) stopSimulationBtn.style.display = 'none';
            });
        });
    } else {
        console.warn("DEBUG: #run-simulation-btn not found!"); // Debug log 12
    }

    // Manteniamo anche l'event listener originale sul form per compatibilità
    if (simulationForm) {
        console.log("DEBUG: Attaching submit listener to #simulation-form"); // Debug log 13
        simulationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log("DEBUG: #simulation-form submitted!"); // Debug log 14

            // Pulisci grafico precedente
            initOrUpdatePortfolioChart();

            // Recupera parametri dal form
            const numAgents = document.getElementById('num-agents').value;
            const initialCapital = document.getElementById('initial-capital').value;
            const strategy = document.getElementById('strategy').value;

            const payload = {
                num_agents: parseInt(numAgents, 10),
                initial_capital: parseFloat(initialCapital),
                strategy: strategy
            };
            console.log("DEBUG: Sending payload to /run_simulation (from form):", payload); // Debug log 15

            // Mostra messaggio di avvio
            const statusMessageEl = document.getElementById('simulation-status-message');
            if(statusMessageEl) {
                statusMessageEl.textContent = 'Avvio simulazione in corso...';
                statusMessageEl.className = 'alert alert-warning';
            }
            // Aggiorna UI pulsanti
            if(runSimulationBtn) runSimulationBtn.disabled = true;
            if(stopSimulationBtn) stopSimulationBtn.style.display = 'inline-block';
            if(stopSimulationBtn) stopSimulationBtn.disabled = false;

            // Invia richiesta POST al backend
            fetch('/run_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            })
            .then(response => {
                console.log("DEBUG: Form submit response received:", response); // Debug log 16
                return response.json();
            })
            .then(data => {
                console.log('DEBUG: Form submit response data:', data); // Debug log 17
                if (data.status === 'started') {
                    // La simulazione è avviata, il monitoraggio inizierà a inviare dati
                    if(statusMessageEl) {
                        statusMessageEl.textContent = 'Simulazione avviata, in attesa di dati...';
                        statusMessageEl.className = 'alert alert-info';
                    }
                } else {
                    // Errore nell'avvio
                    showError(data.error || 'Errore sconosciuto nell\'avvio della simulazione');
                    if(statusMessageEl) {
                        statusMessageEl.textContent = `Errore avvio: ${data.error}`;
                        statusMessageEl.className = 'alert alert-danger';
                    }
                     // Ripristina UI
                    if(runSimulationBtn) runSimulationBtn.disabled = false;
                    if(stopSimulationBtn) stopSimulationBtn.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('DEBUG: Form submit fetch error:', error); // Debug log 18
                showError('Errore di comunicazione con il server.');
                 if(statusMessageEl) {
                        statusMessageEl.textContent = 'Errore di comunicazione.';
                        statusMessageEl.className = 'alert alert-danger';
                    }
                 // Ripristina UI
                if(runSimulationBtn) runSimulationBtn.disabled = false;
                if(stopSimulationBtn) stopSimulationBtn.style.display = 'none';
            });
        });
    } else {
        console.warn("DEBUG: #simulation-form not found!"); // Debug log 19
    }

    // Gestione stop simulazione
    if (stopSimulationBtn) {
        console.log("DEBUG: Attaching click listener to #stop-simulation-btn"); // Debug log 20
        stopSimulationBtn.addEventListener('click', function() {
            console.log("DEBUG: #stop-simulation-btn clicked!"); // Debug log 21
            // Disabilita il pulsante per evitare click multipli
            this.disabled = true;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Interruzione...';

            // Invia richiesta POST al backend
            fetch('/stop_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('Risposta da /stop_simulation:', data);
                if (data.status === 'stop_requested') {
                    // Messaggio di conferma, lo stato effettivo verrà da websocket
                    console.log("Richiesta di interruzione inviata con successo.");
                    // Potremmo aggiornare il testo dello status qui, ma aspettiamo il websocket
                     const statusMessageEl = document.getElementById('simulation-status-message');
                     if(statusMessageEl) {
                         statusMessageEl.textContent = 'Interruzione richiesta, attendere...';
                         statusMessageEl.className = 'alert alert-warning';
                     }
                } else {
                    // Errore nella richiesta
                    showError(data.message || 'Errore sconosciuto nell\'invio della richiesta di interruzione.');
                     // Riabilita il pulsante se la richiesta fallisce
                    this.disabled = false;
                    this.innerHTML = '<i class="fas fa-stop"></i> Interrompi Simulazione';
                }
                // Nota: Non riabilitiamo il pulsante 'Run' qui,
                // l'aggiornamento dello stato via websocket gestirà la UI finale.
            })
            .catch(error => {
                console.error('Errore fetch /stop_simulation:', error);
                showError('Errore di comunicazione con il server per interrompere la simulazione.');
                // Riabilita il pulsante in caso di errore network
                this.disabled = false;
                this.innerHTML = '<i class="fas fa-stop"></i> Interrompi Simulazione';
            });
        });
    } else {
         console.warn("DEBUG: #stop-simulation-btn not found!"); // Debug log 22
    }

    // Gestione cambio tab per inizializzare grafico se necessario
    document.querySelectorAll('.nav-link[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function (event) {
            if (event.target.getAttribute('href') === '#simulation-tab-content') { // Adatta se l'ID del contenuto è diverso
                initOrUpdatePortfolioChart();
            }
        });
    });

}); // Fine DOMContentLoaded

// Gestione degli eventi WebSocket
// Inizializza socket come proprietà globale window se non esiste già
if (typeof window.socket === 'undefined') {
    window.socket = io();
}

window.socket.on('connect', function() {
    console.log('Connesso al server WebSocket');
    $('#connection-status').html('<i class="fas fa-circle text-success"></i> Connected');
    // Sottoscrivi al tab attivo all'avvio o al cambio tab
    const activeTabLink = document.querySelector('.nav-link.active');
    const activeTabId = activeTabLink ? activeTabLink.getAttribute('href').substring(1) : 'index'; // Default a index o altro
    if (activeTabId) {
        window.socket.emit('subscribe', { tab: activeTabId });
        console.log(`Subscribed to tab: ${activeTabId}`);
    }
});

window.socket.on('disconnect', function() {
    console.log('Disconnesso dal server WebSocket');
    $('#connection-status').html('<i class="fas fa-circle text-danger"></i> Disconnected');
});

window.socket.on('dashboard_state', function(data) {
    console.log('Stato dashboard aggiornato:', data);
    updateDashboardState(data);
});

window.socket.on('tab_state', function(data) {
    console.log('Stato tab aggiornato:', data);
    updateTabState(data);
});

window.socket.on('error', function(data) {
    console.error('Errore:', data);
    showError(data.message || data.error || 'Errore sconosciuto dal server'); // Mostra messaggio o errore
});

window.socket.on('progress', function(data) {
    console.log('Progresso:', data);
    updateProgress(data);
});

// *** NUOVO LISTENER PER AGGIORNAMENTI SIMULAZIONE ***
window.socket.on('market_simulation_update', function(data) {
    // console.log('Aggiornamento Simulazione di Mercato:', data); // Loggato dentro la funzione
    // Chiama una nuova funzione per aggiornare la UI specifica della simulazione
    updateRealTimeSimulationUI(data);
});

// Funzioni di aggiornamento
function updateDashboardState(data) {
    // Potrebbe essere vuota se non ci sono elementi globali da aggiornare
}

function updateTabState(data) {
    // Questa funzione potrebbe non essere più necessaria se ogni tab gestisce i propri aggiornamenti
    // tramite eventi specifici come 'market_simulation_update'
    console.log('Received tab_state update (potentially deprecated):', data);
}

function updateDataCollectionState(data) {
    // Aggiorna lo stato della raccolta dati (se necessario)
}

function updateSimulationState(data) {
    // Deprecato in favore di updateRealTimeSimulationUI
    console.warn("updateSimulationState called (deprecated)");
}

function updateNeuralNetworkState(data) {
    // Aggiorna lo stato della rete neurale (se necessario)
}

function updateSelfPlayState(data) {
    // Aggiorna lo stato del self-play (se necessario)
}

function updatePredictionsState(data) {
    // Aggiorna lo stato delle previsioni (se necessario)
}

function updateReportsState(data) {
    // Aggiorna lo stato dei report (se necessario)
}

function updateConfigState(data) {
    // Aggiorna lo stato della configurazione (se necessario)
}

function updateProgress(data) {
    // Aggiorna la barra di progresso generale (se esiste)
}

function showError(message) {
    // Mostra un messaggio di errore
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) {
        console.error("Contenitore alert 'alerts-container' non trovato!");
        return;
    }
    const alert = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    alertsContainer.innerHTML += alert; // Usa += per aggiungere, non sovrascrivere
}

// Gestione dei tab (SPOSTATO DENTRO DOMCONTENTLOADED)
// $('.nav-link').on('click', function(e) { ... }); // jQuery non più usato qui

// Gestione del pulsante refresh (SPOSTATO DENTRO DOMCONTENTLOADED)
// $('#refresh-btn').on('click', function() { ... }); // jQuery non più usato qui

// Gestione del pulsante help (SPOSTATO DENTRO DOMCONTENTLOADED)
// $('#help-btn').on('click', function() { ... }); // jQuery non più usato qui

// Inizializzazione (SPOSTATO DENTRO DOMCONTENTLOADED)
// $(document).ready(function() { ... }); // jQuery non più usato qui

// *** NUOVA FUNZIONE PER AGGIORNARE LA UI CON DATI REAL-TIME ***
function updateRealTimeSimulationUI(data) {
    // Verifica se siamo effettivamente nel tab di simulazione
    const simulationTab = document.getElementById('simulation-tab-content'); // ID del contenuto del tab
    if (!simulationTab || !simulationTab.classList.contains('active')) {
        // console.log("Non nel tab simulazione, ignoro l'aggiornamento UI.");
        return;
    }

    console.log("Aggiornando UI Simulazione con dati:", data);

    const runBtn = document.getElementById('run-simulation-btn');
    const stopBtn = document.getElementById('stop-simulation-btn');
    const statusMessageEl = document.getElementById('simulation-status-message');

    // Aggiorna Status e Progresso
    if (data.status) {
        if (statusMessageEl) {
            statusMessageEl.textContent = `Stato: ${data.status}`;
            statusMessageEl.className = 'alert '; // Reset classi
            if (data.status === 'running') {
                statusMessageEl.classList.add('alert-info');
                if (stopBtn) stopBtn.style.display = 'inline-block';
                if (stopBtn) stopBtn.disabled = false; // Riabilita se era in interruzione
                if (stopBtn) stopBtn.innerHTML = '<i class="fas fa-stop"></i> Interrompi Simulazione'; // Reset testo
                if (runBtn) runBtn.style.display = 'none';
            } else if (data.status === 'completed') {
                statusMessageEl.classList.add('alert-success');
                if (stopBtn) stopBtn.style.display = 'none';
                if (runBtn) runBtn.style.display = 'inline-block';
                if (runBtn) runBtn.disabled = false; // Riabilita run
            } else if (data.status === 'error') {
                statusMessageEl.classList.add('alert-danger');
                statusMessageEl.textContent += ` - ${data.error || 'Errore sconosciuto'}`;
                if (stopBtn) stopBtn.style.display = 'none';
                if (runBtn) runBtn.style.display = 'inline-block';
                if (runBtn) runBtn.disabled = false; // Riabilita run
            } else if (data.status === 'stopped') { // Aggiunto stato 'stopped'
                 statusMessageEl.classList.add('alert-warning');
                 statusMessageEl.textContent = 'Simulazione Interrotta.';
                 if (stopBtn) stopBtn.style.display = 'none';
                 if (runBtn) runBtn.style.display = 'inline-block';
                 if (runBtn) runBtn.disabled = false; // Riabilita run
            } else { // Idle o altro
                statusMessageEl.classList.add('alert-secondary');
                if (stopBtn) stopBtn.style.display = 'none';
                if (runBtn) runBtn.style.display = 'inline-block';
                if (runBtn) runBtn.disabled = false; // Riabilita run
            }
        }
    }
    if (data.progress !== undefined) {
        const progressPercent = Math.round(data.progress);
        const progressBar = document.getElementById('simulation-progress');
        if (progressBar) {
            progressBar.style.width = `${progressPercent}%`;
            progressBar.textContent = `${progressPercent}%`;
            progressBar.setAttribute('aria-valuenow', progressPercent);
        }
        const currentDayEl = document.getElementById('current-date');
        if (currentDayEl) currentDayEl.textContent = data.current_day || '-';

        const currentStepEl = document.getElementById('simulation-current-step');
        if (currentStepEl) currentStepEl.textContent = `${data.current_day_idx || 0} / ${data.total_days || 0}`;
    }

    // Aggiorna KPI Generali
    const agentsCountEl = document.getElementById('agents-count');
    if (agentsCountEl) agentsCountEl.textContent = data.agents ? data.agents.length : 0;

    const transactionsCountEl = document.getElementById('transactions-count');
    if (transactionsCountEl) transactionsCountEl.textContent = data.transactions_count !== undefined ? data.transactions_count : '0';


    // Aggiorna Tabella Agenti
    const agentsTable = document.getElementById('agents-table');
    const agentsTableBody = agentsTable ? agentsTable.getElementsByTagName('tbody')[0] : null;
    if (agentsTableBody) {
        agentsTableBody.innerHTML = ''; // Pulisce la tabella prima di ripopolarla
        if (data.agents && data.agents.length > 0) {
            data.agents.forEach(agent => {
                const pnl = agent.performance ? agent.performance.pnl : 0;
                const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${agent.id}</td>
                    <td>${agent.strategy || 'N/A'}</td>
                    <td>${formatCurrency(agent.cash)}</td>
                    <td>${formatCurrency(agent.performance ? agent.performance.portfolio_value : 0)}</td>
                    <td>${formatCurrency(agent.performance ? agent.performance.current_value : 0)}</td>
                    <td class="${pnlClass}">${formatCurrency(pnl)}</td>
                    <td class="${pnlClass}">${formatPercent(agent.performance ? agent.performance.percentage_return : 0)}</td>
                    <td>${agent.performance ? agent.performance.transactions_count : 0}</td>
                    <td><button class="btn btn-sm btn-info btn-agent-details" data-agent-id="${agent.id}">Dettagli</button></td>
                `;
                // Aggiungi event listener al pulsante Dettagli
                const detailsButton = row.querySelector('.btn-agent-details');
                if (detailsButton) {
                    detailsButton.addEventListener('click', function() {
                        showAgentDetails(this.getAttribute('data-agent-id'));
                    });
                }
                agentsTableBody.appendChild(row);
            });
        } else {
            agentsTableBody.innerHTML = '<tr><td colspan="9" class="text-center">In attesa di dati agenti...</td></tr>';
        }
    } else {
        console.warn("Elemento tbody della tabella agenti non trovato.");
    }

    // Aggiorna Tabella Transazioni Recenti
    const transactionsTable = document.getElementById('transactions-table');
    const transactionsTableBody = transactionsTable ? transactionsTable.getElementsByTagName('tbody')[0] : null;
    if(transactionsTableBody && data.recent_transactions) {
        transactionsTableBody.innerHTML = ''; // Pulisci
        if(data.recent_transactions.length > 0) {
            const transactionsToShow = data.recent_transactions.slice(-10).reverse(); // Mostra le ultime 10, più recenti prima
            transactionsToShow.forEach(tx => {
                const row = document.createElement('tr');
                const actionClass = tx.action === 'buy' ? 'text-success' : (tx.action === 'sell' ? 'text-danger' : '');
                row.innerHTML = `
                    <td>${tx.date || 'N/A'}</td>
                    <td>${tx.agent_id !== undefined ? tx.agent_id : 'N/A'}</td>
                    <td>${tx.symbol || 'N/A'}</td>
                    <td class="${actionClass}">${tx.action || 'N/A'}</td>
                    <td>${tx.quantity || 'N/A'}</td>
                    <td>${formatCurrency(tx.price)}</td>
                    <td>${formatCurrency(tx.total)}</td>
                `;
                transactionsTableBody.appendChild(row);
            });
        } else {
             transactionsTableBody.innerHTML = '<tr><td colspan="7" class="text-center">Nessuna transazione recente.</td></tr>';
        }
    } else if (transactionsTableBody) {
        // Mantieni un messaggio di default se non ci sono dati e la simulazione non è attiva
        if (!data.status || data.status === 'idle' || data.status === 'completed' || data.status === 'error' || data.status === 'stopped') {
             if (transactionsTableBody.rows.length === 0 || transactionsTableBody.rows[0].cells.length === 1) { // Evita sovrascrittura se ci sono già dati
                 transactionsTableBody.innerHTML = '<tr><td colspan="7" class="text-center">Nessuna transazione registrata.</td></tr>';
             }
        }
    }

    // Aggiorna Grafico Portafoglio
    if (window.portfolioChart && data.chart_point) {
        updatePortfolioChart(data.chart_point.date, data.chart_point.value);
    }
}

// Funzioni di utilità per la formattazione
function formatCurrency(value) {
    if (value === undefined || value === null || isNaN(value)) return 'N/A';
    try {
        return value.toLocaleString('it-IT', { style: 'currency', currency: 'EUR', minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } catch(e) {
        return `€${Number(value).toFixed(2)}`;
    }
}

function formatPercent(value) {
     if (value === undefined || value === null || isNaN(value)) return 'N/A';
     return `${(Number(value) * 100).toFixed(2)}%`;
}

// Funzione placeholder per dettagli agente
function showAgentDetails(agentId) {
    alert(`Visualizza dettagli per Agente ${agentId} (logica da implementare)`);
    // Qui potresti aprire un modal o caricare dati specifici per l'agente via API/WebSocket
}

// Funzione per inizializzare o aggiornare il grafico del portafoglio
let portfolioChart = null; // Riferimento globale al grafico

function initOrUpdatePortfolioChart(chartId = 'portfolio-chart') {
    const chartElement = document.getElementById(chartId);
    const ctx = chartElement ? chartElement.getContext('2d') : null;
    if (!ctx) {
        console.error(`Elemento canvas con ID ${chartId} non trovato.`);
        return;
    }

    // Dati di esempio iniziali o pulizia
    const initialData = {
        labels: [],
        datasets: [{
            label: 'Valore Totale Portafoglio Agenti',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            fill: false
        }]
    };

    if (portfolioChart) {
        // Se il grafico esiste già, pulisci i dati
        portfolioChart.data.labels = initialData.labels;
        portfolioChart.data.datasets = initialData.datasets;
        portfolioChart.update();
        console.log("Grafico esistente pulito.");
    } else {
        // Crea un nuovo grafico
        portfolioChart = new Chart(ctx, {
            type: 'line',
            data: initialData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        // type: 'time', // Rimosso type: 'time' per semplicità, le etichette sono stringhe
                        // time: { unit: 'day' },
                        title: { display: true, text: 'Data/Step Simulazione' }
                    },
                    y: {
                        title: { display: true, text: 'Valore (€)' },
                        ticks: {
                            callback: function(value) { return formatCurrency(value); }
                        }
                    }
                },
                plugins: {
                    legend: { display: true },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) label += formatCurrency(context.parsed.y);
                                return label;
                            }
                        }
                    }
                }
            }
        });
        console.log("Nuovo grafico inizializzato.");
    }
}

// Funzione per aggiornare il grafico del portafoglio con nuovi dati
function updatePortfolioChart(date, value) {
    if (!portfolioChart) {
        console.error("Grafico non inizializzato.");
        return;
    }
    
    // Aggiungi il nuovo punto dati
    portfolioChart.data.labels.push(date);
    portfolioChart.data.datasets[0].data.push(value);
    
    // Limita il numero di punti visualizzati (opzionale)
    const maxPoints = 50; // Numero massimo di punti da visualizzare
    if (portfolioChart.data.labels.length > maxPoints) {
        portfolioChart.data.labels = portfolioChart.data.labels.slice(-maxPoints);
        portfolioChart.data.datasets[0].data = portfolioChart.data.datasets[0].data.slice(-maxPoints);
    }
    
    // Aggiorna il grafico
    portfolioChart.update();
}
