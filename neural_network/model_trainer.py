import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging

class DataPreprocessor:
    """Classe per il preprocessing dei dati di mercato"""
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        
        # Configurazione del logging
        self.logger = logging.getLogger('DataPreprocessor')
    
    def prepare_data(self, prices_df, target_symbols=None):
        """
        Prepara i dati per l'addestramento della rete neurale
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            target_symbols: Lista di simboli target (None = tutti)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if target_symbols is None:
            target_symbols = prices_df['symbol'].unique().tolist()
        
        self.logger.info(f"Preparazione dati per {len(target_symbols)} simboli")
        
        # Filtraggio dati
        df = prices_df[prices_df['symbol'].isin(target_symbols)].copy()
        
        # Ordinamento per data
        df = df.sort_values(['symbol', 'date'])
        
        # Normalizzazione
        df_grouped = df.groupby('symbol')
        
        sequences = []
        targets = []
        
        for symbol, group in df_grouped:
            prices = group[['open', 'high', 'low', 'close']].values
            volumes = group['volume'].values.reshape(-1, 1)
            
            # Normalizzazione
            normalized_prices = self.price_scaler.fit_transform(prices)
            normalized_volumes = self.volume_scaler.fit_transform(volumes)
            
            # Creazione sequenze
            for i in range(len(group) - self.sequence_length - 1):
                # Sequenza di input
                seq_prices = normalized_prices[i:i+self.sequence_length]
                seq_volumes = normalized_volumes[i:i+self.sequence_length]
                
                # Target (prezzo di chiusura normalizzato del giorno successivo)
                target = normalized_prices[i+self.sequence_length, 3]  # Indice 3 = prezzo di chiusura
                
                # Combinazione di prezzi e volumi
                seq_combined = np.column_stack((seq_prices, seq_volumes))
                
                sequences.append(seq_combined)
                targets.append(target)
        
        # Conversione in array numpy
        X = np.array(sequences)
        y = np.array(targets)
        
        self.logger.info(f"Creati {len(sequences)} esempi di training")
        
        # Split in train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_price(self, normalized_price):
        """Converte un prezzo normalizzato nel suo valore originale"""
        # Creazione di un array fittizio con tutti i valori necessari
        dummy = np.zeros((1, 4))
        dummy[0, 3] = normalized_price  # Impostiamo il prezzo di chiusura
        
        # Denormalizzazione
        return self.price_scaler.inverse_transform(dummy)[0, 3]


class LSTMModel(nn.Module):
    """Modello LSTM per la previsione dei prezzi"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Inizializzazione dello stato nascosto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagation
        out, _ = self.lstm(x, (h0, c0))
        
        # Prendiamo solo l'output dell'ultimo timestep
        out = self.fc(out[:, -1, :])
        
        return out


class CNNModel(nn.Module):
    """Modello CNN per la previsione dei prezzi"""
    def __init__(self, input_dim, seq_length, num_filters=64, kernel_size=3, dropout=0.2):
        super(CNNModel, self).__init__()
        
        self.cnn1 = nn.Conv1d(input_dim, num_filters, kernel_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)
        
        self.cnn2 = nn.Conv1d(num_filters, num_filters*2, kernel_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2)
        
        # Calcolo della dimensione dell'output dopo le convoluzioni
        def calc_output_size(size, kernel_size, stride=1):
            return (size - kernel_size) // stride + 1
        
        cnn1_out = calc_output_size(seq_length, kernel_size)
        pool1_out = cnn1_out // 2
        cnn2_out = calc_output_size(pool1_out, kernel_size)
        pool2_out = cnn2_out // 2
        
        self.fc_input_dim = num_filters * 2 * pool2_out
        self.fc1 = nn.Linear(self.fc_input_dim, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # Trasposizione per la convoluzione 1D
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_length)
        
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Appiattimento
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerModel(nn.Module):
    """Modello Transformer per la previsione dei prezzi"""
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layer per trasformare l'input in dimensione d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Layer di codifica posizionale
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        
        # Encoder Transformer
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers=num_encoder_layers)
        
        # Layer di output
        self.decoder = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Embedding dell'input
        x = self.input_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Trasposizione per il transformer (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Codifica Transformer
        output = self.transformer_encoder(x)
        
        # Utilizzo dell'output dell'ultimo timestep
        output = output[-1, :, :]  # (batch_size, d_model)
        
        # Layer di output
        output = self.decoder(output)  # (batch_size, 1)
        
        return output


class ModelTrainer:
    """Classe per l'addestramento dei modelli di rete neurale"""
    def __init__(self, model_type='lstm', config=None):
        """
        Inizializza il trainer del modello
        
        Args:
            model_type: Tipo di modello ('lstm', 'cnn', 'transformer')
            config: Configurazione del modello e dell'addestramento
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.data_preprocessor = DataPreprocessor(sequence_length=self.config.get('sequence_length', 10))
        
        # Configurazione dell'hardware
        use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.config.get('use_mps', True)
        use_cuda = torch.cuda.is_available() and self.config.get('use_cuda', False)
        
        if use_mps:
            self.device = torch.device('mps')
        elif use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Configurazione del logging
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("./logs/model_trainer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'ModelTrainer_{model_type}')
        self.logger.info(f"Inizializzato trainer per modello {model_type} su dispositivo {self.device}")
    
    def create_model(self, input_dim):
        """Crea il modello specificato"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=self.config.get('hidden_dim', 64),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'cnn':
            self.model = CNNModel(
                input_dim=input_dim,
                seq_length=self.config.get('sequence_length', 10),
                num_filters=self.config.get('num_filters', 64),
                kernel_size=self.config.get('kernel_size', 3),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=input_dim,
                d_model=self.config.get('d_model', 64),
                nhead=self.config.get('nhead', 8),
                num_encoder_layers=self.config.get('num_encoder_layers', 6),
                dropout=self.config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Tipo di modello non supportato: {self.model_type}")
        
        self.model.to(self.device)
        self.logger.info(f"Creato modello {self.model_type}")
        return self.model
    
    def train(self, prices_df, target_symbols=None):
        """
        Addestra il modello sui dati forniti
        
        Args:
            prices_df: DataFrame con i prezzi delle azioni
            target_symbols: Lista di simboli target (None = tutti)
            
        Returns:
            Dizionario con metriche di addestramento
        """
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.info(f"Avvio addestramento modello {self.model_type}")
        
        # Verifica che 'Returns' sia nel dataframe - se non c'è, calcoliamolo
        if 'Returns' not in prices_df.columns:
            try:
                # Raggruppa per simbolo e calcola i rendimenti
                prices_df = prices_df.copy()
                grouped = prices_df.groupby('symbol')
                
                all_dfs = []
                for name, group in grouped:
                    group = group.copy()
                    group['Returns'] = group['Close'].pct_change()
                    all_dfs.append(group)
                
                prices_df = pd.concat(all_dfs)
                prices_df = prices_df.dropna()
            except Exception as e:
                self.logger.error(f"Impossibile calcolare i rendimenti: {e}")
                # Se ci sono problemi, continua comunque senza Returns
                if 'Close' in prices_df.columns:
                    prices_df['Returns'] = 0.0
        
        try:
            # Preparazione dei dati
            X_train, X_test, y_train, y_test = self.data_preprocessor.prepare_data(prices_df, target_symbols)
            
            # Conversione in tensori PyTorch
            X_train = torch.FloatTensor(X_train).to(self.device)
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)
            
            # Dimensione dell'input
            input_dim = X_train.shape[2]
            seq_length = X_train.shape[1]
            
            # Creazione del modello
            self.model = self.create_model(input_dim)
            self.model.to(self.device)
            
            # Definizione di loss e optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
            
            # Parametri di addestramento
            num_epochs = self.config.get('num_epochs', 100)
            batch_size = self.config.get('batch_size', 32)
            
            # Monitoraggio delle perdite
            epoch_losses = []
            test_losses = []
            
            # Ciclo di addestramento
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                
                # Addestramento a batch
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / (len(X_train) / batch_size)
                epoch_losses.append(avg_loss)
                
                # Valutazione sul test set
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(X_test)
                    test_loss = criterion(test_outputs.squeeze(), y_test).item()
                    test_losses.append(test_loss)
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoca {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}")
            
            # Valutazione finale
            self.model.eval()
            with torch.no_grad():
                final_outputs = self.model(X_test)
                y_pred = final_outputs.squeeze().cpu().numpy()
                y_true = y_test.cpu().numpy()
                
                # Metriche
                mse = ((y_pred - y_true) ** 2).mean()
                rmse = np.sqrt(mse)
            
            # Salvataggio del modello
            self._save_model()
            
            # Plot delle curve di addestramento
            self._plot_training_curves(epoch_losses, test_losses)
            
            self.logger.info(f"Addestramento completato. MSE: {mse:.6f}, RMSE: {rmse:.6f}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'epoch_losses': epoch_losses,
                'test_losses': test_losses
            }
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_model(self):
        """Salva il modello addestrato"""
        if self.model is None:
            self.logger.warning("Nessun modello da salvare")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/{self.model_type}_model_{timestamp}.pt"
        
        # Salvataggio del modello
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.config
        }, model_path)
        
        self.logger.info(f"Modello salvato in {model_path}")
    
    def load_model(self, model_path):
        """Carica un modello salvato"""
        if not os.path.exists(model_path):
            self.logger.error(f"File del modello non trovato: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Estrazione dei parametri
            self.model_type = checkpoint['model_type']
            self.config = checkpoint['config']
            
            # Creazione del modello
            # Poiché non conosciamo l'input_dim, creiamo un modello fittizio
            # che verrà sostituito al momento della previsione
            self.model = None
            self.logger.info(f"Modello {self.model_type} caricato da {model_path}")
            
            # Salvataggio dello state_dict per caricarlo in seguito
            self.model_state_dict = checkpoint['model_state_dict']
            
            return True
        except Exception as e:
            self.logger.error(f"Errore nel caricamento del modello: {e}")
            return False
    
    def predict(self, data):
        """
        Esegue una previsione con il modello
        
        Args:
            data: Array numpy con i dati di input (sequenza di prezzi e volumi)
            
        Returns:
            Prezzo previsto (denormalizzato)
        """
        if self.model is None:
            self.logger.error("Modello non inizializzato")
            return None
        
        try:
            # Normalizzazione dei dati
            # Qui assumiamo che i dati siano già normalizzati o che il modello sia stato addestrato sui dati grezzi
            
            # Creazione di un batch con un solo esempio
            X = torch.FloatTensor(data).unsqueeze(0).to(self.device)
            
            # Previsione
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X).item()
            
            # Denormalizzazione
            prediction_denorm = self.data_preprocessor.inverse_transform_price(prediction)
            
            return prediction_denorm
        except Exception as e:
            self.logger.error(f"Errore nella previsione: {e}")
            return None
    
    def _plot_training_curves(self, train_losses, test_losses):
        """Visualizza le curve di addestramento"""
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {self.model_type.upper()} Model')
        plt.legend()
        plt.grid(True)
        
        # Salvataggio del grafico
        os.makedirs('./reports/figures', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'./reports/figures/training_curves_{self.model_type}_{timestamp}.png')
        plt.close() 