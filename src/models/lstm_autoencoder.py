import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers,
                               batch_first=True, dropout=dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h_n, _) = self.encoder(x)
        latent = self.encoder_fc(h_n[-1])              # (batch, latent_dim)
        
        decoded_input = self.decoder_fc(latent)
        decoded_input = decoded_input.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(decoded_input)
        return out

def create_sequences(data: np.ndarray, seq_len=48) -> np.ndarray:
    """Slide a window of seq_len over the time series."""
    return np.array([data[i:i+seq_len] for i in range(len(data) - seq_len)])

def train_lstm_autoencoder(df_pandas, seq_len=48, epochs=50, lr=1e-3):
    from sklearn.preprocessing import StandardScaler
    
    mlflow.set_experiment("wind-turbine-anomaly")
    
    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.log_params({
            "model": "LSTMAutoencoder",
            "seq_len": seq_len,
            "epochs": epochs,
            "lr": lr
        })
        
        scaler = StandardScaler()
        X = scaler.fit_transform(df_pandas[FEATURES].dropna().values)
        sequences = create_sequences(X, seq_len)
        
        # Train on NORMAL data only (key for reconstruction-based anomaly detection)
        # If you have labels, filter: sequences_normal = sequences[labels == 0]
        
        dataset = torch.FloatTensor(sequences)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = LSTMAutoencoder(input_dim=len(FEATURES)).to(device)
        optim  = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch in loader:
                batch = batch.to(device)
                optim.zero_grad()
                recon = model(batch)
                loss  = loss_fn(recon, batch)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}")
        
        mlflow.pytorch.log_model(model, "lstm_autoencoder_model",
                                  registered_model_name="WindTurbineLSTMAE")
        
        return model, scaler

def get_reconstruction_errors(model, sequences_tensor, device="cpu"):
    model.eval()
    with torch.no_grad():
        recon = model(sequences_tensor.to(device)).cpu().numpy()
    errors = np.mean((sequences_tensor.numpy() - recon) ** 2, axis=(1, 2))
    return errors