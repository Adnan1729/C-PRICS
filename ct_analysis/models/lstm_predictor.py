# LSTMPredictor and RiskTrajectoryPredictor classes

class LSTMPredictor(nn.Module):
    """LSTM model for risk trajectory prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Predict risk score
        prediction = self.fc(last_output)
        return prediction

class RiskTrajectoryPredictor:
    """Manager class for risk trajectory prediction"""
    def __init__(self, sequence_length=5, hidden_size=64, num_layers=2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.model = None
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def prepare_data(self, combined_summary_df):
        """Prepare dataset from combined summary DataFrame"""
        self.dataset = IslandSequenceDataset(
            combined_summary_df, 
            sequence_length=self.sequence_length
        )
        
        # Initialize model
        input_size = len(self.dataset.features)
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

    def train(self, num_epochs=50, batch_size=32):
        """Train the LSTM model"""
        if self.dataset is None or self.model is None:
            raise ValueError("Call prepare_data first")

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def predict_trajectory(self, island_data, future_steps=90):
        """
        Predict future risk trajectory for an island
        
        Parameters:
        -----------
        island_data : pd.DataFrame
            Historical data for the island
        future_steps : int, optional (default=90)
            Number of days to predict into the future
            
        Returns:
        --------
        np.ndarray
            Array of predicted risk scores
        """
        try:
            self.model.eval()
            
            # Sort by date and get recent sequence
            island_data = island_data.sort_values('date')
            recent_data = island_data.iloc[-self.sequence_length:]
            
            # Ensure numerical values
            for col in self.dataset.features:
                if col in recent_data.columns:
                    recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
            
            # Scale the data
            scaler = MinMaxScaler()
            feature_data = recent_data[self.dataset.features].astype(float)
            scaled_data = scaler.fit_transform(feature_data)
            
            # Convert to tensor
            sequence = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
            
            predictions = []
            with torch.no_grad():
                current_sequence = sequence
                
                for _ in range(future_steps):
                    # Predict next risk score
                    output = self.model(current_sequence)
                    pred_risk = float(output.item())  # Convert to float explicitly
                    predictions.append(pred_risk)
                    
                    # Update sequence for next prediction
                    new_sequence = current_sequence.clone()
                    new_sequence = new_sequence[:, 1:, :]  # Remove oldest timestep
                    
                    # Create new feature vector using last known values and predicted risk
                    new_features = new_sequence[:, -1, :].clone()
                    new_features[:, -1] = pred_risk  # Update risk score
                    
                    # Add new timestep
                    new_sequence = torch.cat([
                        new_sequence,
                        new_features.unsqueeze(1)
                    ], dim=1)
                    
                    current_sequence = new_sequence
            
            # Convert predictions to numpy array
            predictions = np.array(predictions, dtype=np.float32)
            
            # Inverse transform predictions
            predictions_2d = predictions.reshape(-1, 1)
            padding = np.zeros((len(predictions), len(self.dataset.features)-1))
            padded_predictions = np.hstack([padding, predictions_2d])
            inverse_transformed = scaler.inverse_transform(padded_predictions)[:, -1]
            
            # Ensure predictions are within valid range [0, 1] and are float type
            return np.clip(inverse_transformed, 0, 1).astype(np.float32)
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return np.array([], dtype=np.float32)
