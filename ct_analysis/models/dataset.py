# IslandSequenceDataset class

class IslandSequenceDataset(Dataset):
    """Dataset for handling multiple island sequences"""
    def __init__(self, data, sequence_length=5):
        self.sequence_length = sequence_length
        self.scalers = {}
        self.sequences = []
        self.targets = []
        
        # Group data by global_id
        grouped = data.groupby('global_id')
        
        # Features to use for prediction
        self.features = [
            'peak_amplitude', 'pattern_width', 'area_under_curve',
            'mean_slope', 'max_slope', 'risk_score'
        ]
        
        # Process each island's data
        for island_id, island_data in grouped:
            if len(island_data) >= sequence_length + 1:  # Need at least sequence_length + 1 points
                # Sort by date
                island_data = island_data.sort_values('date')
                
                # Scale features for this island
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(island_data[self.features])
                self.scalers[island_id] = scaler
                
                # Create sequences
                for i in range(len(scaled_data) - sequence_length):
                    self.sequences.append(scaled_data[i:i+sequence_length])
                    self.targets.append(scaled_data[i+sequence_length, -1])  # risk_score is last feature
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.targets = torch.FloatTensor(np.array(self.targets))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
