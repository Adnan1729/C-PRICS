# CTAnalyzer class
class CTAnalyzer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.feature_cols = [
            'peak_amplitude', 'average_y', 'distance_difference',
            'pattern_width', 'area_under_curve', 'mean_slope', 'max_slope'
        ]
        # Add feature weights to emphasize important indicators
        self.feature_weights = {
            'peak_amplitude': 0.3,
            'average_y': 0.1,
            'distance_difference': 0.1,
            'pattern_width': 0.15,
            'area_under_curve': 0.15,
            'mean_slope': 0.1,
            'max_slope': 0.1
        }
        self.cluster_risks = None

    def fit(self, combined_summary):
        """Train the analyzer on historical data"""
        try:
            # Extract features
            X = combined_summary[self.feature_cols]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply feature weights
            X_weighted = X_scaled * np.array([self.feature_weights[col] for col in self.feature_cols])
            
            # Fit KMeans
            self.kmeans.fit(X_weighted)
            
            # Calculate cluster risk scores
            self.cluster_risks = self._calculate_cluster_risks(X_weighted)
            
            return self
            
        except Exception as e:
            print(f"Error during fitting: {str(e)}")
            # Set default risk scores if fitting fails
            self.cluster_risks = np.linspace(0, 1, self.n_clusters)
            return self

    def _calculate_cluster_risks(self, X_scaled):
        """Calculate risk scores for each cluster using weighted features"""
        cluster_centers = self.kmeans.cluster_centers_
        feature_importance = np.array([self.feature_weights[col] for col in self.feature_cols])
        
        # Calculate weighted distances from ideal state
        weighted_distances = np.sum(np.abs(cluster_centers) * feature_importance, axis=1)
        
        # Normalize to 0-1 range but maintain sensitivity
        risks = (weighted_distances - weighted_distances.min()) / (weighted_distances.max() - weighted_distances.min())
        
        # Apply sigmoid transformation to spread out the middle range
        risks = 1 / (1 + np.exp(-5 * (risks - 0.5)))
        
        return risks

    def predict_risk(self, features_df):
        """Enhanced risk prediction with temporal smoothing"""
        try:
            X = features_df[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            
            # Apply feature weights
            X_weighted = X_scaled * np.array([self.feature_weights[col] for col in self.feature_cols])
            
            clusters = self.kmeans.predict(X_weighted)
            
            # Calculate base risk scores
            risk_scores = []
            for i, (cluster, point) in enumerate(zip(clusters, X_weighted)):
                # Get base risk from cluster
                base_risk = self.cluster_risks[cluster]
                
                # Calculate point-specific adjustments
                feature_deviations = np.abs(point - self.kmeans.cluster_centers_[cluster])
                weighted_deviations = np.sum(feature_deviations)
                
                # Calculate dynamic adjustment factor
                adjustment = self._sigmoid(weighted_deviations - 0.5) * 0.2  # Max 20% adjustment
                
                # Apply temporal smoothing if we have previous scores
                risk = base_risk + adjustment
                if i > 0:
                    risk = 0.7 * risk + 0.3 * risk_scores[-1]  # Exponential smoothing
                
                risk_scores.append(min(max(risk, 0), 1))  # Ensure 0-1 bounds
            
            return np.array(risk_scores)
            
        except Exception as e:
            print(f"Error during risk prediction: {str(e)}")
            return self._fallback_risk_calculation(features_df)
    
    def _fallback_risk_calculation(self, features_df):
        """More nuanced fallback risk calculation"""
        risks = []
        for _, row in features_df.iterrows():
            # Combine multiple features for risk assessment
            peak_risk = min(row['peak_amplitude'] / 10, 1)
            width_risk = min(row['pattern_width'] / 20, 1)
            slope_risk = min(row['max_slope'] / 5, 1)
            
            # Weighted combination
            combined_risk = (0.5 * peak_risk + 
                           0.3 * width_risk + 
                           0.2 * slope_risk)
            risks.append(combined_risk)
        
        return np.array(risks)

    def _sigmoid(self, x):
        """Sigmoid function for smooth risk adjustments"""
        return 1 / (1 + np.exp(-x))
