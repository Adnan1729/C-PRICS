# CTFeatureExtractor class

class CTFeatureExtractor:
    def __init__(self, data, location_col='Location_Norm_m', value_col='Value', 
                 island_id_col='island_id', interpolation_points=100):
        self.data = data
        self.location_col = location_col
        self.value_col = value_col
        self.island_id_col = island_id_col
        self.interpolation_points = interpolation_points
        self.features = {}

    def fit_spline(self, x, y):
        """Fit cubic spline to the data"""
        # Use keyword arguments for CubicSpline
        cs = CubicSpline(x=x, y=y)
        x_new = np.linspace(start=x.min(), stop=x.max(), num=self.interpolation_points)
        y_new = cs(x=x_new)
        return x_new, y_new, cs

    def calculate_features(self, island_id):
        """Calculate features for a specific island"""
        # Get data for this island
        island_data = self.data[self.data[self.island_id_col] == island_id].sort_values(self.location_col)
        x = island_data[self.location_col].values
        y = island_data[self.value_col].values
        
        # Store start and end locations
        start_location = x[0]
        end_location = x[-1]
        
        # Fit spline
        x_interp, y_interp, spline = self.fit_spline(x, y)
        
        # Calculate derivatives
        dy_dx = spline.derivative()(x_interp)
        
        # Calculate features
        features = {
            'island_id': island_id,
            'start_location': start_location,  # Add start location
            'end_location': end_location,      # Add end location
            'peak_amplitude': np.max(y_interp),
            'average_y': np.mean(y_interp),
            'pattern_width': x_interp[-1] - x_interp[0],
            'center_location': x_interp[np.argmax(y_interp)],
            'mean_slope': np.mean(np.abs(dy_dx)),
            'max_slope': np.max(np.abs(dy_dx)),
            'area_under_curve': simpson(y=y_interp, x=x_interp)
        }
        
        # Calculate distance difference
        scalar_distance = np.sqrt((x_interp[-1] - x_interp[0])**2 + 
                                (y_interp[-1] - y_interp[0])**2)
        vector_distance = np.sum(np.sqrt(np.diff(x_interp)**2 + 
                                       np.diff(y_interp)**2))
        features['distance_difference'] = vector_distance - scalar_distance
        
        # Store interpolated values for plotting
        features['x_interp'] = x_interp
        features['y_interp'] = y_interp
        features['dy_dx'] = dy_dx
        features['x_raw'] = x
        features['y_raw'] = y
        
        return features
           
    def extract_all_features(self):
        """Extract features for all islands"""
        for island_id in self.data[self.island_id_col].unique():
            self.features[island_id] = self.calculate_features(island_id)
        return self.features
    
    def plot_all_islands(self, date):
      """Create a single plot showing all islands for a given date"""
      fig = go.Figure()
      
      # Color scale for different risk scores
      colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))  # Red (high risk) to Green (low risk)
      
      for island_id, features in self.features.items():
          risk_score = features.get('risk_score', 0)
          color_idx = int(risk_score * 99)  # Map 0-1 to 0-99
          color = f'rgb({colors[color_idx][0]*255},{colors[color_idx][1]*255},{colors[color_idx][2]*255})'
          
          # Plot original points
          fig.add_trace(
              go.Scatter(
                  x=features['x_raw'],
                  y=features['y_raw'],
                  mode='markers',
                  name=f'Island {island_id} (Risk: {risk_score:.2f})',
                  marker=dict(size=8, color=color)
              )
          )
          
          # Plot spline fit
          fig.add_trace(
              go.Scatter(
                  x=features['x_interp'],
                  y=features['y_interp'],
                  mode='lines',
                  name=f'Spline {island_id}',
                  line=dict(color=color),
                  showlegend=False
              )
          )
      
      fig.update_layout(
          title=f'CT Signal Analysis - {date}',
          xaxis_title='Location (m)',
          yaxis_title='Signal Value',
          showlegend=True,
          height=600,
          width=1200
      )
      
      return fig

