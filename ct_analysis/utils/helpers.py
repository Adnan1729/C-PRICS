# Helper functions

def get_sorted_files(input_dir):
    """Get list of CSV files sorted by date in filename"""
    files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Extract date from filename (assuming format YYYY-MM-DD.csv)
            date_str = filename.replace('.csv', '')
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                files.append((date, filename))
            except ValueError:
                print(f"Warning: Couldn't parse date from filename: {filename}")
                continue
    
    # Sort files by date
    files.sort(key=lambda x: x[0])
    return files

def process_file(input_file, output_dir, channel, date_str, analyzer=None, identifier=None):
    """
    Process a single file and save results
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_dir : str
        Path to output directory
    channel : str
        Channel to process
    date_str : str
        Date string for the file
    analyzer : CTAnalyzer, optional
        Analyzer for risk scoring
    identifier : IslandIdentifier, optional
        Island identifier for consistent IDs across dates
    """
    try:
        # Read and process data
        data = pd.read_csv(input_file)
        channel_data = data[data['Channel'] == channel].copy()
        
        if channel_data.empty:
            print(f"No data found for channel {channel} in file {input_file}")
            return None
        
        # Extract features
        extractor = CTFeatureExtractor(channel_data)
        features = extractor.extract_all_features()
        
        # Create summary DataFrame with location bounds
        summary_data = []
        for island_id, feature_dict in features.items():
            # Extract non-array features including start and end locations
            feature_summary = {k: v for k, v in feature_dict.items() 
                             if not isinstance(v, np.ndarray) and 
                             k in ['island_id', 'start_location', 'end_location',
                                  'peak_amplitude', 'average_y', 'pattern_width',
                                  'center_location', 'mean_slope', 'max_slope',
                                  'area_under_curve', 'distance_difference']}
            summary_data.append(feature_summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Add date column
        summary_df['date'] = date_str
        
        # Apply island identification if provided
        if identifier is not None:
            # Register islands and get consistent global IDs
            summary_df = identifier.register_islands(summary_df, date_str)
            
            # Update features with global IDs
            for idx, row in summary_df.iterrows():
                old_id = row['island_id']
                features[old_id]['global_id'] = row['global_id']
                features[old_id]['match_confidence'] = row['match_confidence']
        
        # Calculate risk scores if analyzer is provided
        if analyzer is not None:
            summary_df['risk_score'] = analyzer.predict_risk(summary_df)
            
            # Update features with risk scores
            for idx, row in summary_df.iterrows():
                features[row['island_id']]['risk_score'] = row['risk_score']
        
        # Save summary data
        output_file = os.path.join(output_dir, 'summaries', f'{date_str}_summary.csv')
        summary_df.to_csv(output_file, index=False)
        
        # Generate and save plot
        if hasattr(extractor, 'plot_all_islands'):
            fig = extractor.plot_all_islands(date_str)
            plot_file = os.path.join(output_dir, 'plots', f'{date_str}_analysis.html')
            fig.write_html(plot_file)
        
        return summary_df
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return None

  def analyze_temporal_evolution(combined_summary_file, output_dir):
    """Perform temporal evolution analysis"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print("Reading data...")
    data = pd.read_csv(combined_summary_file)
    
    # Initialize analyzer
    analyzer = TemporalEvolutionAnalyzer()
    
    # Analyze each island
    print("Analyzing island evolution patterns...")
    evolution_results = []
    
    for global_id in data['global_id'].unique():
        island_data = data[data['global_id'] == global_id]
        if len(island_data) >= 3:  # Minimum points for trend analysis
            metrics = analyzer.analyze_island_evolution(island_data)
            metrics['global_id'] = global_id
            
            # Add location information
            metrics['start_location'] = island_data['start_location'].mean()
            metrics['end_location'] = island_data['end_location'].mean()
            evolution_results.append(metrics)
            
            # Create individual evolution plot
            analyzer.plot_island_evolution(island_data, global_id, output_dir)
    
    evolution_df = pd.DataFrame(evolution_results)
    
    # Create spectral evolution plots
    print("Creating spectral evolution plots...")
    for feature in analyzer.features_to_track:
        analyzer.plot_spectral_heatmap(data, feature, output_dir)
    
    # Save evolution metrics
    evolution_df.to_csv(os.path.join(output_dir, 'evolution_metrics.csv'), index=False)
    
    # Create summary visualizations
    create_summary_visualizations(evolution_df, output_dir)
    
    return evolution_df
