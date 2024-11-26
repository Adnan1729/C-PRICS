# Threshold analysis functions

def predict_threshold_crossing(risk_scores, dates, threshold, max_forecast_days=365):
    """
    Predict when risk score will cross a threshold based on historical trend.
    
    Parameters:
        risk_scores: array of historical risk scores
        dates: array of corresponding dates
        threshold: risk score threshold to predict crossing
        max_forecast_days: maximum number of days to forecast into future
    
    Returns:
        predicted_date: datetime or None if threshold won't be crossed within max_forecast_days
    """
    if len(risk_scores) < 2:
        return None
        
    try:
        # Convert dates to numerical days since first observation
        dates_numeric = np.array([(pd.to_datetime(d) - pd.to_datetime(dates[0])).days 
                                for d in dates])
        
        # Fit polynomial regression (degree 2 for better trend capture)
        coeffs = np.polyfit(dates_numeric, risk_scores, deg=2)
        poly = np.poly1d(coeffs)
        
        # Find root of polynomial - threshold equation
        # We're looking for future dates only
        future_days = np.arange(dates_numeric[-1] + 1, 
                              dates_numeric[-1] + max_forecast_days)
        
        future_risks = poly(future_days)
        
        # Find first crossing of threshold
        crossing_idx = np.where(future_risks >= threshold)[0]
        
        if len(crossing_idx) > 0:
            days_until_threshold = future_days[crossing_idx[0]] - dates_numeric[-1]
            predicted_date = pd.to_datetime(dates[-1]) + pd.Timedelta(days=days_until_threshold)
            return predicted_date
        
        return None
        
    except Exception as e:
        print(f"Error in threshold prediction: {str(e)}")
        return None

  def analyze_threshold_crossings(combined_summary_df, evolution_df, output_dir, thresholds=[0.9, 1.0], prediction_days=180):
    """
    Analyze when each island is predicted to cross specific risk thresholds
    """
    # Initialize LSTM predictor if not already done
    risk_predictor = RiskTrajectoryPredictor()
    risk_predictor.prepare_data(combined_summary_df)
    risk_predictor.train()
    
    threshold_results = []
    
    for _, row in evolution_df.iterrows():
        global_id = row['global_id']
        
        # Get historical data for this island
        island_data = combined_summary_df[
            combined_summary_df['global_id'] == global_id
        ].sort_values('date')
        
        if len(island_data) >= risk_predictor.sequence_length:
            # Get the last known date and risk score
            last_date = pd.to_datetime(island_data['date'].max())
            last_risk = island_data['risk_score'].iloc[-1]
            
            # Get start and end locations from the island_data
            start_location = island_data['start_location'].mean()
            end_location = island_data['end_location'].mean()
            
            # Get predictions
            predictions = risk_predictor.predict_trajectory(island_data, future_steps=prediction_days)
            
            # Create future dates
            future_dates = pd.date_range(
                start=last_date,
                periods=len(predictions) + 1,
                freq='D'
            )[1:]
            
            # Initialize result dictionary
            result = {
                'global_id': global_id,
                'start_location': start_location,
                'end_location': end_location,
                'last_measurement_date': last_date.strftime('%Y-%m-%d'),
                'current_risk_score': last_risk
            }
            
            # Find crossing dates for each threshold
            for threshold in thresholds:
                # Check if already crossed
                if last_risk >= threshold:
                    result[f'threshold_{threshold}_crossed'] = True
                    result[f'threshold_{threshold}_crossing_date'] = 'Already Crossed'
                    result[f'days_until_{threshold}'] = 0
                else:
                    # Find first crossing of threshold
                    crossing_indices = np.where(predictions >= threshold)[0]
                    
                    if len(crossing_indices) > 0:
                        crossing_idx = crossing_indices[0]
                        crossing_date = future_dates[crossing_idx]
                        
                        result[f'threshold_{threshold}_crossed'] = True
                        result[f'threshold_{threshold}_crossing_date'] = crossing_date.strftime('%Y-%m-%d')
                        result[f'days_until_{threshold}'] = (crossing_date - last_date).days
                    else:
                        result[f'threshold_{threshold}_crossed'] = False
                        result[f'threshold_{threshold}_crossing_date'] = 'Not Within Prediction Window'
                        result[f'days_until_{threshold}'] = None
            
            # Calculate velocity and acceleration of risk increase
            if len(predictions) > 1:
                risk_velocity = (predictions[1] - predictions[0])  # risk/day
                result['risk_velocity'] = risk_velocity
                
                if len(predictions) > 2:
                    risk_acceleration = (predictions[2] - 2*predictions[1] + predictions[0])  # risk/day²
                    result['risk_acceleration'] = risk_acceleration
                else:
                    result['risk_acceleration'] = 0
            else:
                result['risk_velocity'] = 0
                result['risk_acceleration'] = 0
            
            threshold_results.append(result)
    
    # Create DataFrame and sort by urgency
    results_df = pd.DataFrame(threshold_results)
    
    # Sort by days until first threshold crossing (excluding already crossed)
    for threshold in thresholds:
        mask = results_df[f'days_until_{threshold}'].notna()
        results_df.loc[mask, 'min_days_to_threshold'] = results_df.loc[mask, f'days_until_{threshold}']
    
    results_df['min_days_to_threshold'] = results_df['min_days_to_threshold'].fillna(float('inf'))
    results_df = results_df.sort_values('min_days_to_threshold')
    
    # Save results
    output_file = os.path.join(output_dir, 'threshold_crossing_predictions.csv')
    results_df.to_csv(output_file, index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_islands': len(results_df),
        'islands_crossing_thresholds': {
            str(threshold): {
                'already_crossed': len(results_df[results_df[f'threshold_{threshold}_crossing_date'] == 'Already Crossed']),
                'will_cross': len(results_df[results_df[f'threshold_{threshold}_crossed'] & 
                                          (results_df[f'threshold_{threshold}_crossing_date'] != 'Already Crossed')]),
                'wont_cross': len(results_df[~results_df[f'threshold_{threshold}_crossed']])
            }
            for threshold in thresholds
        },
        'average_days_to_threshold': {
            str(threshold): results_df[results_df[f'days_until_{threshold}'].notna()][f'days_until_{threshold}'].mean()
            for threshold in thresholds
        }
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'threshold_crossing_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    return results_df, summary_stats
