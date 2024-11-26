# Data processing functions

def process_single_file(file_path):
    """Process a single CT data file and return results"""
    try:
        # Read the CSV file
        ct = pd.read_csv(file_path, usecols=cols_to_use)
        
        final_results = []
        
        # Process each channel
        for ct_channel in ct_channels:
            # Get data for this channel
            ct_channel_df = ct[['CT_Index', 'Location_Norm_m', ct_channel]].dropna(subset=[ct_channel])
            
            # Find peaks and their scores
            peaks_df = find_peak_islands(ct_channel_df, ct_channel)
            
            if not peaks_df.empty:
                # Add channel name
                peaks_df['Channel'] = ct_channel
                final_results.append(peaks_df)
        
        if final_results:
            # Combine all results
            return pd.concat(final_results, ignore_index=True)
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return pd.DataFrame()
def sigmoid_value(val):
    """Placeholder function for sigmoid value calculation"""
    return 0.0

def extract_date_from_filename(filename):
    """Extract date from filename format 'OWW-2100-134404-222214-2021-08-12.csv'"""
    try:
        parts = filename.replace('.csv', '').split('-')
        year = parts[-3]
        month = parts[-2]
        day = parts[-1]
        date_str = f"{year}-{month}-{day}"
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except Exception as e:
        print(f"Warning: Could not extract valid date from filename: {filename}. Error: {str(e)}")
        return None

def find_peak_islands(df, col):
    """Find islands of consecutive peaks and their triplet scores"""
    # Create a boolean mask for peaks
    peaks = df[col] >= peak_amp
    
    # Create groups of consecutive peaks
    peak_groups = (peaks != peaks.shift()).cumsum()[peaks]
    
    peak_data = []
    island_counter = 1  # Counter for island IDs
    
    for group_id in peak_groups.unique():
        # Get indices for this group of peaks
        group_indices = peak_groups[peak_groups == group_id].index
        
        # Only process groups with 3 or more consecutive peaks
        if len(group_indices) >= 3:
            group_df = df.loc[group_indices]
            
            # Create all possible triplets within this group
            for i in range(len(group_df) - 2):
                triplet_indices = group_df.index[i:i+3]
                triplet_data = group_df.loc[triplet_indices]
                
                triplet_score = sum(triplet_data[col].values + 
                                  [sigmoid_value(v) for v in triplet_data[col].values])
                
                # Store data for each point in the triplet
                for point in triplet_indices:
                    peak_data.append({
                        'CT_Index': df.loc[point, 'CT_Index'],
                        'Location_Norm_m': df.loc[point, 'Location_Norm_m'],
                        'Value': df.loc[point, col],
                        'triplet_score': triplet_score,
                        'island_id': island_counter,
                        'island_size': len(group_indices)
                    })
            
            island_counter += 1  # Increment island ID for next group
    
    if not peak_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    peaks_df = pd.DataFrame(peak_data)
    
    # For each CT_Index, keep only the highest triplet score
    peaks_df = peaks_df.loc[peaks_df.groupby('CT_Index')['triplet_score'].idxmax()]
    
    return peaks_df
