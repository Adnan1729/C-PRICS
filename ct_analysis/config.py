# Configuration file for CT Analysis Pipeline

# Data directories
INPUT_DIR = '/path/to/input/data'
OUTPUT_DIR = '/path/to/output/data'

# Analysis parameters
PEAK_AMPLITUDE = 3

# Column definitions
COLUMNS_TO_USE = [
    'CT_Index', 'ELR', 'TrackId', '45_top_right', '45_top_left', '6_top_right',
    '6_top_left', '9_top_right', '9_top_left', '135_top_right', '135_top_left', 
    '18_top_right', '18_top_left', 'Location_Norm_m'
]

CT_CHANNELS = [
    '45_top_right', '45_top_left', '6_top_right', '6_top_left',
    '9_top_right', '9_top_left', '135_top_right', '135_top_left',
    '18_top_right', '18_top_left'
]

# Pipeline configuration
PIPELINE_CONFIG = {
    'CHANNEL': '18_top_right',
    'ISLAND_DISTANCE_THRESHOLD': 5.0,
    'ISLAND_OVERLAP_THRESHOLD': 0.3,
    'NUM_CLUSTERS': 5,
    'INTERPOLATION_POINTS': 100,
    'FEATURE_WEIGHTS': {
        'peak_amplitude': 0.3,
        'average_y': 0.1,
        'distance_difference': 0.1,
        'pattern_width': 0.15,
        'area_under_curve': 0.15,
        'mean_slope': 0.1,
        'max_slope': 0.1
    },
    'RISK_SMOOTHING_FACTOR': 0.7,
    'RISK_ADJUSTMENT_MAX': 0.2
}
