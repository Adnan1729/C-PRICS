# C-PRICS (Continuous Pre-Risk Intelligent Category Space)

## Overview
C-PRICS is a machine learning-based framework for early detection and continuous monitoring of cyclic top defects in railway tracks. Unlike traditional threshold-based detection systems, it employs continuous monitoring and feature-based analysis to identify emerging patterns before they reach critical levels.

## Key Features
- Continuous risk scoring replacing binary classification
- Early pattern detection capabilities
- Feature-based risk assessment
- Granular defect characterization
- Pattern propagation tracking
- Data-driven prediction for maintenance planning

## Technical Architecture
The system operates in three main phases:

### Phase 1: Signal Processing & Risk Scoring
- Triplet peak analysis
- Feature extraction (pattern width, peak amplitude, area under curve, etc.)
- K-means clustering for risk categorization

### Phase 2: Pattern Evolution Analysis
- Island tracking algorithm
- Spatial-temporal correlation
- Growth rate analysis
- Pattern merging/splitting detection

### Phase 3: Future Risk Prediction
- LSTM-based sequence modeling
- 90-day risk trajectory prediction
- Risk velocity and acceleration calculation

## Project Structure
```
CT-Analysis-Pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── README.md
│   └── api.md
├── examples/
│   └── cyclic_tops_classification.ipynb    # Original notebook with complete pipeline execution
├── output/
│   ├── plots/                  # Directory for generated plots
│   │   ├── evolution_plots/    # Individual island evolution plots
│   │   ├── spectral_plots/     # Spectral heatmaps
│   │   └── summary_plots/      # Growth rate and risk analysis plots
│   ├── summaries/              # Directory for CSV outputs
│   │   ├── combined_summary/   # Combined analysis results
│   │   ├── evolution_metrics/  # Island evolution metrics
│   │   └── threshold_results/  # Threshold crossing predictions
│   └── metadata/               # Analysis metadata and configurations
└── ct_analysis/
    ├── __init__.py
    ├── config.py               # Contains CONFIG dictionary with all parameters
    ├── preprocessing/
    │   ├── __init__.py
    │   └── data_processor.py   
    │       # Functions:
    │       # - process_single_file()
    │       # - extract_date_from_filename()
    │       # - find_peak_islands()
    │       # - sigmoid_value()
    ├── feature_extraction/
    │   ├── __init__.py
    │   └── feature_extractor.py
    │       # Classes:
    │       # - CTFeatureExtractor
    │           # Methods:
    │           # - fit_spline()
    │           # - calculate_features()
    │           # - extract_all_features()
    │           # - plot_all_islands()
    ├── identification/
    │   ├── __init__.py
    │   └── island_identifier.py
    │       # Classes:
    │       # - IslandIdentifier
    │           # Methods:
    │           # - calculate_overlap()
    │           # - find_matches()
    │           # - register_new_island()
    │           # - update_island()
    │           # - check_splits_and_merges()
    │           # - mark_split()
    │           # - mark_merge()
    │           # - register_islands()
    │           # - get_history_summary()
    ├── analysis/
    │   ├── __init__.py
    │   ├── ct_analyzer.py
    │   │   # Classes:
    │   │   # - CTAnalyzer
    │   │       # Methods:
    │   │       # - fit()
    │   │       # - _calculate_cluster_risks()
    │   │       # - predict_risk()
    │   │       # - _fallback_risk_calculation()
    │   │       # - _sigmoid()
    │   ├── temporal_analyzer.py
    │   │   # Classes:
    │   │   # - TemporalEvolutionAnalyzer
    │   │       # Methods:
    │   │       # - initialize_risk_predictor()
    │   │       # - analyze_island_evolution()
    │   │       # - plot_spectral_heatmap()
    │   │       # - plot_island_evolution()
    │   └── threshold_analyzer.py
    │       # Functions:
    │       # - predict_threshold_crossing()
    │       # - analyze_threshold_crossings()
    ├── models/
    │   ├── __init__.py
    │   ├── lstm_predictor.py
    │   │   # Classes:
    │   │   # - LSTMPredictor
    │   │       # Methods:
    │   │       # - forward()
    │   │   # - RiskTrajectoryPredictor
    │   │       # Methods:
    │   │       # - prepare_data()
    │   │       # - train()
    │   │       # - predict_trajectory()
    │   └── dataset.py
    │       # Classes:
    │       # - IslandSequenceDataset
    │           # Methods:
    │           # - __len__()
    │           # - __getitem__()
    ├── visualization/
    │   ├── __init__.py
    │   └── plotters.py
    │       # Functions:
    │       # - create_summary_visualizations()
    │       # - plot_spectral_heatmap()
    │       # - plot_island_evolution()
    └── utils/
        ├── __init__.py
        └── helpers.py
            # Functions:
            # - get_sorted_files()
            # - process_file()
            # - analyze_temporal_evolution()
```

# Cyclic Tops Classification System - Algorithm Overview

## System Configuration
```python
Configuration Parameters:
- CHANNEL: Default "18_top_right"
- Input/Output Paths
- Distance Threshold: 5.0 meters
- Overlap Threshold: 30%
- Number of Clusters: 5
- Interpolation Points: 100
- Risk Smoothing Factor: 0.7
- Risk Adjustment Maximum: 0.2

Feature Weights:
- Peak Amplitude: 0.3
- Average Y: 0.1
- Distance Difference: 0.1
- Pattern Width: 0.15
- Area Under Curve: 0.15
- Mean Slope: 0.1
- Max Slope: 0.1
```

## Algorithm Components

### 1. Signal Processing and Feature Extraction

```python
def process_single_file(file_path):
    1. Load CT data from CSV
    2. For each CT channel:
        a. Extract channel-specific data
        b. Identify peak regions:
            - Find consecutive peaks above threshold
            - Group peaks into islands (3+ consecutive peaks)
            - Calculate triplet scores for each group
        c. Store peak information:
            - Location
            - Amplitude
            - Island ID
    3. Return processed data frame

def extract_features(island_data):
    1. Fit cubic spline to data points
    2. Calculate key features:
        - Peak amplitude
        - Average signal value
        - Pattern width
        - Center location
        - Mean and max slopes
        - Area under curve
        - Distance metrics
    3. Return feature dictionary
```

### 2. Island Identification System

```python
class IslandIdentifier:
    def calculate_overlap(island1, island2):
        1. Calculate intersection of location ranges
        2. Compute overlap ratio
        3. Return overlap percentage

    def find_matches(new_island):
        1. Get active islands in system
        2. For each existing island:
            a. Calculate location differences
            b. Compute overlap percentage
            c. Generate confidence score
        3. Return matches sorted by confidence

    def register_islands(new_data):
        1. Check for potential splits/merges
        2. Process each new island:
            If no match found:
                Register as new island
            If match found:
                Update existing island record
        3. Return updated island registry
```

### 3. Risk Analysis System

```python
class RiskAnalyzer:
    def train_model(historical_data):
        1. Prepare feature set:
            - Scale features
            - Apply feature weights
        2. Train KMeans classifier
        3. Calculate risk scores for clusters

    def predict_risk(island_features):
        1. Scale new features
        2. Apply feature weights
        3. Determine cluster
        4. Calculate base risk
        5. Apply adjustments:
            - Point-specific modifications
            - Temporal smoothing
        6. Return final risk score

class TrajectoryPredictor:
    def prepare_sequences(historical_data):
        1. Create training sequences
        2. Scale features
        3. Initialize LSTM model

    def predict_future(island_data, days=90):
        1. Process recent data
        2. Generate predictions:
            For each future step:
                a. Predict next risk score
                b. Update sequence
        3. Return risk trajectory
```

### 4. Main Processing Pipeline

```python
def main_pipeline():
    1. Initialize Systems:
        - Create IslandIdentifier
        - Initialize RiskAnalyzer
        - Setup TrajectoryPredictor

    2. First Processing Pass:
        For each input file:
            a. Process raw signals
            b. Extract features
            c. Identify islands
            d. Store initial results

    3. Risk Analysis:
        a. Train risk model
        b. Second pass through files:
            - Calculate risk scores
            - Update island records
        c. Generate combined summary

    4. Evolution Analysis:
        a. Train LSTM predictor
        b. Analyze patterns:
            - Growth rates
            - Risk trajectories
            - Threshold crossings
        c. Generate visualizations

    5. Output Generation:
        - Save processed data
        - Generate summary reports
        - Create visualization plots
        - Export system metadata
```

## Error Handling and Validation

```python
Throughout all operations:
1. Input Validation:
    - Check data format
    - Verify numerical ranges
    - Validate timestamps

2. Processing Safeguards:
    - Handle missing data
    - Catch calculation exceptions
    - Implement numerical stability checks

3. Output Validation:
    - Verify risk score ranges (0-1)
    - Validate prediction timelines
    - Ensure data consistency
```

## System Outputs

```python
1. Processed Data:
    - Island identification records
    - Feature calculations
    - Risk assessments

2. Analysis Results:
    - Evolution patterns
    - Risk predictions
    - Threshold crossing estimates

3. Visualizations:
    - Time series plots
    - Risk heat maps
    - Evolution patterns

4. Summary Reports:
    - System performance metrics
    - Risk distribution analysis
    - Prediction confidence levels
```


## Installation
```bash
pip install -r requirements.txt
```

## Usage
See examples/cyclic_tops_classification.ipynb for detailed usage examples.

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Plotly
- Matplotlib

## Future Work
- Development of performance metrics (accuracy, precision)
- Testing in controlled environments
- Bug fixes for LSTM prediction display

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is private and is not intended for public use, distribution, or reproduction. Unauthorized access, sharing, or redistribution of any part of this repository is strictly prohibited.

If you have any questions or wish to request access, please contact the repository owner.

## Authors
- Adnan Mahmud
- Andrew Holdsworth
- Rob York
- Stephen Carpenter

*Intelligent Infrastructure Team, Network Rail*
