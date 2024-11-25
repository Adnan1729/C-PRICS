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

## Pseudo Code
```
Algorithm: C-PRICS Main Pipeline

Input: Raw CT Data CSV Files
Output: Risk Predictions and Analysis Reports

Phase 1: Signal Processing & Risk Scoring
1. For each input file:
   a. Detect peaks above threshold
   b. Analyze peak triplets
   c. Extract features (width, amplitude, area, slopes)
   d. Standardize feature vectors
   e. Apply K-means clustering
   f. Calculate risk scores

Phase 2: Pattern Evolution
1. For each identified pattern:
   a. Track pattern islands across time
   b. Calculate growth rates
   c. Detect pattern merging/splitting
   d. Analyze spatial drift

Phase 3: Predictive Analytics
1. For each tracked pattern:
   a. Create feature sequences
   b. Feed into LSTM model
   c. Generate risk trajectories
   d. Calculate crossing probabilities
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
[Add appropriate license information]

## Authors
- Adnan Mahmud
- Andrew Holdsworth
- Rob York
- Stephen Carpenter

*Intelligent Infrastructure Team, Network Rail*
