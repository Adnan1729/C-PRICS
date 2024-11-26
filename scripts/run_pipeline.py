# Main pipeline execution script

#!/usr/bin/env python
"""
Main execution script for the CT Analysis Pipeline.
This script orchestrates the entire analysis process from data loading to results generation.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import configurations
from ct_analysis.config import PIPELINE_CONFIG

# Import pipeline components
from ct_analysis.preprocessing.data_processor import process_file
from ct_analysis.identification.island_identifier import IslandIdentifier
from ct_analysis.analysis.ct_analyzer import CTAnalyzer
from ct_analysis.analysis.temporal_analyzer import TemporalEvolutionAnalyzer
from ct_analysis.analysis.threshold_analyzer import analyze_threshold_crossings
from ct_analysis.utils.helpers import get_sorted_files, analyze_temporal_evolution

class PipelineExecutor:
    """
    Class to handle the execution of the CT analysis pipeline.
    Manages the workflow and maintains the state of the analysis.
    """
    
    def __init__(self, config=PIPELINE_CONFIG):
        """Initialize pipeline with configuration"""
        self.config = config
        self.identifier = None
        self.analyzer = None
        self.temporal_analyzer = None
        self.all_summaries = []
        self.final_summary = None
        self.evolution_df = None
        
    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            os.path.join(self.config['PHASE1_DIR'], 'summaries'),
            os.path.join(self.config['PHASE1_DIR'], 'plots'),
            self.config['PHASE3_DIR']
        ]
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            
    def initialize_components(self):
        """Initialize pipeline components"""
        self.identifier = IslandIdentifier(
            distance_threshold=self.config['ISLAND_DISTANCE_THRESHOLD'],
            overlap_threshold=self.config['ISLAND_OVERLAP_THRESHOLD']
        )
        self.analyzer = CTAnalyzer(n_clusters=self.config['NUM_CLUSTERS'])
        self.analyzer.feature_weights = self.config['FEATURE_WEIGHTS']
        
    def process_files_first_pass(self, sorted_files):
        """First pass: Process files for ML training and initial island identification"""
        print("\nFirst pass: Processing files for ML training and initial island identification...")
        for date, filename in sorted_files:
            print(f"Processing {filename}...")
            input_file = os.path.join(self.config['INPUT_DIR'], filename)
            date_str = date.strftime('%Y-%m-%d')
            
            summary_df = process_file(
                input_file=input_file,
                output_dir=self.config['PHASE1_DIR'],
                channel=self.config['CHANNEL'],
                date_str=date_str,
                identifier=self.identifier
            )
            
            if summary_df is not None:
                self.all_summaries.append(summary_df)
                
    def train_risk_analyzer(self):
        """Train the risk analyzer on collected data"""
        print("\nTraining risk analyzer...")
        combined_summary = pd.concat(self.all_summaries, ignore_index=True)
        self.analyzer.fit(combined_summary)
        
    def process_files_second_pass(self, sorted_files):
        """Second pass: Process files with risk scoring"""
        print("\nSecond pass: Processing files with risk scoring...")
        self.all_summaries = []  # Reset summaries for second pass
        
        for date, filename in sorted_files:
            print(f"Processing {filename}...")
            input_file = os.path.join(self.config['INPUT_DIR'], filename)
            date_str = date.strftime('%Y-%m-%d')
            
            summary_df = process_file(
                input_file=input_file,
                output_dir=self.config['PHASE1_DIR'],
                channel=self.config['CHANNEL'],
                date_str=date_str,
                analyzer=self.analyzer,
                identifier=self.identifier
            )
            
            if summary_df is not None:
                self.all_summaries.append(summary_df)
                
    def save_phase1_results(self):
        """Save Phase 1 results and generate evolution summary"""
        # Save final combined summary
        self.final_summary = pd.concat(self.all_summaries, ignore_index=True)
        self.final_summary.sort_values(['date', 'global_id'], inplace=True)
        combined_summary_path = os.path.join(
            self.config['PHASE1_DIR'], 
            f'combined_summary_{self.config["CHANNEL"]}.csv'
        )
        self.final_summary.to_csv(combined_summary_path, index=False)
        
        # Save evolution history
        history_df = self.identifier.get_history_summary()
        history_df.to_csv(os.path.join(self.config['PHASE1_DIR'], 'island_history.csv'), index=False)
        
        # Generate evolution summary
        self.generate_evolution_summary()
        
    def generate_evolution_summary(self):
        """Generate and save evolution summary"""
        print("\nGenerating evolution summary...")
        evolution_summary = []
        
        for global_id in self.final_summary['global_id'].unique():
            island_data = self.final_summary[
                self.final_summary['global_id'] == global_id
            ].sort_values('date')
            
            risk_volatility = island_data['risk_score'].std() if len(island_data) > 1 else 0
            
            summary = {
                'global_id': global_id,
                'first_appearance': island_data['date'].min(),
                'last_appearance': island_data['date'].max(),
                'num_observations': len(island_data),
                'avg_start_location': island_data['start_location'].mean(),
                'avg_end_location': island_data['end_location'].mean(),
                'location_drift': max(
                    island_data['start_location'].max() - island_data['start_location'].min(),
                    island_data['end_location'].max() - island_data['end_location'].min()
                ),
                'avg_risk_score': island_data['risk_score'].mean(),
                'max_risk_score': island_data['risk_score'].max(),
                'risk_score_trend': np.polyfit(
                    range(len(island_data)), 
                    island_data['risk_score'], 
                    1
                )[0] if len(island_data) > 1 else 0,
                'risk_volatility': risk_volatility,
                'risk_acceleration': np.diff(island_data['risk_score']).mean() if len(island_data) > 2 else 0
            }
            evolution_summary.append(summary)
        
        self.evolution_df = pd.DataFrame(evolution_summary)
        self.evolution_df.to_csv(
            os.path.join(self.config['PHASE1_DIR'], 'island_evolution_summary.csv'), 
            index=False
        )
        
    def run_phase3_analysis(self):
        """Run Phase 3 temporal evolution analysis"""
        print("\n=== Phase 3: Temporal Evolution Analysis ===")
        
        # Initialize and train LSTM predictor
        print("\nInitializing LSTM predictor...")
        self.temporal_analyzer = TemporalEvolutionAnalyzer()
        self.temporal_analyzer.initialize_risk_predictor(self.final_summary)
        
        # Perform temporal evolution analysis
        print("\nPerforming temporal evolution analysis...")
        combined_summary_path = os.path.join(
            self.config['PHASE1_DIR'], 
            f'combined_summary_{self.config["CHANNEL"]}.csv'
        )
        temporal_evolution_df = analyze_temporal_evolution(
            combined_summary_path, 
            self.config['PHASE3_DIR']
        )
        
        # Perform threshold crossing analysis
        print("\nAnalyzing threshold crossings...")
        threshold_results, threshold_summary = analyze_threshold_crossings(
            combined_summary_df=self.final_summary,
            evolution_df=temporal_evolution_df,
            output_dir=self.config['PHASE3_DIR']
        )
        
        return threshold_summary
        
    def save_metadata(self, sorted_files, threshold_summary):
        """Save final analysis metadata"""
        metadata = {
            'channel': self.config['CHANNEL'],
            'files_processed': len(sorted_files),
            'total_islands': len(self.final_summary),
            'unique_islands': len(self.evolution_df),
            'number_of_clusters': self.config['NUM_CLUSTERS'],
            'cluster_risks': self.analyzer.cluster_risks.tolist(),
            'feature_weights': self.config['FEATURE_WEIGHTS'],
            'risk_smoothing_factor': self.config['RISK_SMOOTHING_FACTOR'],
            'risk_adjustment_max': self.config['RISK_ADJUSTMENT_MAX'],
            'date_range': [
                str(self.final_summary['date'].min()), 
                str(self.final_summary['date'].max())
            ],
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': self.config,
            'risk_score_statistics': {
                'mean': float(self.final_summary['risk_score'].mean()),
                'std': float(self.final_summary['risk_score'].std()),
                'min': float(self.final_summary['risk_score'].min()),
                'max': float(self.final_summary['risk_score'].max()),
                'volatility': float(
                    self.final_summary.groupby('global_id')['risk_score'].std().mean()
                )
            },
            'threshold_analysis': threshold_summary,
            'temporal_analysis_completed': True
        }
        
        with open(os.path.join(self.config['OUTPUT_BASE_DIR'], 'full_analysis_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

def main():
    """Main function to execute the pipeline"""
    try:
        print("\n=== Starting C-PRICS Analysis Pipeline ===")
        
        # Initialize pipeline
        pipeline = PipelineExecutor()
        pipeline.setup_directories()
        pipeline.initialize_components()
        
        # Phase 1: Initial Processing and Risk Analysis
        print("\n=== Phase 1: Signal Processing and Risk Analysis ===")
        print(f"Channel: {pipeline.config['CHANNEL']}")
        print(f"Input Directory: {pipeline.config['INPUT_DIR']}")
        
        # Get sorted files
        sorted_files = get_sorted_files(pipeline.config['INPUT_DIR'])
        print(f"\nFound {len(sorted_files)} files to process")
        
        # Execute pipeline phases
        pipeline.process_files_first_pass(sorted_files)
        pipeline.train_risk_analyzer()
        pipeline.process_files_second_pass(sorted_files)
        pipeline.save_phase1_results()
        
        # Run Phase 3 analysis
        threshold_summary = pipeline.run_phase3_analysis()
        
        # Save final metadata
        pipeline.save_metadata(sorted_files, threshold_summary)
        
        # Print completion message
        print("\n=== Analysis Pipeline Complete ===")
        print(f"Processed {len(sorted_files)} files")
        print(f"Found {len(pipeline.evolution_df)} unique islands")
        print(f"Results saved to {pipeline.config['OUTPUT_BASE_DIR']}")
        print(f"Phase 1 results: {pipeline.config['PHASE1_DIR']}")
        print(f"Phase 3 results: {pipeline.config['PHASE3_DIR']}")
        
    except Exception as e:
        print(f"\nError in processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
