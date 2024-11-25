import os
from ct_analysis.config import INPUT_DIR, OUTPUT_DIR
from ct_analysis.preprocessing.data_processor import process_single_file, extract_date_from_filename

def main():
    """Main function to process all files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.csv'):
            continue
            
        print(f"Processing file: {filename}")
        
        date = extract_date_from_filename(filename)
        if not date:
            continue
            
        file_path = os.path.join(INPUT_DIR, filename)
        results_df = process_single_file(file_path)
        
        if not results_df.empty:
            results_df['Date'] = date
            output_file = os.path.join(OUTPUT_DIR, f"{date}.csv")
            results_df.to_csv(output_file, index=False)
            print(f"Saved results to: {output_file}")

if __name__ == "__main__":
    main()
