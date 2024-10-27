from tabulate import tabulate
from typing import Dict, List, Any
import pandas as pd
import numpy as np

class ResultsTracker:
    def __init__(self):
        self.results = []
        
    def add_result(self, data_dir: str, architecture: str, classifier: str, 
                   cv_metrics: Dict[str, List[float]], test_metrics: Dict[str, Any]):
        """Add a single result to the tracker"""
        result = {
            'Dataset': f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}",  # os.path.basename(data_dir),
            'Architecture': architecture,
            'Classifier': classifier,
            # Cross-validation metrics (mean ± std)
            'CV_Accuracy': f"{np.mean(cv_metrics['accuracies'])*100:.1f}±{np.std(cv_metrics['accuracies'])*100:.1f}",
            'CV_Precision': f"{np.mean(cv_metrics['precisions'])*100:.1f}±{np.std(cv_metrics['precisions'])*100:.1f}",
            'CV_Recall': f"{np.mean(cv_metrics['recalls'])*100:.1f}±{np.std(cv_metrics['recalls'])*100:.1f}",
            'CV_F1': f"{np.mean(cv_metrics['f1_scores'])*100:.1f}±{np.std(cv_metrics['f1_scores'])*100:.1f}",
            # Test metrics
            'Test_Accuracy': f"{test_metrics['accuracy']*100:.1f}",
            'Test_Precision': f"{test_metrics['precision']*100:.1f}",
            'Test_Recall': f"{test_metrics['recall']*100:.1f}",
            'Test_F1': f"{test_metrics['f1']*100:.1f}",
            'Test_Confusion_Matrix': f"{test_metrics['confusion_matrix']}"
        }
        self.results.append(result)
    
    def display_results(self):
        """Display results in a formatted table"""
        if not self.results:
            print("No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        # Group by dataset and architecture
        grouped = df.groupby(['Dataset', 'Architecture'])
        
        for (dataset, arch), group in grouped:
            print(f"\n=== Results: Dataset: <{dataset}> | Pretext Model: <{arch}> ===")
            
            # Select and rename columns for better display
            display_cols = [
                'Classifier',
                'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1',
                'CV_Accuracy', 'CV_Precision', 'CV_Recall', 'CV_F1'
            ]
            
            display_df = group[display_cols].copy()
            
            # Format the table
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".3f"))

            # Again for the confusion matrices
            display_cols = [
                'Classifier',
                'Test_Confusion_Matrix'
            ]
            
            display_df = group[display_cols].copy()
            
            # Format the table
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    def save_results(self, filename: str):
        """Save results to a CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")