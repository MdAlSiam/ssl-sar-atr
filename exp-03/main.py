from data_processor import *
from utils import *
from ResultsTracker import ResultsTracker

# Main execution script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['process', 'pretext', 'downstream'], required=True)
    parser.add_argument('--data_dir', type=str, help='Path to raw data directory')
    parser.add_argument('--split_index', type=int, default=0, help='Which 20% split to use for testing (0-4)')
    parser.add_argument('--architecture', type=str, help='Architecture name for pretext/downstream task')
    parser.add_argument('--classifier', type=str, help='Classifier for downstream task')
    args = parser.parse_args()
    
    if args.mode == 'process':
        # Process and save data
        save_processed_data(args.data_dir, args.split_index)
    
    elif args.mode == 'pretext':
        # Train pretext model
        data_path = os.path.join(DATA_DIR, f"{args.data_dir.split('/')[-2]}_{args.data_dir.split('/')[-1]}_split_{args.split_index}")
        run_pretext_pipeline(data_path, args.architecture, args.split_index)
    
    elif args.mode == 'downstream':
        # Run downstream task
        data_path = os.path.join(DATA_DIR, f"{args.data_dir.split('/')[-2]}_{args.data_dir.split('/')[-1]}_split_{args.split_index}")
        clf, cv_scores, test_metrics = run_downstream_pipeline(data_path, args.architecture, args.classifier, args.split_index)
        
        # Save results
        results = {
            'data_dir': args.data_dir,
            'split_index': args.split_index,
            'architecture': args.architecture,
            'classifier': args.classifier,
            'cv_scores': cv_scores,
            'test_metrics': test_metrics
        }

        results_tracker = ResultsTracker()
        results_tracker.add_result(
            data_dir=args.data_dir,
            architecture=args.architecture,
            classifier=args.classifier,
            cv_metrics=cv_scores,
            test_metrics=test_metrics
        )
        results_tracker.display_results()
        
        results_path = os.path.join(
            RESULTS_DIR,
            f"results_{args.data_dir.split('/')[-2]}_{args.data_dir.split('/')[-1]}_split_{args.split_index}_{args.architecture}_{args.classifier}.txt"
        )
        
        with open(results_path, 'w') as f:
            f.write(str(results))