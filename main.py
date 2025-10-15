"""
Main pipeline script for Paediatric Appendicitis AI Project (CPU-only, single-threaded version).
Chained pipeline: runs the full pipeline and then the evaluation on processed data.
"""

import os
import argparse
import logging
import sys
import pandas as pd
import joblib
from src import data_extraction

sys.path.append('src')

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """
    Run the complete CPU-only, single-threaded ML pipeline from start to finish.
    This includes data preprocessing, feature engineering, model training, and evaluation.
    """
    logger.info("Starting CPU-only, single-threaded pipeline execution...")

    try:
        # Import necessary modules from the 'src' directory
        from src.data_preprocessing import DataPreprocessor
        from src.feature_engineering import FeatureEngineer
        from src.model_training import ModelTrainer
        from src.evaluation import ModelEvaluator

        # 1. Data Preprocessing Stage
        logger.info("=== STAGE 1: Data Preprocessing ===")
        preprocessor = DataPreprocessor()
        # Define the path for the merged dataset
        sample_data_path = "data/processed/merged_data.csv"
        # If the merged data doesn't exist, create a sample version
        if not os.path.exists(sample_data_path):
            logger.info("Creating sample data structure...")
            data_extraction.main()

        # Load the merged dataset
        df = preprocessor.load_data("merged_data")

        if df is not None:
            # Prepare the dataset for modeling
            df = preprocessor.prepare_dataset(df, 'report_text_us', 'severity_grade_us')
            # Split the data into training, validation, and test sets
            train_df, val_df, test_df = preprocessor.split_data(df)
            # Save the processed dataframes to disk
            preprocessor.save_processed_data(train_df, val_df, test_df)
            logger.info("Preprocessing stage completed successfully!")
        else:
            logger.warning("No data found. Please add your dataset to data/raw/")
            return

        # 2. Feature Engineering Stage
        logger.info("=== STAGE 2: Feature Engineering ===")
        feature_engineer = FeatureEngineer()
        # Load the training and testing data
        train_df = pd.read_csv("data/processed/train_data.csv")
        test_df = pd.read_csv("data/processed/test_data.csv")
        if train_df is not None and test_df is not None:
            # Extract TF-IDF features from the processed text
            X_train_tfidf, X_test_tfidf = feature_engineer.extract_tfidf_features(
                train_df['processed_text'],
                test_df['processed_text']
            )
            # Build a vocabulary from medical keywords
            keyword_vocab = feature_engineer.build_keyword_vocabulary(train_df['medical_keywords'])
            # Create keyword-based features
            X_train_keywords = feature_engineer.create_keyword_features(train_df['medical_keywords'], keyword_vocab)
            X_test_keywords = feature_engineer.create_keyword_features(test_df['medical_keywords'], keyword_vocab)
            # Generate text-based statistical features
            X_train_stats = feature_engineer.create_text_statistics(train_df['processed_text'])
            X_test_stats = feature_engineer.create_text_statistics(test_df['processed_text'])
            # Combine all engineered features
            X_train_combined = feature_engineer.combine_features([X_train_tfidf, X_train_keywords, X_train_stats])
            X_test_combined = feature_engineer.combine_features([X_test_tfidf, X_test_keywords, X_test_stats])
            # Save the combined feature sets
            feature_engineer.save_features(
                [X_train_combined, X_test_combined],
                ['X_train_combined', 'X_test_combined']
            )
            logger.info("Feature engineering stage completed successfully!")

        # 3. Model Training Stage
        logger.info("=== STAGE 3: Model Training ===")
        trainer = ModelTrainer()
        # Load the combined training features
        X_train = feature_engineer.load_features(['X_train_combined'])[0]
        if X_train is not None:
            # Train several baseline machine learning models
            models = trainer.train_baseline_models(X_train, train_df['label_encoded'])
            # Train an ensemble model
            ensemble = trainer.train_ensemble(X_train, train_df['label_encoded'])
            # Save the trained models
            trainer.save_models()
            logger.info("Training stage completed successfully!")

        # 4. Model Evaluation Stage
        logger.info("=== STAGE 4: Model Evaluation ===")
        evaluator = ModelEvaluator()
        # Load the test data and features
        test_df = pd.read_csv("data/processed/test_data.csv")
        X_test = feature_engineer.load_features(['X_test_combined'])[0]
        if test_df is not None and X_test is not None:
            # Load the best performing model for evaluation
            try:
                best_model = joblib.load("models/best_model.pkl")
            except:
                best_model = joblib.load("models/random_forest_model.pkl")
            # Evaluate the model's performance on the test set
            results = evaluator.evaluate_model(
                best_model,
                X_test,
                test_df['label_encoded'],
                "Best Model"
            )
            # Format the results for display
            comparison_data = [{
                'Method': 'Best Model',
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
            }]
            comparison_df = pd.DataFrame(comparison_data)
            print("\n=== FINAL RESULTS ===")
            print(comparison_df.to_string(index=False))
            # Save the final results to a CSV file
            os.makedirs('results', exist_ok=True)
            comparison_df.to_csv('results/final_results.csv', index=False)
            logger.info("Evaluation stage completed successfully!")

        logger.info("CPU-only, single-threaded pipeline completed successfully!")

        # --- CHAINED EVALUATION LOGIC ---
        # Run a separate evaluation pipeline to compare multiple methods
        logger.info("=== CHAINED EVALUATION PIPELINE ===")
        run_chained_evaluation()

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


def run_chained_evaluation():
    """
    Run a separate evaluation pipeline to compare the Random Forest model
    against manual extraction and a GPT-4 baseline.
    """
    from src.evaluation import ModelEvaluator
    from src.feature_engineering import FeatureEngineer
    import joblib
    import pandas as pd

    logger.info("Starting chained evaluation using processed test and train data...")

    # Load processed test and training data
    test_df = pd.read_csv("data/processed/test_data.csv")
    train_df = pd.read_csv("data/processed/train_data.csv")

    # Load the trained Random Forest model and the TF-IDF vectorizer
    rf_model = joblib.load("models/random_forest_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    # --- Recreate features for the test set ---
    feature_engineer = FeatureEngineer()

    # 1. Transform test data using the fitted TF-IDF vectorizer
    X_test_tfidf = tfidf.transform(test_df['processed_text'].fillna("").astype(str))

    # 2. Create keyword features using the vocabulary from the training set
    keyword_vocab = feature_engineer.build_keyword_vocabulary(train_df['medical_keywords'])
    X_test_keywords = feature_engineer.create_keyword_features(test_df['medical_keywords'], keyword_vocab)

    # 3. Generate text statistics features for the test set
    X_test_stats = feature_engineer.create_text_statistics(test_df['processed_text'])

    # 4. Combine all features for the test set
    X_test_combined = feature_engineer.combine_features([X_test_tfidf, X_test_keywords, X_test_stats])

    # Get the true labels for the test set
    label_data = test_df['label_encoded']

    evaluator = ModelEvaluator()

    # Evaluate the Random Forest model
    rf_results = evaluator.evaluate_model(
        rf_model,
        X_test_combined,
        label_data,
        "Random Forest"
    )

    # Evaluate manual extraction results if the gold standard file is available
    try:
        manual_results_df = pd.read_csv("data/raw/adjudicated_gold_standard_labels.csv")
        manual_eval_results = evaluator.evaluate_manual_extraction(manual_results_df)
    except FileNotFoundError:
        manual_eval_results = None

    # Evaluate GPT-4 performance on a sample of the test set
    chatgpt_texts = test_df['processed_text'].tolist()
    chatgpt_labels = label_data.tolist()
    chatgpt_results = evaluator.evaluate_chatgpt4(
        chatgpt_texts,
        chatgpt_labels,
        sample_size=103
    )

    # Combine all evaluation results for comparison
    all_results = [rf_results]
    if manual_eval_results is not None:
        all_results.append(manual_eval_results)
    all_results.append(chatgpt_results)
    comparison_df = evaluator.create_comparison_table(all_results)

    # Display and save the final comparison table in various formats
    print("\n=== CHAINED FINAL COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))
    print("\n--- Markdown Table ---\n")
    print(comparison_df.to_markdown(index=False))

    # Save results to CSV, LaTeX, and Markdown files
    comparison_df.to_csv('results/chained_method_comparison.csv', index=False)
    comparison_df.to_latex('results/chained_method_comparison.tex', index=False)
    with open('results/chained_method_comparison.md', 'w') as mdfile:
        mdfile.write(comparison_df.to_markdown(index=False))

    evaluator.save_evaluation_results(all_results, comparison_df)

    print("Chained evaluation stages completed successfully!")





if __name__ == "__main__":
    # Set up an argument parser to allow running specific pipeline stages
    parser = argparse.ArgumentParser(description='Paediatric Appendicitis AI Pipeline (CPU-only, single-threaded)')
    parser.add_argument('--stage', type=str,
                        choices=['all', 'preprocessing', 'features', 'training', 'evaluation', 'create_sample', 'chained_eval'],
                        default='all', help='Pipeline stage to run')

    args = parser.parse_args()

    # Execute the chosen stage
    if args.stage == 'create_sample':
        data_extraction.main()
    elif args.stage == 'all':
        run_full_pipeline()
    elif args.stage == 'chained_eval':
        run_chained_evaluation()
    else:
        logger.info(f"Individual stage execution not implemented in CPU-only version. Use 'all' to run full pipeline.")