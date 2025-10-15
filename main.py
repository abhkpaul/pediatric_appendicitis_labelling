#!/usr/bin/env python3
"""
Main pipeline script for Paediatric Appendicitis AI Project (CPU-only, single-threaded version).
Chained pipeline: runs the full pipeline and then the evaluation on processed data.
"""

import os
import argparse
import yaml
import logging
import sys
import pandas as pd
import numpy as np
import joblib


# Add the src directory to Python path
sys.path.append('src')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Run the complete CPU-only, single-threaded ML pipeline."""
    logger.info("Starting CPU-only, single-threaded pipeline execution...")

    try:
        from src.data_preprocessing import DataPreprocessor
        from src.feature_engineering import FeatureEngineer
        from src.model_training import ModelTrainer
        from src.evaluation import ModelEvaluator

        # 1. Data Preprocessing
        logger.info("=== STAGE 1: Data Preprocessing ===")
        preprocessor = DataPreprocessor()
        sample_data_path = "/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/data/processed/merged_data.csv"
        if not os.path.exists(sample_data_path):
            logger.info("Creating sample data structure...")
            create_sample_data()

        df = preprocessor.load_data("merged_data")

        if df is not None:
            df = preprocessor.prepare_dataset(df, 'report_text_us', 'severity_grade_us')
            train_df, val_df, test_df = preprocessor.split_data(df)
            preprocessor.save_processed_data(train_df, val_df, test_df)
            logger.info("✅ Preprocessing stage completed successfully!")
        else:
            logger.warning("No data found. Please add your dataset to data/raw/")
            return

        # 2. Feature Engineering
        logger.info("=== STAGE 2: Feature Engineering ===")
        feature_engineer = FeatureEngineer()
        train_df = pd.read_csv("/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/data/processed/train_data.csv")
        test_df = pd.read_csv("/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/data/processed/test_data.csv")
        if train_df is not None and test_df is not None:
            X_train_tfidf, X_test_tfidf = feature_engineer.extract_tfidf_features(
                train_df['processed_text'],
                test_df['processed_text']
            )
            keyword_vocab = feature_engineer.build_keyword_vocabulary(train_df['medical_keywords'])
            X_train_keywords = feature_engineer.create_keyword_features(train_df['medical_keywords'], keyword_vocab)
            X_test_keywords = feature_engineer.create_keyword_features(test_df['medical_keywords'], keyword_vocab)
            X_train_stats = feature_engineer.create_text_statistics(train_df['processed_text'])
            X_test_stats = feature_engineer.create_text_statistics(test_df['processed_text'])
            X_train_combined = feature_engineer.combine_features([X_train_tfidf, X_train_keywords, X_train_stats])
            X_test_combined = feature_engineer.combine_features([X_test_tfidf, X_test_keywords, X_test_stats])
            feature_engineer.save_features(
                [X_train_combined, X_test_combined],
                ['X_train_combined', 'X_test_combined']
            )
            logger.info("✅ Feature engineering stage completed successfully!")

        # 3. Model Training
        logger.info("=== STAGE 3: Model Training ===")
        trainer = ModelTrainer()
        X_train = feature_engineer.load_features(['X_train_combined'])[0]
        if X_train is not None:
            models = trainer.train_baseline_models(X_train, train_df['label_encoded'])
            ensemble = trainer.train_ensemble(X_train, train_df['label_encoded'])
            trainer.save_models()
            logger.info("✅ Training stage completed successfully!")

        # 4. Evaluation
        logger.info("=== STAGE 4: Model Evaluation ===")
        evaluator = ModelEvaluator()
        test_df = pd.read_csv("/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/data/processed/test_data.csv")
        X_test = feature_engineer.load_features(['X_test_combined'])[0]
        if test_df is not None and X_test is not None:
            try:
                best_model = joblib.load("/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/models/best_model.pkl")
            except:
                best_model = joblib.load("/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/models/random_forest_model.pkl")
            results = evaluator.evaluate_model(
                best_model,
                X_test,
                test_df['label_encoded'],
                "Best Model"
            )
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
            os.makedirs('results', exist_ok=True)
            comparison_df.to_csv('/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/results/final_results.csv', index=False)
            logger.info("✅ Evaluation stage completed successfully!")

        logger.info("🎉 CPU-only, single-threaded pipeline completed successfully!")

        # --- CHAINED EVALUATION LOGIC ---
        logger.info("=== CHAINED EVALUATION PIPELINE ===")
        run_chained_evaluation()



    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


def run_chained_evaluation():
    from src.evaluation import ModelEvaluator
    from src.feature_engineering import FeatureEngineer # Import FeatureEngineer
    import joblib
    import pandas as pd # Import pandas

    logger.info("Starting chained evaluation using processed test and train data...")

    # Load processed test data
    test_df = pd.read_csv("data/processed/test_data.csv")
    train_df = pd.read_csv("data/processed/train_data.csv") # Load train data to build keyword vocab

    # Load trained model and TF-IDF vectorizer (must be the one fit on train set!)
    rf_model = joblib.load("models/random_forest_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    # --- START: MODIFIED SECTION ---
    # Recreate all features for the test set
    feature_engineer = FeatureEngineer()

    # 1. TF-IDF Features
    X_test_tfidf = tfidf.transform(test_df['processed_text'].fillna("").astype(str))

    # 2. Keyword Features (requires building vocab from train set)
    keyword_vocab = feature_engineer.build_keyword_vocabulary(train_df['medical_keywords'])
    X_test_keywords = feature_engineer.create_keyword_features(test_df['medical_keywords'], keyword_vocab)

    # 3. Text Statistics Features
    X_test_stats = feature_engineer.create_text_statistics(test_df['processed_text'])

    # 4. Combine all features
    X_test_combined = feature_engineer.combine_features([X_test_tfidf, X_test_keywords, X_test_stats])
    # --- END: MODIFIED SECTION ---

    label_data = test_df['label_encoded']

    evaluator = ModelEvaluator()

    # Use the combined features for evaluation
    rf_results = evaluator.evaluate_model(
        rf_model,
        X_test_combined, # Use the combined features
        label_data,
        "Random Forest"
    )

    # Evaluate manual extraction if available
    try:
        manual_results_df = pd.read_csv("/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/data/raw/adjudicated_gold_standard_labels.csv")
        manual_eval_results = evaluator.evaluate_manual_extraction(manual_results_df)
    except FileNotFoundError:
        manual_eval_results = None

    # Evaluate ChatGPT-4 (ensure OpenAI API key is configured)
    chatgpt_texts = test_df['processed_text'].tolist()
    chatgpt_labels = label_data.tolist()
    chatgpt_results = evaluator.evaluate_chatgpt4(
        chatgpt_texts,
        chatgpt_labels,
        sample_size=103  # You can change to len(test_df) for full test set
    )

    # Combine all results for comparison
    all_results = [rf_results]
    if manual_eval_results is not None:
        all_results.append(manual_eval_results)
    all_results.append(chatgpt_results)
    comparison_df = evaluator.create_comparison_table(all_results)

    # Display and export the comparison table
    print("\n=== CHAINED FINAL COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))
    print("\n--- Markdown Table ---\n")
    print(comparison_df.to_markdown(index=False))

    # Save results for dissertation (CSV, LaTeX, Markdown)
    comparison_df.to_csv('/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/results/chained_method_comparison.csv', index=False)
    comparison_df.to_latex('/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/results/chained_method_comparison.tex', index=False)
    with open('/Users/abhk/Documents/Git/pediatric_appendicitis_labelling/results/chained_method_comparison.md', 'w') as mdfile:
        mdfile.write(comparison_df.to_markdown(index=False))

    evaluator.save_evaluation_results(all_results, comparison_df)

    print("✅ Chained evaluation stages completed successfully!")


def create_sample_data():
    """Create sample data for testing the pipeline."""
    import pandas as pd
    import os

    os.makedirs("data/raw", exist_ok=True)

    # Create more comprehensive sample dataset
    sample_data = {
        'report_text': [
            "Ultrasound shows inflamed appendix measuring 8mm with surrounding fluid. Findings consistent with acute appendicitis.",
            "Operative findings: Gangrenous appendix with localized perforation. Purulent fluid in pelvis.",
            "US: Normal appendix visualized measuring 4mm. No evidence of appendicitis.",
            "OR: Simple acute appendicitis. Appendix erythematous and swollen but no gangrene or perforation.",
            "Ultrasound: Appendix not clearly visualized. Non-specific inflammatory changes in RLQ.",
            "Operative report: Perforated appendix with abscess formation. Extensive contamination.",
            "US: Appendix diameter 7mm with wall thickening and hyperemia. Suggestive of early appendicitis.",
            "OR: Gangrenous changes of appendix without perforation. Minimal free fluid.",
            "Ultrasound: Unremarkable appendix. No signs of inflammation.",
            "Operative findings: Simple appendicitis with fibrinous exudate.",
        ],
        'severity_grade': [
            'Simple/Uncomplicated',
            'Perforated',
            'Normal',
            'Simple/Uncomplicated',
            'Simple/Uncomplicated',
            'Perforated',
            'Simple/Uncomplicated',
            'Gangrenous',
            'Normal',
            'Simple/Uncomplicated'
        ],
        'manually_extracted_grade': [
            'Simple/Uncomplicated',
            'Perforated',
            'Normal',
            'Simple/Uncomplicated',
            'Simple/Uncomplicated',
            'Perforated',
            'Simple/Uncomplicated',
            'Gangrenous',
            'Normal',
            'Simple/Uncomplicated'
        ],
        'surgeon_adjudicated_grade': [
            'Simple/Uncomplicated',
            'Perforated',
            'Normal',
            'Simple/Uncomplicated',
            'Simple/Uncomplicated',
            'Perforated',
            'Simple/Uncomplicated',
            'Gangrenous',
            'Normal',
            'Simple/Uncomplicated'
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv("data/raw/merged_data.csv", index=False)
    logger.info("Sample data created at data/raw/merged_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paediatric Appendicitis AI Pipeline (CPU-only, single-threaded)')
    parser.add_argument('--stage', type=str,
                        choices=['all', 'preprocessing', 'features', 'training', 'evaluation', 'create_sample', 'chained_eval'],
                        default='all', help='Pipeline stage to run')

    args = parser.parse_args()

    if args.stage == 'create_sample':
        create_sample_data()
    elif args.stage == 'all':
        run_full_pipeline()
    elif args.stage == 'chained_eval':
        run_chained_evaluation()
    else:
        logger.info(f"Individual stage execution not implemented in CPU-only version. Use 'all' to run full pipeline.")