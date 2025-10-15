import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
import yaml
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.tfidf_vectorizer = None
        self.onehot_encoder = None

        # Get categorical columns from config or define here
        self.categorical_cols = self.config['features'].get('categorical_cols', [])

    def extract_tfidf_features(self, train_texts, test_texts, val_texts=None):
        """Extract TF-IDF features from text."""
        tfidf_config = self.config['features']['tfidf']

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_texts)
        directory_name = "models"
        try:
            import os
            os.mkdir(directory_name)
            logger.info(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            logger.info(f"Directory '{directory_name}' already exists.")
        except Exception as e:
            logger.info(f"An error occurred: {e}")
        joblib.dump(self.tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
        X_test_tfidf = self.tfidf_vectorizer.transform(test_texts)

        logger.info(f"TF-IDF features shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")

        if val_texts is not None:
            X_val_tfidf = self.tfidf_vectorizer.transform(val_texts)
            return X_train_tfidf, X_val_tfidf, X_test_tfidf

        return X_train_tfidf, X_test_tfidf

    def extract_onehot_features(self, df_train, df_test, df_val=None):
        """Extract one-hot encoding for categorical columns and align splits."""
        if not self.categorical_cols:
            logger.warning("No categorical columns specified for one-hot encoding.")
            return None, None, None

        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.onehot_encoder.fit(df_train[self.categorical_cols])

        X_train_oh = self.onehot_encoder.transform(df_train[self.categorical_cols])
        X_test_oh = self.onehot_encoder.transform(df_test[self.categorical_cols])
        X_val_oh = self.onehot_encoder.transform(df_val[self.categorical_cols]) if df_val is not None else None

        oh_columns = self.onehot_encoder.get_feature_names_out(self.categorical_cols)
        X_train_oh_df = pd.DataFrame(X_train_oh, columns=oh_columns, index=df_train.index)
        X_test_oh_df = pd.DataFrame(X_test_oh, columns=oh_columns, index=df_test.index)
        X_val_oh_df = pd.DataFrame(X_val_oh, columns=oh_columns, index=df_val.index) if df_val is not None else None

        logger.info(f"One-hot features shape - Train: {X_train_oh_df.shape}, Test: {X_test_oh_df.shape}")
        return X_train_oh_df, X_val_oh_df, X_test_oh_df

    def create_ner_features(self, entity_lists, entity_vocab):
        """Create binary features for medical entities."""
        features = np.zeros((len(entity_lists), len(entity_vocab)))
        for i, entities in enumerate(entity_lists):
            for entity in entities:
                if entity in entity_vocab:
                    features[i, entity_vocab[entity]] = 1
        return features

    def build_entity_vocabulary(self, entity_lists, min_freq=5):
        """Build vocabulary of medical entities."""
        from collections import Counter
        entity_counter = Counter()
        for entities in entity_lists:
            entity_counter.update(entities)
        entity_vocab = {entity: idx for idx, (entity, count) in
                        enumerate(entity_counter.items()) if count >= min_freq}
        logger.info(f"Built entity vocabulary with {len(entity_vocab)} entities")
        return entity_vocab

    def reduce_dimensionality(self, features, n_components=100):
        """Reduce feature dimensionality using TruncatedSVD."""
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = svd.fit_transform(features)
        logger.info(f"Reduced features from {features.shape[1]} to {n_components} dimensions")
        return reduced_features, svd

    def save_features(self, features, feature_names, suffix=""):
        """Save extracted features."""
        import os
        features_path = self.config['data']['features_path']
        os.makedirs(features_path, exist_ok=True)

        for name, feature in zip(feature_names, features):
            if hasattr(feature, 'toarray'):
                # Sparse matrix
                feature = feature.toarray()

            np.save(f"{features_path}{name}{suffix}.npy", feature)

        # Save vectorizers
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, f"{features_path}tfidf_vectorizer{suffix}.pkl")

        logger.info(f"Features saved to {features_path}")

    def build_keyword_vocabulary(self, keyword_series, save_vocab=True):
        """
        Build a vocabulary (set) of unique keywords from a pandas Series.
        Only fit on train data.
        """
        vocabulary = set()
        for keywords in keyword_series.dropna():
            if isinstance(keywords, str):
                for kw in keywords.split(','):
                    vocabulary.add(kw.strip())
            elif isinstance(keywords, list):
                vocabulary.update(keywords)
        self.keyword_vocab = sorted(vocabulary)
        if save_vocab:
            joblib.dump(self.keyword_vocab, "models/keyword_vocab.pkl")
        return self.keyword_vocab


    def create_keyword_features(self, keyword_series, vocabulary=None):
        """
        Generate a binary feature matrix using a provided vocabulary (from train only).
        """
        import numpy as np
        vocab = vocabulary if vocabulary is not None else self.keyword_vocab
        features = np.zeros((len(keyword_series), len(vocab)), dtype=int)
        for i, keywords in enumerate(keyword_series):
            keyword_set = set()
            if isinstance(keywords, str):
                keyword_set = set([kw.strip() for kw in keywords.split(',') if kw.strip()])
            elif isinstance(keywords, list):
                keyword_set = set(keywords)
            for j, kw in enumerate(vocab):
                if kw in keyword_set:
                    features[i, j] = 1
        return features

    def create_text_statistics(self, text_series):
        """
        Given a pandas Series of text, returns a numpy array of basic statistics:
        - num_chars: number of characters in the text
        - num_words: number of words in the text
        - avg_word_length: average length of words
        """
        import numpy as np

        num_chars = text_series.fillna("").apply(len)
        num_words = text_series.fillna("").apply(lambda x: len(x.split()))
        avg_word_length = text_series.fillna("").apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )

        features = np.vstack([num_chars, num_words, avg_word_length]).T
        return features

    def combine_features(self, feature_list):
        """
        Combines a list of feature matrices horizontally.
        Supports numpy arrays, pandas DataFrames, and scipy sparse matrices.
        Returns a combined numpy array.
        """
        import numpy as np
        from scipy.sparse import issparse

        arrays = []
        for f in feature_list:
            if issparse(f):  # For sparse matrices like csr_matrix
                arrays.append(f.toarray())
            elif isinstance(f, np.ndarray):
                arrays.append(f)
            elif hasattr(f, "values"):  # For pandas DataFrame
                arrays.append(f.values)
            else:
                raise TypeError(f"Unsupported feature type: {type(f)}")
        return np.hstack(arrays)

    def load_features(self, feature_names, feature_dir="data/features/"):
        """
        Loads feature arrays from disk given a list of filenames (without extension).
        Returns a list of numpy arrays.
        """
        import numpy as np
        import os

        arrays = []
        for name in feature_names:
            path = os.path.join(feature_dir, f"{name}.npy")
            arrays.append(np.load(path))
        return arrays


# Example usage
if __name__ == "__main__":
    feature_engineer = FeatureEngineer()

    # Load processed data
    train_df = pd.read_csv("data/processed/train_data.csv")
    test_df = pd.read_csv("data/processed/test_data.csv")

    # Extract TF-IDF features
    X_train_tfidf, X_test_tfidf = feature_engineer.extract_tfidf_features(
        train_df['processed_text'],
        test_df['processed_text']
    )

    # Save features
    feature_engineer.save_features(
        [X_train_tfidf, X_test_tfidf],
        ['X_train_tfidf', 'X_test_tfidf']
    )