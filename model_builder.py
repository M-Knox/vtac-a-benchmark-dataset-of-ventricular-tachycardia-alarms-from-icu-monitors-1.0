# model_builder.py
import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from imblearn.over_sampling import SMOTE
from db_connection import get_database


class VTaCModelBuilder:
    def __init__(self, model_dir='models'):
        """
        Initialize the model builder

        Args:
            model_dir: Directory to save trained models
        """
        self.db = get_database()
        self.model_dir = model_dir

        # Create the model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Initialize empty dataframes for features and labels
        self.features_df = None
        self.labels = None

    def extract_statistical_features(self, event_id, signal_type):
        # Use aggregation to calculate statistics on the server
        pipeline = [
            {"$match": {
                "metadata.event_id": event_id,
                "metadata.signal_type": signal_type
            }},
            {"$group": {
                "_id": None,
                "mean": {"$avg": "$value"},
                "min": {"$min": "$value"},
                "max": {"$max": "$value"},
                "count": {"$sum": 1},
                "values": {"$push": "$value"}  # This might still be large, use with caution
            }}
        ]

        result = list(self.db.waveform_data.aggregate(pipeline))

        if not result:
            return None

        stats = result[0]
        values_array = np.array(stats["values"])

        features = {
            f"{signal_type}_mean": stats["mean"],
            f"{signal_type}_min": stats["min"],
            f"{signal_type}_max": stats["max"],
            f"{signal_type}_range": stats["max"] - stats["min"],
            # Calculate the rest locally with the array
            f"{signal_type}_std": np.std(values_array),
            f"{signal_type}_median": np.median(values_array),
            f"{signal_type}_q25": np.percentile(values_array, 25),
            f"{signal_type}_q75": np.percentile(values_array, 75),
            f"{signal_type}_iqr": np.percentile(values_array, 75) - np.percentile(values_array, 25),
        }

        return features

    def extract_signal_crossing_features(self, event_id, signal_type):
        """
        Extract signal crossing features (zero crossings, threshold crossings)
        that may be relevant for arrhythmia detection
        """
        try:
            # First attempt: retrieve and process data in chunks without MongoDB sorting
            print(f"Extracting features for event {event_id}, signal {signal_type}")

            # Get data without sorting in MongoDB
            cursor = self.db.waveform_data.find({
                "metadata.event_id": event_id,
                "metadata.signal_type": signal_type
            })

            # Process data locally
            data = []
            for doc in cursor:
                data.append((doc.get("timestamp", 0), doc["value"]))

            if not data:
                print(f"No data found for event {event_id}, signal {signal_type}")
                return None

            # Sort locally
            data.sort(key=lambda x: x[0])

            values = [item[1] for item in data]

            if len(values) < 10:
                print(f"Not enough data points ({len(values)}) for event {event_id}, signal {signal_type}")
                return None

            # Process the data
            values_array = np.array(values)
            normalized = values_array - np.mean(values_array)
            zero_crossings = np.sum(np.diff(np.signbit(normalized)))

            thresholds = {
                25: np.percentile(values_array, 25),
                50: np.percentile(values_array, 50),
                75: np.percentile(values_array, 75)
            }

            threshold_crossings = {}
            for p, threshold in thresholds.items():
                above_threshold = values_array > threshold
                crossings = np.sum(np.diff(above_threshold.astype(int)) != 0)
                threshold_crossings[p] = crossings

            features = {
                f"{signal_type}_zero_crossings": zero_crossings,
                f"{signal_type}_25p_crossings": threshold_crossings[25],
                f"{signal_type}_50p_crossings": threshold_crossings[50],
                f"{signal_type}_75p_crossings": threshold_crossings[75],
            }

            return features

        except pymongo.errors.OperationFailure as e:
            print(f"MongoDB operation failed in extract_signal_crossing_features: {str(e)}")
            print("MongoDB query failed - considering reducing dataset size or chunking the data")
            return None
        except Exception as e:
            print(f"Unexpected error in extract_signal_crossing_features: {str(e)}")
            return None

    def extract_metadata_features(self, event_id):
        """
        Extract metadata features from an alarm event

        Args:
            event_id: ID of the alarm event

        Returns:
            Dictionary of metadata features
        """
        # Get the alarm event document
        event = self.db.alarm_events.find_one({"event_id": event_id})

        if not event:
            return None

        features = {}

        # Number of available signals
        if "waveform_metadata" in event and "ecg_leads" in event["waveform_metadata"]:
            features["num_ecg_leads"] = len(event["waveform_metadata"]["ecg_leads"])
        else:
            features["num_ecg_leads"] = 0

        if "waveform_metadata" in event and "pulsatile_signals" in event["waveform_metadata"]:
            features["num_pulsatile_signals"] = len(event["waveform_metadata"]["pulsatile_signals"])
        else:
            features["num_pulsatile_signals"] = 0

        return features

    def build_feature_dataset(self, limit=None):
        """
        Build a dataset of features for all events that have waveform data

        Args:
            limit: Optional limit on number of events to process

        Returns:
            DataFrame of features, numpy array of labels
        """
        print("Building feature dataset...")

        # Find all events with waveform data
        events_with_data = list(self.db.waveform_data.distinct("metadata.event_id"))

        if limit:
            events_with_data = events_with_data[:limit]

        print(f"Found {len(events_with_data)} events with waveform data")

        # For each event, extract features from available signals
        all_features = []
        all_labels = []

        for i, event_id in enumerate(events_with_data):
            if i % 10 == 0:
                print(f"Processing event {i + 1}/{len(events_with_data)}")

            # Get event label (true or false alarm)
            event = self.db.alarm_events.find_one({"event_id": event_id})
            if not event:
                print(f"Event {event_id} not found in alarm_events collection")
                continue

            is_true_alarm = event.get("is_true_alarm", False)

            # Get all available signal types for this event
            signal_types = self.db.waveform_data.distinct(
                "metadata.signal_type",
                {"metadata.event_id": event_id}
            )

            # Extract features for each signal type
            event_features = {"event_id": event_id}

            # Add metadata features
            metadata_features = self.extract_metadata_features(event_id)
            if metadata_features:
                event_features.update(metadata_features)

            # Process each signal type
            for signal_type in signal_types:
                # Statistical features
                stat_features = self.extract_statistical_features(event_id, signal_type)
                if stat_features:
                    event_features.update(stat_features)

                # Signal crossing features
                crossing_features = self.extract_signal_crossing_features(event_id, signal_type)
                if crossing_features:
                    event_features.update(crossing_features)

            # Add to our dataset only if we have features
            if len(event_features) > 1:  # More than just event_id
                all_features.append(event_features)
                all_labels.append(is_true_alarm)

        # Convert to dataframe
        if all_features:
            features_df = pd.DataFrame(all_features)

            # Debug: Print data types to see what's going on
            print("Column data types before processing:")
            print(features_df.dtypes)

            # Set event_id as index but keep it in the dataframe
            features_df.set_index('event_id', inplace=True, drop=False)

            # Handle numeric and non-numeric columns differently
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            object_cols = features_df.select_dtypes(include=['object']).columns

            # Fill numeric columns with mean
            if not numeric_cols.empty:
                features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

            # Fill non-numeric columns (except event_id) with 'unknown'
            for col in object_cols:
                if col != 'event_id':
                    features_df[col] = features_df[col].fillna('unknown')

            # Save as instance variables
            self.features_df = features_df
            self.labels = np.array(all_labels)

            print(f"Feature dataset built with {len(features_df)} events and {features_df.shape[1]} features")

            # Print class distribution
            unique, counts = np.unique(self.labels, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            print(f"Class distribution: {class_distribution}")

            return features_df, np.array(all_labels)
        else:
            print("No features could be extracted")
            return None, None

    def train_random_forest(self, test_size=0.2, n_estimators=100, random_state=42, use_smote=True):
        """
        Train a random forest model on the feature dataset

        Args:
            test_size: Fraction of data to use for testing
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
            use_smote: Whether to use SMOTE for oversampling the minority class

        Returns:
            Trained model, test features, test labels
        """
        # Check if features are available
        if self.features_df is None or self.labels is None:
            print("Feature dataset not built. Building now...")
            self.build_feature_dataset()

            if self.features_df is None or self.labels is None:
                print("Failed to build feature dataset")
                return None, None, None

        # Remove non-numeric columns and event_id
        numeric_cols = self.features_df.select_dtypes(include=['number']).columns
        features_for_model = self.features_df[numeric_cols].drop(columns=['event_id'], errors='ignore')

        # Check for class imbalance
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")

        if len(unique_labels) < 2:
            print("ERROR: Only one class present in the dataset. Cannot train a meaningful model.")
            return None, None, None

        # Use stratified sampling to maintain class proportions
        X_train, X_test, y_train, y_test = train_test_split(
            features_for_model, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels  # This ensures both train and test sets have examples from each class
        )

        # Apply SMOTE to address class imbalance if needed
        if use_smote and len(unique_labels) > 1:
            try:
                print("Applying SMOTE to address class imbalance...")
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Training data shape: {X_train.shape}")
                unique_labels_after, counts_after = np.unique(y_train, return_counts=True)
                print(f"After SMOTE - Class distribution: {dict(zip(unique_labels_after, counts_after))}")
            except Exception as e:
                print(f"SMOTE failed: {str(e)}. Proceeding without oversampling.")

        print(f"Training random forest on {X_train.shape[0]} events, testing on {X_test.shape[0]} events")

        # Check test set class distribution
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print(f"Test set class distribution: {dict(zip(unique_test, counts_test))}")

        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'  # Adjust weights inversely proportional to class frequencies
        )
        model.fit(X_train, y_train)

        # Check feature importance
        feature_importances = pd.DataFrame({
            'feature': features_for_model.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Top 10 most important features:")
        print(feature_importances.head(10))

        # Save feature importances plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'feature_importances.png'))
        plt.close()

        return model, X_test, y_test

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model and produce performance metrics

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of performance metrics
        """
        if model is None:
            print("No model to evaluate")
            return None

        # Make predictions
        y_pred = model.predict(X_test)

        # Check if we have both classes in the test set
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            print(f"WARNING: Test set contains only class {unique_classes[0]}. Some metrics will be undefined.")

        # Calculate metrics with handling for edge cases
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
        }

        # Handle case when there are no positive samples
        try:
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        except:
            metrics['precision'] = 0

        try:
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        except:
            metrics['recall'] = 0

        try:
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        except:
            metrics['f1'] = 0

        # Only calculate AUC if both classes are present
        if len(unique_classes) > 1 and all(count > 0 for count in np.bincount(y_test)):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                metrics['auc'] = roc_auc_score(y_test, y_prob)

                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.4f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.savefig(os.path.join(self.model_dir, 'roc_curve.png'))
                plt.close()
            except Exception as e:
                print(f"Error calculating AUC: {str(e)}")
                metrics['auc'] = float('nan')
        else:
            metrics['auc'] = float('nan')
            print("Cannot calculate AUC because test set does not contain both classes")

        print("Model performance metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))
        plt.close()

        return metrics

    def train_with_cross_validation(self, n_folds=5, n_estimators=100, random_state=42, use_smote=True):
        """
        Train and evaluate model using cross-validation

        Args:
            n_folds: Number of folds for cross-validation
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
            use_smote: Whether to use SMOTE for oversampling

        Returns:
            Best model, average metrics across folds
        """
        # Check if features are available
        if self.features_df is None or self.labels is None:
            print("Feature dataset not built. Building now...")
            self.build_feature_dataset()

            if self.features_df is None or self.labels is None:
                print("Failed to build feature dataset")
                return None, None

        # Remove non-numeric columns and event_id
        numeric_cols = self.features_df.select_dtypes(include=['number']).columns
        features_for_model = self.features_df[numeric_cols].drop(columns=['event_id'], errors='ignore')

        # Check for class imbalance
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")

        if len(unique_labels) < 2:
            print("ERROR: Only one class present in the dataset. Cannot train a meaningful model.")
            return None, None

        # Setup cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        all_metrics = []
        best_model = None
        best_score = -1

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(features_for_model, self.labels)):
            print(f"\nFold {fold + 1}/{n_folds}")

            X_train, X_test = features_for_model.iloc[train_idx], features_for_model.iloc[test_idx]
            y_train, y_test = self.labels[train_idx], self.labels[test_idx]

            # Apply SMOTE to address class imbalance if needed (only on training data)
            if use_smote and len(unique_labels) > 1:
                try:
                    print("Applying SMOTE to address class imbalance...")
                    smote = SMOTE(random_state=random_state)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"After SMOTE - Training data shape: {X_train.shape}")
                except Exception as e:
                    print(f"SMOTE failed: {str(e)}. Proceeding without oversampling.")

            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            fold_metrics = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            # Calculate AUC if possible
            if len(np.unique(y_test)) > 1:
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fold_metrics['auc'] = roc_auc_score(y_test, y_prob)
                except Exception as e:
                    print(f"Error calculating AUC: {str(e)}")
                    fold_metrics['auc'] = float('nan')
            else:
                fold_metrics['auc'] = float('nan')

            all_metrics.append(fold_metrics)

            # Track best model
            if fold_metrics['f1'] > best_score:
                best_score = fold_metrics['f1']
                best_model = model

            print(f"Fold {fold + 1} metrics:")
            for metric, value in fold_metrics.items():
                if metric != 'fold':
                    print(f"{metric}: {value:.4f}")

        # Calculate average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            values = [fold_metrics[metric] for fold_metrics in all_metrics]
            avg_metrics[metric] = np.mean([v for v in values if not np.isnan(v)])

        print("\nAverage metrics across all folds:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Train a final model on the full dataset
        print("\nTraining final model on full dataset...")
        X = features_for_model
        y = self.labels

        if use_smote and len(unique_labels) > 1:
            try:
                smote = SMOTE(random_state=random_state)
                X, y = smote.fit_resample(X, y)
            except Exception as e:
                print(f"SMOTE failed: {str(e)}. Proceeding without oversampling.")

        final_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        final_model.fit(X, y)

        return final_model, avg_metrics

    def save_model(self, model, model_name=None):
        """
        Save the trained model to disk

        Args:
            model: Trained model to save
            model_name: Name for the model file

        Returns:
            Path to the saved model
        """
        if model is None:
            print("No model to save")
            return None

        if model_name is None:
            # Generate a timestamped name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"vt_alarm_model_{timestamp}.pkl"

        model_path = os.path.join(self.model_dir, model_name)

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved to {model_path}")

        # Save feature names for future use
        if self.features_df is not None:
            feature_names = self.features_df.columns.drop('event_id').tolist()
            feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_names, f)
            print(f"Feature names saved to {feature_path}")

        return model_path

    def ensure_indexes(self):
        # Check if the index already exists before creating it
        existing_indexes = list(self.db.alarm_events.list_indexes())
        event_id_index_exists = any(idx.get('name') == 'event_id_1' for idx in existing_indexes)

        if not event_id_index_exists:
            self.db.alarm_events.create_index("event_id")

        # For the compound index, we can just ensure it exists
        self.db.waveform_data.create_index([
            ("metadata.event_id", 1),
            ("metadata.signal_type", 1)
        ])

        print("Indexes created or verified")

    def explore_data(self):
        """Explore the structure of the dataset"""
        if self.features_df is None:
            print("No feature dataset available")
            return

        print("\n--- Data Exploration ---")
        print(f"Dataset shape: {self.features_df.shape}")
        print("\nFeature data types:")
        print(self.features_df.dtypes)

        print("\nSample data:")
        print(self.features_df.head())

        print("\nMissing values per column:")
        print(self.features_df.isnull().sum())

        print("\nClass distribution:")
        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            print(dict(zip(unique, counts)))

    def run_full_pipeline(self, limit=None, use_cross_validation=True, n_folds=5, test_size=0.2, n_estimators=100):
        """
        Run the full model building pipeline

        Args:
            limit: Maximum number of events to process
            use_cross_validation: Whether to use cross-validation
            n_folds: Number of folds for cross-validation
            test_size: Fraction of data to use for testing (if not using cross-validation)
            n_estimators: Number of trees in the random forest

        Returns:
            Path to the saved model
        """
        print("Running full model building pipeline...")

        # Create indexes if they don't exist
        self.ensure_indexes()

        try:
            # Build feature dataset
            self.build_feature_dataset(limit=limit)

            # Explore the data
            self.explore_data()

            # Check if we have enough data for modeling
            if self.features_df is None or len(self.features_df) < 5:
                print("Not enough data for modeling")
                return None

            # Check class distribution
            if self.labels is not None:
                unique_labels = np.unique(self.labels)
                if len(unique_labels) < 2:
                    print("ERROR: Only one class present in the dataset. Cannot train a binary classifier.")
                    print("Please ensure your dataset has examples of both true and false alarms.")
                    return None

            # Train model
            if use_cross_validation:
                model, metrics = self.train_with_cross_validation(
                    n_folds=n_folds,
                    n_estimators=n_estimators
                )
            else:
                model, X_test, y_test = self.train_random_forest(
                    test_size=test_size,
                    n_estimators=n_estimators,
                    use_smote=True
                )

                if model is None:
                    print("Model training failed")
                    return None

                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)

            if model is None:
                print("Model training failed")
                return None

            # Save model
            model_path = self.save_model(model)

            # Save metrics
            if metrics:
                metrics_path = os.path.join(self.model_dir, 'model_metrics.json')
                pd.Series(metrics).to_json(metrics_path)
                print(f"Model metrics saved to {metrics_path}")

            return model_path

        except Exception as e:
            print(f"Error in model building pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Create and run the model builder
    builder = VTaCModelBuilder()
    model_path = builder.run_full_pipeline(
        limit=None,  # Use limit=None for full dataset
        use_cross_validation=True,  # Use cross-validation for small datasets
        n_folds=5,  # Number of cross-validation folds
        n_estimators=100  # Number of trees in the random forest
    )

    print(f"Model building complete. Model saved to {model_path}")