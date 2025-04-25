# model_deployment.py
import os
import pickle
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
from flask import jsonify
from db_connection import get_database


class VTaCModelDeployment:
    def __init__(self, model_dir='models'):
        """
        Initialize the model deployment

        Args:
            model_dir: Directory containing trained models
        """
        self.db = get_database()
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None

        try:
            # Load the latest model
            self.load_latest_model()
        except Exception as e:
            print(f"Warning: Could not load model during initialization: {e}")

    def load_latest_model(self):
        """
        Load the most recent model from the models directory

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:

            #Check if the model directory exists
            if not os.path.exists(self.model_dir):
                os.makesdir(self.model_dir)
                print(f"Created missing model directory: {self.model_dir}")
                return False

            # Find all model files
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl') and f != 'feature_names.pkl']

            if not model_files:
                print("No model files found in the models directory")
                return False

            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)), reverse=True)
            latest_model = model_files[0]

            # Load the model
            model_path = os.path.join(self.model_dir, latest_model)
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            print(f"Loaded model from {model_path}")

            # Load feature names
            feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"Loaded feature names from {feature_path}")
            else:
                print("Feature names file not found")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # Debug - after loading
        if self.model is not None:
            print(f"Model loaded successfully, type: {type(self.model).__name__}")
        else:
            print("Failed to load model")

        if self.feature_names is not None:
            print(f"Feature names loaded successfully, count: {len(self.feature_names)}")
        else:
            print("Failed to load feature names")

    def load_specific_model(self, model_filename):
        """
        Load a specific model by filename

        Args:
            model_filename: Name of the model file

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = os.path.join(self.model_dir, model_filename)

            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found")
                return False

            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            print(f"Loaded model from {model_path}")

            # Load feature names
            feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"Loaded feature names from {feature_path}")
            else:
                print("Feature names file not found")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def extract_features_for_event(self, event_id):
        """
        Extract features for a specific event

        Args:
            event_id: ID of the alarm event

        Returns:
            DataFrame of features for the event
        """

        try:
            from model_builder import VTaCModelBuilder

            # Create a model builder to extract features
            builder = VTaCModelBuilder()

            # Get event data
            event = self.db.alarm_events.find_one({"event_id": event_id})
            if not event:
                print(f"Event {event_id} not found")
                return None

            # Get all available signal types for this event
            signal_types = self.db.waveform_data.distinct(
                "metadata.signal_type",
                {"metadata.event_id": event_id}
            )

            if not signal_types:
                print(f"No signal data found for event {event_id}")
                return None

            # Extract features
            features = {"event_id": event_id}

            # Add metadata features
            metadata_features = builder.extract_metadata_features(event_id)
            if metadata_features:
                features.update(metadata_features)

            # Process each signal type
            for signal_type in signal_types:
                # Statistical features
                stat_features = builder.extract_statistical_features(event_id, signal_type)
                if stat_features:
                    features.update(stat_features)

                # Signal crossing features
                crossing_features = builder.extract_signal_crossing_features(event_id, signal_type)
                if crossing_features:
                    features.update(crossing_features)

            # Convert to dataframe
            features_df = pd.DataFrame([features])
            features_df.set_index('event_id', inplace=True)

            return features_df

        except ImportError:
            print("Could not import VTaCModelBuilder - feature extraction unavailable")
            return pd.DataFrame([{"event_id": event_id}]).set_index('event_id')

    def predict_alarm_validity(self, event_id):
        """
        Predict whether an alarm is true or false

        Args:
            event_id: ID of the alarm event

        Returns:
            Dictionary with prediction results
        """
        # Check if model is loaded
        if self.model is None:
            return {
                "error": "Model not loaded",
                "prediction": False,
                "probability": 0.5,
                "confidence": 0.01
            }

        # Extract features
        features_df = self.extract_features_for_event(event_id)

        if features_df is None or features_df.empty:
            return {
                "error": "Could not extract features for event",
                "prediction": None,
                "probability": None,
                "confidence": None
            }

        # Ensure we have all required features
        # Ensure we have all required features
        if self.feature_names:

            missing_features_list = [f for f in self.feature_names if f not in features_df.columns]
            if missing_features_list:
                print(f"Missing features: {missing_features_list}")

            # Create a dictionary of missing features
            missing_features = {}
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    missing_features[feature] = [0] * len(features_df)

            # Add all missing features at once
            if missing_features:
                missing_df = pd.DataFrame(missing_features, index=features_df.index)
                features_df = pd.concat([features_df, missing_df], axis=1)

            # Keep only the features the model expects
            features_df = features_df[self.feature_names]

            # Debug - print final feature set
            print(f"Final feature set shape: {features_df.shape}")

        # Make prediction
        try:
            prediction = bool(self.model.predict(features_df)[0])
            probability = float(self.model.predict_proba(features_df)[0][1])  # Probability of true alarm

            # Calculate confidence (distance from 0.5)
            confidence =  max(0.01, abs(probability - 0.5) * 2)  # Minimum confidence of 1%

            # Debug line
            print(f"Raw prediction: {prediction}, probability: {probability}")

            return {
                "event_id": event_id,
                "prediction": prediction,  # True = true alarm, False = false alarm
                "prediction_label": "True Alarm" if prediction else "False Alarm",
                "probability": probability,  # Probability of being a true alarm
                "confidence": confidence,  # Confidence in the prediction
                "error": None
            }

        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                "error": f"Error making prediction: {str(e)}",
                "prediction": None,
                "probability": None,
                "confidence": None
            }

    def batch_predict(self, event_ids):
        """
        Make predictions for a batch of events

        Args:
            event_ids: List of event IDs

        Returns:
            Dictionary mapping event IDs to predictions
        """
        results = {}

        for event_id in event_ids:
            results[event_id] = self.predict_alarm_validity(event_id)

        return results


# Flask route handlers for model deployment
def register_model_routes(app):
    """
    Register model-related routes with the Flask app

    Args:
        app: Flask application instance
    """
    model_deployment = VTaCModelDeployment()

    @app.route('/api/model/predict/<event_id>')
    def predict_alarm(event_id):
        """API endpoint to predict if an alarm is true or false"""
        try:
            result = model_deployment.predict_alarm_validity(event_id)

            if result["error"]:
                return jsonify({"error": result["error"]}), 400

            return jsonify(result)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in predict_alarm: {error_details}")
            return jsonify({
                'error': f'Error making prediction: {str(e)}',
                'details': error_details
            }), 500

    @app.route('/api/model/status')
    def model_status():
        """API endpoint to check model status"""
        try:
            # Check if model is loaded
            if model_deployment.model is None:
                return jsonify({
                    "status": "No model loaded",
                    "ready": False
                })

            # Get model details if available
            model_type = type(model_deployment.model).__name__

            # Get metrics if available
            metrics_path = os.path.join(model_deployment.model_dir, 'model_metrics.json')
            metrics = None
            if os.path.exists(metrics_path):
                metrics = pd.read_json(metrics_path, typ='series').to_dict()

            return jsonify({
                "status": "Model loaded and ready",
                "model_type": model_type,
                "feature_count": len(model_deployment.feature_names) if model_deployment.feature_names else None,
                "metrics": metrics,
                "ready": True
            })

        except Exception as e:
            return jsonify({
                "status": f"Error checking model status: {str(e)}",
                "ready": False
            })


if __name__ == "__main__":
    # Test the model deployment
    deployment = VTaCModelDeployment()

    # Check if we have a model
    if deployment.model is None:
        print("No model loaded. Please train a model first.")
    else:
        # Find an event to test with
        db = get_database()
        test_event = db.waveform_data.distinct("metadata.event_id", {})

        if test_event:
            # Make a prediction
            result = deployment.predict_alarm_validity(test_event[0])
            print(f"Prediction for event {test_event[0]}:")
            print(result)
        else:
            print("No events found with waveform data")