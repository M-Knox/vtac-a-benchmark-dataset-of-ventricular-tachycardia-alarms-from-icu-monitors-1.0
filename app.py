# app.py
from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import sys
from matplotlib.figure import Figure
from db_connection import get_database
from model_deployment import register_model_routes, VTaCModelDeployment

app = Flask(__name__)

# Get database connection
db = get_database()

# Initialize model deployment
model_deployment = VTaCModelDeployment()

# Register model routes
register_model_routes(app)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/alarm-stats')
def alarm_stats():
    """Get alarm statistics"""
    # Get alarm distribution
    pipeline = [
        {
            "$group": {
                "_id": "$is_true_alarm",
                "count": {"$sum": 1}
            }
        }
    ]
    result = list(db.alarm_events.aggregate(pipeline))

    # Format for the frontend
    false_count = 0
    true_count = 0
    for item in result:
        if item['_id'] == True:
            true_count = item['count']
        else:
            false_count = item['count']

    total_count = true_count + false_count
    true_percentage = (true_count / total_count * 100) if total_count > 0 else 0
    false_percentage = (false_count / total_count * 100) if total_count > 0 else 0

    stats = {
        'true_count': true_count,
        'false_count': false_count,
        'total_count': total_count,
        'true_percentage': round(true_percentage, 2),
        'false_percentage': round(false_percentage, 2)
    }

    return jsonify(stats)


@app.route('/api/top-patients')
def top_patients():
    """Get patients with most false alarms"""
    pipeline = [
        {
            "$match": {"is_true_alarm": False}
        },
        {
            "$group": {
                "_id": "$patient_id",
                "false_alarm_count": {"$sum": 1}
            }
        },
        {
            "$sort": {"false_alarm_count": -1}
        },
        {
            "$limit": 10
        }
    ]
    result = list(db.alarm_events.aggregate(pipeline))

    # Format for the frontend
    patients = []
    for item in result:
        patients.append({
            'patient_id': item['_id'],
            'false_alarm_count': item['false_alarm_count']
        })

    return jsonify(patients)


@app.route('/api/waveform-image/<event_id>')
def waveform_image(event_id):
    """Generate waveform image for a specific event"""
    try:
        # Get event information
        event = db.alarm_events.find_one({"event_id": event_id})
        if not event:
            return jsonify({"error": "Event not found"}), 404

        # Get waveform data
        waveform_data = list(db.waveform_data.find({"metadata.event_id": event_id}))
        if not waveform_data:
            return jsonify({"error": "No waveform data found for this event"}), 404

        # Create figure and subplots
        fig, axs = plt.subplots(len(waveform_data), 1, figsize=(10, 2 * len(waveform_data)))

        # Handle single plot case
        if len(waveform_data) == 1:
            axs = [axs]

        # Plot each waveform
        for i, data in enumerate(waveform_data):
            signal_type = data['metadata']['signal_type']
            signal_data = np.array(data['data'])
            time_data = np.arange(len(signal_data)) / data['metadata'].get('sample_rate',
                                                                           250)  # Assume 250Hz if not specified

            axs[i].plot(time_data, signal_data)
            axs[i].set_title(f"{signal_type}")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Amplitude")

        # Add event information
        alarm_type = event.get('alarm_type', 'Unknown')
        is_true = "True" if event.get('is_true_alarm', False) else "False"
        plt.suptitle(f"Event: {event_id}, Alarm Type: {alarm_type}, True Alarm: {is_true}")

        plt.tight_layout()

        # Save to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Encode to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            "image": f"data:image/png;base64,{img_base64}",
            "event_id": event_id,
            "alarm_type": alarm_type,
            "is_true_alarm": event.get('is_true_alarm', False)
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route('/api/events')
def get_events():
    """Get list of alarm events with pagination"""
    # Get query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    filter_true = request.args.get('true_alarms')
    filter_false = request.args.get('false_alarms')

    # Build query
    query = {}
    if filter_true == 'true' and filter_false != 'true':
        query['is_true_alarm'] = True
    elif filter_true != 'true' and filter_false == 'true':
        query['is_true_alarm'] = False

    # Get total count
    total_count = db.alarm_events.count_documents(query)

    # Get paginated results
    events = list(db.alarm_events.find(query)
                  .skip((page - 1) * per_page)
                  .limit(per_page))

    # Format results
    result = []
    for event in events:
        event['_id'] = str(event['_id'])  # Convert ObjectId to string
        result.append(event)

    return jsonify({
        'events': result,
        'total': total_count,
        'page': page,
        'per_page': per_page,
        'pages': (total_count + per_page - 1) // per_page
    })


@app.route('/api/event/<event_id>')
def get_event(event_id):
    """Get details for a specific event"""
    event = db.alarm_events.find_one({"event_id": event_id})

    if not event:
        return jsonify({"error": "Event not found"}), 404

    # Convert ObjectId to string
    event['_id'] = str(event['_id'])

    # Get prediction if available
    prediction = model_deployment.predict_alarm_validity(event_id)

    return jsonify({
        "event": event,
        "prediction": prediction
    })


@app.route('/api/analyze-event/<event_id>')
def analyze_event(event_id):
    """Analyze an event and provide detailed information"""
    try:
        # Get event data
        event = db.alarm_events.find_one({"event_id": event_id})
        if not event:
            return jsonify({"error": "Event not found"}), 404

        # Get features
        features_df = model_deployment.extract_features_for_event(event_id)
        if features_df is None:
            return jsonify({"error": "Could not extract features"}), 500

        # Get prediction
        prediction = model_deployment.predict_alarm_validity(event_id)

        # Get waveform data
        signals = db.waveform_data.distinct("metadata.signal_type", {"metadata.event_id": event_id})

        # Format features for display
        features_dict = features_df.to_dict(orient='records')[0] if not features_df.empty else {}

        return jsonify({
            "event": {
                "event_id": event_id,
                "patient_id": event.get("patient_id", "Unknown"),
                "alarm_type": event.get("alarm_type", "Unknown"),
                "is_true_alarm": event.get("is_true_alarm", False),
                "timestamp": event.get("timestamp", "Unknown"),
            },
            "features": features_dict,
            "prediction": prediction,
            "available_signals": signals
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route('/api/features/<event_id>')
def get_features(event_id):
    """Get extracted features for an event"""
    features_df = model_deployment.extract_features_for_event(event_id)

    if features_df is None:
        return jsonify({"error": "Could not extract features"}), 500

    return jsonify(features_df.to_dict(orient='records')[0] if not features_df.empty else {})


try:
    from model_deployment import register_model_routes, VTaCModelDeployment

    model_deployment = VTaCModelDeployment()
except Exception as e:
    print(f"Warning: Could not initialize model deployment: {e}")


    # Define placeholder functions and classes if needed
    def register_model_routes(app):
        @app.route('/api/model/status')
        def model_status():
            return jsonify({"status": "Model not available", "ready": False})


    class VTaCModelDeployment:
        def predict_alarm_validity(self, event_id):
            return {"error": "Model not available", "prediction": False}


if __name__ == '__main__':
    app.run(debug=True)


