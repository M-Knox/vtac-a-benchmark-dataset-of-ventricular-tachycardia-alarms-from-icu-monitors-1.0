# analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
from db_connection import get_database
import wfdb
import os
from datetime import datetime, timedelta


class VTaCAnalytics:
    def __init__(self):
        """Initialize the analytics class with database connection"""
        self.db = get_database()

    def get_alarm_distribution(self):
        """Get distribution of true vs false alarms"""
        pipeline = [
            {
                "$group": {
                    "_id": "$is_true_alarm",
                    "count": {"$sum": 1}
                }
            }
        ]

        result = list(self.db.alarm_events.aggregate(pipeline))

        # Convert to dataframe for easier visualization
        df = pd.DataFrame(result)
        df.columns = ['Is True Alarm', 'Count']

        # Visualize
        plt.figure(figsize=(10, 6))
        plt.bar(
            ['False Alarm', 'True Alarm'],
            [
                df[df['Is True Alarm'] == False]['Count'].values[0] if False in df['Is True Alarm'].values else 0,
                df[df['Is True Alarm'] == True]['Count'].values[0] if True in df['Is True Alarm'].values else 0
            ]
        )
        plt.title('Distribution of True vs False VT Alarms')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('alarm_distribution.png')
        plt.close()

        return df

    def get_hospital_distribution(self):
        """
        Get distribution of alarms by hospital
        Note: This would work if hospital information was available in the data
        """
        pipeline = [
            {
                "$lookup": {
                    "from": "patients",
                    "localField": "patient_id",
                    "foreignField": "patient_id",
                    "as": "patient_info"
                }
            },
            {
                "$unwind": "$patient_info"
            },
            {
                "$group": {
                    "_id": "$patient_info.hospital",
                    "count": {"$sum": 1}
                }
            }
        ]

        result = list(self.db.alarm_events.aggregate(pipeline))

        # If hospital data is available, visualize it
        if result and any(r.get('_id') for r in result):
            df = pd.DataFrame(result)
            plt.figure(figsize=(10, 6))
            plt.bar(df['_id'], df['count'])
            plt.title('Distribution of Alarms by Hospital')
            plt.ylabel('Count')
            plt.xlabel('Hospital')
            plt.tight_layout()
            plt.savefig('hospital_distribution.png')
            plt.close()
            return df
        else:
            print("Hospital information not available in the dataset")
            return None

    def get_patient_with_most_false_alarms(self):
        """Identify patients with the highest number of false alarms"""
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

        result = list(self.db.alarm_events.aggregate(pipeline))

        # Convert to dataframe for easier visualization
        df = pd.DataFrame(result)
        df.columns = ['Patient ID', 'False Alarm Count']

        # Visualize
        plt.figure(figsize=(12, 6))
        plt.bar(df['Patient ID'], df['False Alarm Count'])
        plt.title('Patients with Most False Alarms')
        plt.ylabel('False Alarm Count')
        plt.xlabel('Patient ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('patients_most_false_alarms.png')
        plt.close()

        return df

    def analyze_signals_for_event(self, event_id):
        """
        Analyze the waveform signals for a specific event
        This demonstrates NoSQL's ability to handle time-series data
        """
        # Get event information
        event = self.db.alarm_events.find_one({"event_id": event_id})
        if event is None:
            print(f"Event {event_id} not found")
            return None

        # Break down the pipeline into smaller steps to avoid memory issues
        # Step 1: Get distinct signal types for this event
        signal_types = self.db.waveform_data.distinct(
            "metadata.signal_type",
            {"metadata.event_id": event_id}
        )

        if not signal_types:
            print(f"No waveform data found for event {event_id}")
            return None

        # Create a figure for plotting
        plt.figure(figsize=(15, 10))
        result_data = []

        # Process each signal type separately
        for i, signal_type in enumerate(signal_types):
            # Get data for this signal type with a limit to prevent memory issues
            cursor = self.db.waveform_data.find(
                {
                    "metadata.event_id": event_id,
                    "metadata.signal_type": signal_type
                }
            ).sort("timestamp", 1).limit(5000)  # Limiting to prevent memory issues

            try:
                timestamps = []
                values = []
                for doc in cursor:
                    timestamps.append(doc["timestamp"])
                    values.append(doc["value"])
            except pymongo.errors.NetworkTimeout:
                print(f"Timeout while retrieving data for signal {signal_type}. Continuing with available data.")


            # Skip if no data
            if  values is None:
                continue

            # Convert to numpy array for easier processing
            values_array = np.array(values)

            # Create time axis (assuming 250Hz sampling rate)
            time_axis = np.arange(len(values)) / 250  # seconds

            # Store data for return
            result_data.append({
                "_id": signal_type,
                "timestamps": timestamps,
                "values": values
            })

            # Plot in a subplot
            plt.subplot(len(signal_types), 1, i + 1)
            plt.plot(time_axis, values_array)
            plt.title(f'Signal: {signal_type}')
            plt.ylabel('Amplitude')

            # Mark the alarm point (5-minute mark)
            plt.axvline(x=300, color='r', linestyle='--', label='Alarm')

            if i == len(signal_types) - 1:
                plt.xlabel('Time (seconds)')

        plt.suptitle(f'Waveform Analysis for Event {event_id} - {"True" if event["is_true_alarm"] else "False"} Alarm')
        plt.tight_layout()
        plt.savefig(f'waveform_analysis_{event_id}.png')
        plt.close()

        return result_data

    def run_basic_analytics(self):
        """Run all basic analytics functions"""
        print("Running VTaC Analytics...")

        # Check if we have data
        alarm_count = self.db.alarm_events.count_documents({})
        if alarm_count == 0:
            print("No data found in the database. Please load data first.")
            return

        print(f"Found {alarm_count} alarm events in database.")

        # Run analytics with error handling
        try:
            print("Analyzing alarm distribution...")
            self.get_alarm_distribution()
        except Exception as e:
            print(f"Error during alarm distribution analysis: {e}")

        try:
            print("Analyzing hospital distribution...")
            self.get_hospital_distribution()
        except Exception as e:
            print(f"Error during hospital distribution analysis: {e}")

        try:
            print("Finding patients with most false alarms...")
            self.get_patient_with_most_false_alarms()
        except Exception as e:
            print(f"Error finding patients with most false alarms: {e}")

        # Find one event with waveform data for analysis with reduced amount of data
        try:
            # Only find an event_id, don't try to process all data at once
            waveform_doc = self.db.waveform_data.find_one({}, {"metadata.event_id": 1})
            if waveform_doc:
                event_id = waveform_doc['metadata']['event_id']
                print(f"Analyzing waveform data for event {event_id}...")
                try:
                    self.analyze_signals_for_event(event_id)
                except Exception as e:
                    print(f"Error analyzing waveform data for event {event_id}: {e}")
        except Exception as e:
            print(f"Error finding event with waveform data: {e}")

        print("Analytics completed. Check any generated visualization files.")

if __name__ == "__main__":
    # Create and run analytics
    analytics = VTaCAnalytics()
    analytics.run_basic_analytics()