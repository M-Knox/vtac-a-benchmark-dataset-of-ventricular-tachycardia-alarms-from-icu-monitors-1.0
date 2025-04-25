# data_loader.py
import os
import pandas as pd
import wfdb
import numpy as np
from datetime import datetime, timedelta
from db_connection import get_database, initialize_collections


class VTaCDataLoader:
    def __init__(self, data_dir, batch_size=100):
        """
        Initialize the data loader

        Args:
            data_dir: Path to the VTaC dataset directory
            batch_size: Number of records to process in batch before DB insert
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.db = get_database()

        # Initialize collections if needed
        if self.db is not None:
            initialize_collections(self.db)

        # Load labels and data split information
        self.labels_df = pd.read_csv(os.path.join(data_dir, 'event_labels.csv'))
        self.data_split_df = pd.read_csv(os.path.join(data_dir, 'benchmark_data_split.csv'))

        # Merge the dataframes to have all information in one place
        self.merged_df = pd.merge(
            self.labels_df,
            self.data_split_df,
            on=['record', 'event'],
            how='left'
        )

    def load_patient_data(self):
        """
        Extract and load patient information into MongoDB
        """
        print("Loading patient data...")

        # Get unique patient records
        patient_records = self.merged_df['record'].unique()

        patients_batch = []
        for record in patient_records:
            # Create patient document
            patient_doc = {
                "patient_id": record,
                "alarm_events": [
                    {"event_id": event}
                    for event in self.merged_df[self.merged_df['record'] == record]['event'].tolist()
                ]
            }
            patients_batch.append(patient_doc)

            # Insert batch if reached batch size
            if len(patients_batch) >= self.batch_size:
                self._insert_patient_batch(patients_batch)
                patients_batch = []

        # Insert any remaining patients
        if patients_batch:
            self._insert_patient_batch(patients_batch)

        print(f"Loaded {len(patient_records)} patient records")

    def _insert_patient_batch(self, patients_batch):
        """Insert a batch of patient documents with upsert"""
        if self.db is None:
            print("Database connection not established")
            return

        for patient in patients_batch:
            self.db.patients.update_one(
                {"patient_id": patient["patient_id"]},
                {"$set": patient},
                upsert=True
            )

    def load_alarm_events(self):
        """
        Extract and load alarm event data into MongoDB
        """
        print("Loading alarm events data...")

        alarm_events_batch = []

        for _, row in self.merged_df.iterrows():
            # Extract timestamp information from filename or use a deterministic value
            # Many medical datasets encode the timestamp in the filename or have it in metadata

            # Option 1: If you have actual timestamps in your data
            # timestamp = extract_timestamp_from_data(row)

            # Modified timestamp generation to avoid date range errors
            baseline_date = datetime(2023, 1, 1, 0, 0, 0)  # January 1, 2023
            event_str = ''.join(filter(str.isdigit, str(row['event'])))
            # Take only last 6 digits to avoid overflow
            if event_str:
                event_num = int(event_str[-6:]) if len(event_str) > 6 else int(event_str)
            else:
                event_num = 0  # Default if no digits found
            event_timestamp = baseline_date + timedelta(minutes=event_num)  # Safer increment

            # Create alarm event document
            alarm_doc = {
                "event_id": row['event'],
                "patient_id": row['record'],
                "is_true_alarm": str(row['decision']).lower() == 'true',
                "alarm_onset_time": event_timestamp,  # Using deterministic timestamp
                "waveform_metadata": {
                    "duration_seconds": 360,  # 6 minutes
                    "sampling_rate": 250,  # Hz as per documentation
                },
                "data_split": row.get('split', 'unknown')  # train/validation/test
            }

            # Get the actual waveform data to extract metadata about available signals
            waveform_path = os.path.join(
                self.data_dir,
                'waveforms',
                row['record'],
                row['event']
            )

            try:
                # Try to read the header file to get signal information
                record = wfdb.rdheader(waveform_path)

                # Extract signal names
                signal_names = record.sig_name

                # Categorize signals into ECG and pulsatile
                ecg_leads = [name for name in signal_names if 'ECG' in name or 'Lead' in name]
                pulsatile_signals = [name for name in signal_names
                                     if any(x in name for x in ['PPG', 'PLETH', 'ABP', 'ART'])]

                # Add to waveform metadata
                alarm_doc["waveform_metadata"]["ecg_leads"] = ecg_leads
                alarm_doc["waveform_metadata"]["pulsatile_signals"] = pulsatile_signals

            except Exception as e:
                print(f"Error reading header for {waveform_path}: {e}")

            alarm_events_batch.append(alarm_doc)

            # Insert batch if reached batch size
            if len(alarm_events_batch) >= self.batch_size:
                self._insert_alarm_events_batch(alarm_events_batch)
                alarm_events_batch = []

        # Insert any remaining events - FIXED: This should be outside the loop
        if alarm_events_batch:
            self._insert_alarm_events_batch(alarm_events_batch)

        # FIXED: This should be outside the loop
        print(f"Loaded {len(self.merged_df)} alarm events")

    def _insert_alarm_events_batch(self, events_batch):
        """Insert a batch of alarm event documents with upsert"""
        if self.db is None:
            print("Database connection not established")
            return

        for event in events_batch:
            self.db.alarm_events.update_one(
                {"event_id": event["event_id"]},
                {"$set": event},
                upsert=True
            )

    def load_waveform_sample(self, num_events=10):
        """
        Load a sample of waveform data into MongoDB time series collection

        Args:
            num_events: Number of events to sample for waveform loading
        """
        print(f"Loading waveform data for {num_events} sample events...")

        # Sample a few events for demonstration
        sample_events = self.merged_df.sample(n=min(num_events, len(self.merged_df)))

        for _, row in sample_events.iterrows():
            waveform_path = os.path.join(
                self.data_dir,
                'waveforms',
                row['record'],
                row['event']
            )

            try:
                # Read the record
                record = wfdb.rdrecord(waveform_path)

                # Generate timestamps (5 min before + 1 min after alarm)
                # Alarm is at the 5-minute mark
                alarm_time = datetime.now()  # Placeholder
                start_time = alarm_time - timedelta(minutes=5)

                # Calculate timestamps for each sample
                num_samples = record.p_signal.shape[0]
                timestamps = [
                    start_time + timedelta(seconds=i / 250)  # 250 Hz
                    for i in range(num_samples)
                ]

                # Process each signal
                for i, signal_name in enumerate(record.sig_name):
                    # Extract signal values
                    signal_values = record.p_signal[:, i]

                    # Prepare data points in batches
                    waveform_docs = []
                    for j in range(0, len(timestamps), self.batch_size):
                        end_idx = min(j + self.batch_size, len(timestamps))
                        batch_docs = [
                            {
                                "metadata": {
                                    "event_id": row['event'],
                                    "signal_type": signal_name
                                },
                                "timestamp": timestamps[k],
                                "value": float(signal_values[k])
                            }
                            for k in range(j, end_idx)
                        ]
                        waveform_docs.extend(batch_docs)

                    # Insert waveform data
                    if waveform_docs:
                        self.db.waveform_data.insert_many(waveform_docs)

                print(f"Loaded waveform data for event {row['event']}")

            except Exception as e:
                print(f"Error loading waveform for {waveform_path}: {e}")

        print(f"Completed loading sample waveform data")

    def run_full_import(self, include_waveforms=False, waveform_sample_size=5):
        """
        Run the full import process

        Args:
            include_waveforms: Whether to include waveform data (can be large)
            waveform_sample_size: Number of events to sample for waveform loading
        """
        if self.db is None:
            print("Database connection not established. Import aborted.")
            return False

        try:
            # Load patient data
            self.load_patient_data()

            # Load alarm events
            self.load_alarm_events()

            # Optionally load waveform samples
            if include_waveforms:
                self.load_waveform_sample(num_events=waveform_sample_size)

            print("Data import completed successfully")
            return True

        except Exception as e:
            print(f"Error during data import: {e}")
            return False


if __name__ == "__main__":
    # Use raw string for path (method 1)
    data_dir = r"C:\Users\Administrator\Desktop\JKUAT Academics\Year Three Sem 2\NoSQL Databases\NoSQL Capstone Project\vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0"

    # Create and run the loader
    loader = VTaCDataLoader(data_dir)
    loader.run_full_import(include_waveforms=True, waveform_sample_size=5)