# db_connection.py
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi


def get_database():
    """
    Connect to MongoDB Atlas and return the database object
    """
    # Replace with your MongoDB Atlas connection string
    # Format: mongodb+srv://<username>:<password>@<cluster-url>/
    connection_string = os.environ.get('MONGODB_URI',
                                       'mongodb+srv://M_Knox:f9Iao1SEGTrU3ToT@vcat.wmhffwc.mongodb.net/?retryWrites=true&w=majority&appName=VCaT')

    # Create a client connection
    client = MongoClient(connection_string,
                         server_api=ServerApi('1'),
                         serverSelectionTimeoutMS=120000,
                         socketTimeoutMS=90000,
                         connectTimeoutMS=60000
                         )

    # Test the connection
    try:
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        return None

    # Return the healthcare database
    db = client['healthcare_vtac']
    return db


def initialize_collections(db):
    """
    Initialize collections with validators if they don't exist
    """
    # Create patient collection if it doesn't exist
    if 'patients' not in db.list_collection_names():
        db.create_collection('patients')
        print("Created patients collection")

    # Create alarm_events collection if it doesn't exist
    if 'alarm_events' not in db.list_collection_names():
        db.create_collection('alarm_events')
        print("Created alarm_events collection")

    # Create time series collection for waveform data if it doesn't exist
    if 'waveform_data' not in db.list_collection_names():
        db.create_collection(
            'waveform_data',
            timeseries={
                'timeField': 'timestamp',
                'metaField': 'metadata',
                'granularity': 'seconds'
            }
        )
        print("Created waveform_data time series collection")

    # Create indexes for better query performance
    db.patients.create_index('patient_id', unique=True)
    db.alarm_events.create_index('event_id', unique=True)
    db.alarm_events.create_index('patient_id')
    db.waveform_data.create_index([('event_id', 1), ('signal_type', 1)])

    print("Database initialization complete")


if __name__ == "__main__":
    # Test the connection
    db = get_database()
    if db is not None:  # Explicit None comparison
        initialize_collections(db)
    else:
        print("Failed to initialize database connection")