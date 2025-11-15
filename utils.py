import firebase_admin
from firebase_admin import credentials, storage
import os

# Path to your Firebase admin JSON
cred = credentials.Certificate("firebase-admin.json")

# Initialize Firebase
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "storageBucket": "<your-bucket-name>.appspot.com"
    })

bucket = storage.bucket()

def upload_to_firebase(file_path, file_name=None):
    file_name = file_name or os.path.basename(file_path)
    blob = bucket.blob(file_name)

    blob.upload_from_filename(file_path)
    blob.make_public()  # optional

    return blob.public_url
