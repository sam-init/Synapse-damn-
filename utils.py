import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = "Synapse"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
print("ROLE:", supabase.auth.get_user())


def upload_to_supabase(file_path, file_name=None):
    """Uploads a file to Supabase Storage and returns its public URL."""
    
    file_name = file_name or os.path.basename(file_path)

    with open(file_path, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(
            file_name, f, {"content-type": "application/octet-stream"}
        )

    # Get public URL
    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
    return public_url
