import os
import sys
import gkeepapi
from dotenv import load_dotenv

def authenticate_google_keep():
    """
    Authenticate with Google Keep using environment variables.
    
    Returns:
        gkeepapi.Keep: Authenticated Keep instance
    """
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve credentials from environment variables
    email = os.getenv('GOOGLE_KEEP_EMAIL')
    master_token = os.getenv('GOOGLE_KEEP_MASTER_TOKEN')
    
    # Validate credentials
    if not email or not master_token:
        print("Error: Missing Google Keep credentials. Set GOOGLE_KEEP_EMAIL and GOOGLE_KEEP_MASTER_TOKEN env vars.")
        sys.exit(1)

    # Initialize Keep instance
    keep = gkeepapi.Keep()

    try:
        # Attempt authentication
        failure = keep.authenticate(email, master_token)
        
        if failure:
            print("Authentication failed. Check your credentials.")
            sys.exit(1)
        
        # Sync notes
        keep.sync()
        
        return keep
    
    except Exception as e:
        print(f"Authentication error: {e}")
        sys.exit(1)

def list_notes(keep):
    """
    List all notes with their details.
    
    Args:
        keep (gkeepapi.Keep): Authenticated Keep instance
    """
    print("Your Google Keep Notes:")
    for note in keep.all():
        print(f"Title: {note.title}")
        print(f"Text: {note.text}")
        print(f"Color: {note.color}")
        print(f"Pinned: {note.pinned}")
        print("---")

def main():
    """
    Main function to authenticate and list notes.
    """
    keep = authenticate_google_keep()
    list_notes(keep)

if __name__ == "__main__":
    main()
