import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
