"""List available models in the new Client API"""
from google import genai
import os

# Configure API
GEMINI_API_KEY = "AIzaSyBU4ls0_sslkyKRGHNJXzntEyx68PEPeGA"
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
client = genai.Client()

print("Available models:\n")
try:
    for model in client.models.list():
        print(f"  • {model.name}")
        if 'flash' in model.name.lower():
            print(f"    ↳ FLASH MODEL FOUND!")
except Exception as e:
    print(f"Error: {e}")
