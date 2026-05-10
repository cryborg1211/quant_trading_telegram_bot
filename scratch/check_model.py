import google.generativeai as genai
import os
from dotenv import load_dotenv

# Path to the project .env file
env_path = r"c:\Users\caokh\Desktop\vscode\stock_price_v3\.env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env at", env_path)
else:
    try:
        genai.configure(api_key=api_key)
        print("Querying available Gemini models...\n")
        
        models = genai.list_models()
        
        print(f"{'Model Name':<40} | {'Display Name'}")
        print("-" * 80)
        
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"{m.name:<40} | {m.display_name}")
                
    except Exception as e:
        print(f"ERROR: Failed to connect to Gemini API: {e}")
