import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Print environment variables
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:10]}...")
print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL')}")
print(f"USE_FREE_MODELS: {os.getenv('USE_FREE_MODELS')}")
print(f"LOCAL_MODEL_NAME: {os.getenv('LOCAL_MODEL_NAME')}")