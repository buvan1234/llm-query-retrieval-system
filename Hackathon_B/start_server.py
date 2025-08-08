#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Intelligent Query–Retrieval System.
This script starts the FastAPI server with proper configuration.
"""

import uvicorn
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    """Start the FastAPI server."""
    print("🚀 Starting LLM-Powered Intelligent Query–Retrieval System...")
    print("=" * 60)
    
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: The following environment variables are not set:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n   Please set these variables in your .env file or environment.")
        print("   You can copy env_example.txt to .env and fill in your API keys.")
        print("\n   The system will use FREE MODELS by default (USE_FREE_MODELS=True)")
        print("   This means it will use local models that don't require API keys.")
        print("   Set USE_FREE_MODELS=False to use OpenAI API instead.")
    
    # Check free model configuration
    use_free_models = os.getenv("USE_FREE_MODELS", "True").lower() == "true"
    if use_free_models:
        print("✅ Using FREE MODELS (local models)")
        print("   - No API keys required")
        print("   - Models run locally on your machine")
        print("   - Set USE_FREE_MODELS=False to use OpenAI API")
    else:
        print("💰 Using PAID MODELS (OpenAI API)")
        print("   - Requires OpenAI API key")
        print("   - Set USE_FREE_MODELS=True to use free local models")
    
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"📡 Server will start on: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/api/v1/health")
    print("\n" + "=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
