#!/usr/bin/env python3
"""
Entry point for the Cloud FinOps Environment server.

Starts the FastAPI server on port 7860 for Hugging Face Spaces deployment.
"""

import os
import uvicorn
from my_env.server.app import app

if __name__ == "__main__":
    # Get configuration from environment variables
    workers = int(os.getenv("WORKERS", "4"))
    port = int(os.getenv("PORT", "7860"))
    
    # Run on port 7860 as required by hackathon
    # Configure for WebSocket support with proper settings
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Use 1 worker for HF Spaces to avoid port conflicts
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=30
    )
