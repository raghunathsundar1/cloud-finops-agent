#!/usr/bin/env python3
"""
Entry point for the Cloud FinOps Environment server.

Starts the FastAPI server on port 7860 for Hugging Face Spaces deployment.
"""

import uvicorn
from my_env.server.app import app

if __name__ == "__main__":
    # Run on port 7860 as required by hackathon
    uvicorn.run(app, host="0.0.0.0", port=7860)
