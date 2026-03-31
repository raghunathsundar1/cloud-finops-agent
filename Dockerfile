# Use a lightweight Python image
FROM python:3.11-slim

# Set environment variables to prevent Python from buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a working directory inside the container
WORKDIR /app

# Install git for GitHub dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv (the fast package manager you are using)
RUN pip install uv

# Copy your requirements or pyproject.toml first (for caching)
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of your environment files into the container
COPY . /app/

# Expose the strict Hugging Face port
EXPOSE 7860

# Command to run the server
CMD ["python", "server.py"]
