# CrewAI MLB Analysis Container
# Base: Python 3.11 slim for CrewAI compatibility
FROM python:3.11-slim

# Install system dependencies for ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
# CrewAI with tools, plus MLB-specific packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    'crewai[tools]' \
    'langchain-openai' \
    'langchain-community' \
    pandas \
    numpy \
    requests \
    python-dotenv

# Copy application code
COPY . .

# Default command
CMD ["python", "main.py"]