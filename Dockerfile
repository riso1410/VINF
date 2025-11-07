FROM python:3.11-slim-bookworm

# Install Java (required for PySpark)
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    wget \
    bzip2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and helper scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Ensure pipeline script is executable
RUN chmod +x /app/scripts/run_spark_pipeline.sh

# Create output directories
RUN mkdir -p /app/data/index /app/data/raw_html /app/data/scraped /app/logs

# Configure PySpark executables
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python

# Default command
CMD ["./scripts/run_spark_pipeline.sh"]
