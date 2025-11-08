FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    SCALA_VERSION=2.13.12 \
    SCALA_HOME=/usr/local/scala \
    PYSPARK_PYTHON=/usr/local/bin/python \
    PYSPARK_DRIVER_PYTHON=/usr/local/bin/python

ENV PATH="${JAVA_HOME}/bin:${SCALA_HOME}/bin:${PATH}"

# Install Scala 2.13
RUN wget -q https://downloads.lightbend.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz \
    && tar -xzf scala-${SCALA_VERSION}.tgz -C /usr/local \
    && ln -s /usr/local/scala-${SCALA_VERSION} ${SCALA_HOME} \
    && rm scala-${SCALA_VERSION}.tgz

# Add Scala libraries to Spark classpath
ENV SPARK_DIST_CLASSPATH="${SCALA_HOME}/lib/*" \
    PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraClassPath=${SCALA_HOME}/lib/* --conf spark.executor.extraClassPath=${SCALA_HOME}/lib/* pyspark-shell"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN chmod +x ./scripts/run_spark_pipeline.sh

# Create data directories
RUN mkdir -p data/index data/raw_html data/scraped logs

CMD ["./scripts/run_spark_pipeline.sh"]
