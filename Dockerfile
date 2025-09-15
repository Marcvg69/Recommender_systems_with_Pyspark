FROM python:3.10-jammy

# Install Java (for Spark) + minimal tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Let PySpark find java on PATH (JAVA_HOME not strictly required)
ENV PYSPARK_PYTHON=python

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 8501
ENTRYPOINT ["tini", "--"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
