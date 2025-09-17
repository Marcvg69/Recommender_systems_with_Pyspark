# src/utils/spark.py
from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "recsys", memory: str = "2g") -> SparkSession:
    # Ensure JAVA_HOME is set in Docker; locally, user should have Java installed.
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )
    return spark
