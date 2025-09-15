from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "recsys", memory: str = "2g") -> SparkSession:
    # Ensure JAVA_HOME is set in Docker; locally, user should have Java installed.
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark