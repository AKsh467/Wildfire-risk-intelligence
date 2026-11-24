"""SparkSession Factory and Configuration Manager for Local PySpark Execution.

Configures an optimized local SparkSession with Arrow serialization, balanced
shuffle partitions for local multi-threading, and memory limits.
"""

from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("spark_session")

_SPARK_INSTANCE = None


def is_spark_available() -> bool:
    """Check if pyspark is installed and importable."""
    try:
        import pyspark
        return True
    except ImportError:
        return False


def get_spark_session(
    app_name: str = "WildfireRiskIntelligence",
    master: str = "local[*]",
    driver_memory: str = "4g",
    shuffle_partitions: int = 8,
):
    """Create or retrieve a singleton SparkSession configured for local execution.

    Args:
        app_name: Name of the Spark application.
        master: Spark master URL (default 'local[*]' for all local cores).
        driver_memory: Maximum heap size for the driver process.
        shuffle_partitions: Default number of partitions to use when shuffling data.

    Returns:
        pyspark.sql.SparkSession instance.
    """
    global _SPARK_INSTANCE

    if not is_spark_available():
        raise ImportError(
            "PySpark is not installed in the active Python environment. "
            "Please run: pip install pyspark>=3.4.0"
        )

    from pyspark.sql import SparkSession

    if _SPARK_INSTANCE is not None:
        return _SPARK_INSTANCE

    logger.info(
        f"Initializing PySpark session '{app_name}' (master={master}, memory={driver_memory}, "
        f"shuffle_partitions={shuffle_partitions})"
    )

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
    )

    _SPARK_INSTANCE = builder.getOrCreate()
    # Set Spark log level to WARN to reduce terminal noise
    _SPARK_INSTANCE.sparkContext.setLogLevel("WARN")

    logger.info(f"PySpark session active. Spark Version: {_SPARK_INSTANCE.version}")
    return _SPARK_INSTANCE


def stop_spark_session():
    """Stop the active SparkSession and clear singleton reference."""
    global _SPARK_INSTANCE
    if _SPARK_INSTANCE is not None:
        logger.info("Stopping PySpark session...")
        _SPARK_INSTANCE.stop()
        _SPARK_INSTANCE = None
