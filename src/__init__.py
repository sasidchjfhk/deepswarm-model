"""
Swarm IDS - Production ML Pipeline for Network Intrusion Detection

A complete end-to-end machine learning pipeline built with industry best practices.
"""

__version__ = "1.0.0"
__author__ = "Swarm IDS Team"
__license__ = "MIT"

from loguru import logger

# Configure default logger
logger.add(
    "logs/swarm_ids_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)
