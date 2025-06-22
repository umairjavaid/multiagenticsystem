"""
Logging utilities for the multiagenticsystem package.
"""

import logging
import sys
from typing import Optional


def setup_logger(verbose: bool = False) -> None:
    """Setup the main logger for the package."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)
