"""
Common components shared across models.

This module provides reusable building blocks:
- TwoStreamPreprocessing: M/P channel preprocessing
"""

from .preprocessing import TwoStreamPreprocessing

__all__ = [
    "TwoStreamPreprocessing",
]
