"""Centralized safety thresholds for bounding resource usage.

This module defines common constants used across the codebase to avoid
duplicated magic numbers and to make tuning easier.
"""
from __future__ import annotations

# Maximum bytes to read from a single uploaded or local file (10 MB)
MAX_READ_BYTES: int = 10 * 1024 * 1024

# Buffer size threshold used by streaming scripts before flushing to disk (50 MB)
BUF_LIMIT: int = 50 * 1024 * 1024

# Default subprocess timeout for short tasks (seconds)
SUBPROCESS_TIMEOUT: int = 120

# Timeout for longer benchmark runs (seconds)
BENCHMARK_TIMEOUT: int = 300

# Max allowed vertices for OBJ loaders to prevent huge uploads
MAX_OBJ_VERTICES: int = 500_000

# Max size for TenSEAL homomorphic context blobs (50 MB)
MAX_CONTEXT_BYTES: int = 50 * 1024 * 1024

__all__ = [
    "MAX_READ_BYTES",
    "BUF_LIMIT",
    "SUBPROCESS_TIMEOUT",
    "BENCHMARK_TIMEOUT",
    "MAX_OBJ_VERTICES",
    "MAX_CONTEXT_BYTES",
]
