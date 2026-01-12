"""
Code execution utilities for reward workers.

Provides safe(r) code execution with timeout and memory limits.
"""
from .local_exec import code_exec_local

__all__ = ['code_exec_local']
