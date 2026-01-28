"""
This package implements the JEDI (Jailbreak dEfense via Detection and Intervention)
defense mechanism.

This __init__.py file allows the 'Guard' class to be imported directly from the
package top level for convenience.
"""

from .guard import Guard

__all__ = ["Guard"]
