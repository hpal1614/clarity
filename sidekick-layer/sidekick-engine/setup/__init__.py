"""
Setup and Configuration Package
"""

from .cosmos_db_setup import (
    setup_cosmos_database,
    create_containers,
    verify_setup
)

__all__ = [
    "setup_cosmos_database",
    "create_containers", 
    "verify_setup"
]