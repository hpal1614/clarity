"""
Sidekicks Module - All AI Agent Implementations
"""

# Import the main working Sidekick
from .fixxy_sidekick import FixxySidekick

# Import placeholders for other Sidekicks
from .placeholder_sidekicks import (
    FindySidekick, PredictySidekick, PlannySidekick,
    SyncieSidekick, WatchySidekick, TrackieSidekick,
    HelpieSidekick, CoachySidekick, GreetieSidekick
)

__all__ = [
    'FixxySidekick',
    'FindySidekick', 'PredictySidekick', 'PlannySidekick',
    'SyncieSidekick', 'WatchySidekick', 'TrackieSidekick', 
    'HelpieSidekick', 'CoachySidekick', 'GreetieSidekick'
]