"""
Tracking module for RTMLib pose tracking.

Provides algorithms and utilities for multi-person tracking:
- Hungarian algorithm for optimal assignment
- Velocity prediction and position smoothing
- Teleportation detection (reject unrealistic jumps)
- Track re-identification (recover lost tracks)
"""

from .algorithms import (
    compute_iou,
    compute_iou_matrix,
    compute_distance_matrix,
    hungarian_matching,
    distance_matching,
    SCIPY_AVAILABLE
)
from .velocity import VelocityTracker
from .teleportation import TeleportationDetector
from .reid import TrackReidentifier, TerminatedTrack
from .static_filter import StaticTrackFilter

__all__ = [
    # Algorithms
    'compute_iou',
    'compute_iou_matrix',
    'compute_distance_matrix',
    'hungarian_matching',
    'distance_matching',
    'SCIPY_AVAILABLE',
    # Velocity
    'VelocityTracker',
    # Teleportation
    'TeleportationDetector',
    # Re-identification
    'TrackReidentifier',
    'TerminatedTrack',
    # Static track filtering
    'StaticTrackFilter',
]
