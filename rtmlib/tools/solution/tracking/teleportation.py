"""
Size-aware teleportation detection for robust tracking.

Rejects detections that would cause unrealistic position jumps based on
object size. A movement of > k × bbox_size is considered teleportation.
"""

from typing import Tuple
import numpy as np


class TeleportationDetector:
    """Detects and rejects unrealistic position jumps (teleportation).

    Uses object bbox size as the scale for determining what constitutes
    a realistic vs unrealistic movement. Movements > threshold × bbox_size
    are rejected as likely misdetections or track switches.

    Attributes:
        max_displacement_factor: Max movement as multiple of bbox size
        min_bbox_size: Minimum bbox size to prevent division by tiny values
    """

    def __init__(
        self,
        max_displacement_factor: float = 3.0,
        min_bbox_size: float = 10.0
    ):
        """Initialize teleportation detector.

        Args:
            max_displacement_factor: Max displacement as multiple of bbox size.
                Default 3.0 means object can move up to 3× its size per frame.
            min_bbox_size: Minimum bbox size for scale calculation (pixels).
                Prevents division issues with very small bboxes.
        """
        self.max_displacement_factor = max_displacement_factor
        self.min_bbox_size = min_bbox_size

    def is_teleportation(
        self,
        bbox_prev: np.ndarray,
        bbox_curr: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Check if bbox transition would cause teleportation.

        Args:
            bbox_prev: Previous bbox as [x1, y1, x2, y2]
            bbox_curr: Current bbox as [x1, y1, x2, y2]

        Returns:
            Tuple of (is_teleportation, displacement, relative_displacement)
                - is_teleportation: True if movement is unrealistic
                - displacement: Absolute displacement in pixels
                - relative_displacement: Displacement as multiple of bbox size
        """
        # Calculate centers
        center_prev = np.array([
            (bbox_prev[0] + bbox_prev[2]) / 2,
            (bbox_prev[1] + bbox_prev[3]) / 2
        ])
        center_curr = np.array([
            (bbox_curr[0] + bbox_curr[2]) / 2,
            (bbox_curr[1] + bbox_curr[3]) / 2
        ])

        # Calculate displacement
        displacement = np.linalg.norm(center_curr - center_prev)

        # Calculate scale (average bbox diagonal or larger dimension)
        width_prev = bbox_prev[2] - bbox_prev[0]
        height_prev = bbox_prev[3] - bbox_prev[1]
        width_curr = bbox_curr[2] - bbox_curr[0]
        height_curr = bbox_curr[3] - bbox_curr[1]

        # Use average of larger dimensions as scale
        avg_size = (
            max(width_prev, height_prev) +
            max(width_curr, height_curr)
        ) / 2
        avg_size = max(avg_size, self.min_bbox_size)

        # Calculate relative displacement
        relative_displacement = displacement / avg_size

        # Check threshold
        is_teleport = relative_displacement > self.max_displacement_factor

        return is_teleport, float(displacement), float(relative_displacement)

    def validate_match(
        self,
        bbox_prev: np.ndarray,
        bbox_curr: np.ndarray
    ) -> bool:
        """Validate if a bbox match is realistic (not teleportation).

        Args:
            bbox_prev: Previous bbox
            bbox_curr: Current bbox

        Returns:
            True if match is valid (not teleportation)
        """
        is_teleport, _, _ = self.is_teleportation(bbox_prev, bbox_curr)
        return not is_teleport

    def filter_matches(
        self,
        matches: list,
        bboxes_prev: list,
        bboxes_curr: list
    ) -> Tuple[list, list, list]:
        """Filter matches to remove teleporting ones.

        Args:
            matches: List of (prev_idx, curr_idx) matches
            bboxes_prev: List of previous bboxes
            bboxes_curr: List of current bboxes

        Returns:
            Tuple of (valid_matches, rejected_prev_indices, rejected_curr_indices)
        """
        valid_matches = []
        rejected_prev = []
        rejected_curr = []

        for prev_idx, curr_idx in matches:
            bbox_prev = bboxes_prev[prev_idx]
            bbox_curr = bboxes_curr[curr_idx]

            if self.validate_match(bbox_prev, bbox_curr):
                valid_matches.append((prev_idx, curr_idx))
            else:
                rejected_prev.append(prev_idx)
                rejected_curr.append(curr_idx)

        return valid_matches, rejected_prev, rejected_curr
