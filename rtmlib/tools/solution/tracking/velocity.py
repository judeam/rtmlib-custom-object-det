"""
Velocity tracking and position smoothing for robust pose tracking.

Provides:
- EMA-based velocity estimation
- Position prediction during occlusions
- Position smoothing to reduce jitter
"""

from typing import Dict, Tuple, Optional
import numpy as np


class VelocityTracker:
    """Tracks per-object velocities and provides position prediction/smoothing.

    Uses exponential moving average (EMA) for both velocity estimation
    and position smoothing to provide stable, predictable tracking.

    Attributes:
        velocity_alpha: Weight for new velocity (higher = more responsive)
        smoothing_alpha: Weight for new position (higher = less smoothing)
        velocities: Dict mapping track_id to (dx, dy) velocity
        last_bboxes: Dict mapping track_id to last known bbox
    """

    def __init__(
        self,
        velocity_alpha: float = 0.8,
        smoothing_alpha: float = 0.85
    ):
        """Initialize velocity tracker.

        Args:
            velocity_alpha: EMA weight for velocity updates (0.0-1.0).
                Higher values make velocity more responsive to changes.
                Default 0.8 means 80% new velocity, 20% old velocity.
            smoothing_alpha: EMA weight for position smoothing (0.0-1.0).
                Higher values mean less smoothing (more responsive).
                Default 0.85 means 85% new position, 15% smoothed.
        """
        self.velocity_alpha = velocity_alpha
        self.smoothing_alpha = smoothing_alpha
        self.velocities: Dict[int, Tuple[float, float]] = {}
        self.last_bboxes: Dict[int, np.ndarray] = {}

    def reset(self):
        """Reset all tracked velocities and positions."""
        self.velocities.clear()
        self.last_bboxes.clear()

    def update(
        self,
        track_id: int,
        bbox_prev: np.ndarray,
        bbox_curr: np.ndarray,
        frame_delta: int = 1
    ) -> Tuple[float, float]:
        """Update velocity estimate for a track.

        Args:
            track_id: Track identifier
            bbox_prev: Previous bbox [x1, y1, x2, y2]
            bbox_curr: Current bbox [x1, y1, x2, y2]
            frame_delta: Number of frames between prev and curr

        Returns:
            Smoothed velocity as (dx, dy) in pixels per frame
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

        # Raw velocity (normalized by frame delta)
        if frame_delta > 0:
            raw_velocity = (center_curr - center_prev) / frame_delta
        else:
            raw_velocity = center_curr - center_prev

        # EMA smoothing
        if track_id in self.velocities:
            old_velocity = np.array(self.velocities[track_id])
            smoothed = (
                self.velocity_alpha * raw_velocity +
                (1 - self.velocity_alpha) * old_velocity
            )
        else:
            smoothed = raw_velocity

        self.velocities[track_id] = (float(smoothed[0]), float(smoothed[1]))
        self.last_bboxes[track_id] = bbox_curr.copy()

        return self.velocities[track_id]

    def predict(
        self,
        track_id: int,
        bbox: np.ndarray,
        frame_delta: int = 1
    ) -> np.ndarray:
        """Predict bbox position using tracked velocity.

        Args:
            track_id: Track identifier
            bbox: Current/last known bbox [x1, y1, x2, y2]
            frame_delta: Number of frames to predict forward

        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        if track_id not in self.velocities:
            return bbox.copy()

        dx, dy = self.velocities[track_id]
        predicted = bbox.copy()

        # Shift all coordinates by velocity
        predicted[0] += dx * frame_delta  # x1
        predicted[1] += dy * frame_delta  # y1
        predicted[2] += dx * frame_delta  # x2
        predicted[3] += dy * frame_delta  # y2

        return predicted

    def smooth(
        self,
        track_id: int,
        bbox_curr: np.ndarray,
        bbox_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply position smoothing to reduce jitter.

        Args:
            track_id: Track identifier
            bbox_curr: Current detected bbox
            bbox_prev: Previous bbox (uses stored if None)

        Returns:
            Smoothed bbox [x1, y1, x2, y2]
        """
        if bbox_prev is None:
            bbox_prev = self.last_bboxes.get(track_id)

        if bbox_prev is None:
            # No previous bbox, return current as-is
            return bbox_curr.copy()

        # EMA smoothing on bbox coordinates
        smoothed = (
            self.smoothing_alpha * bbox_curr +
            (1 - self.smoothing_alpha) * bbox_prev
        )

        return smoothed

    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Get current velocity for a track.

        Args:
            track_id: Track identifier

        Returns:
            Velocity as (dx, dy) or None if not tracked
        """
        return self.velocities.get(track_id)

    def get_last_bbox(self, track_id: int) -> Optional[np.ndarray]:
        """Get last known bbox for a track.

        Args:
            track_id: Track identifier

        Returns:
            Last bbox or None if not tracked
        """
        return self.last_bboxes.get(track_id)

    def remove_track(self, track_id: int):
        """Remove a track from velocity tracking.

        Args:
            track_id: Track identifier to remove
        """
        self.velocities.pop(track_id, None)
        self.last_bboxes.pop(track_id, None)

    def get_active_tracks(self) -> set:
        """Get set of currently tracked track IDs.

        Returns:
            Set of track IDs with velocity data
        """
        return set(self.velocities.keys())
