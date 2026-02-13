"""
Track re-identification for recovering lost tracks.

Stores recently terminated tracks and attempts to recover them when
unmatched detections appear in predicted locations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TerminatedTrack:
    """Information about a terminated track for potential recovery.

    Attributes:
        track_id: Original track ID to recover
        last_bbox: Last known bounding box [x1, y1, x2, y2]
        last_velocity: Velocity at termination (dx, dy)
        termination_frame: Frame when track was terminated
        last_keypoints: Optional keypoints for pose-based matching
    """
    track_id: int
    last_bbox: np.ndarray
    last_velocity: Tuple[float, float]
    termination_frame: int
    last_keypoints: Optional[np.ndarray] = None


class TrackReidentifier:
    """Recovers lost tracks using velocity extrapolation and spatial matching.

    When tracks exceed max_age without detection, they're stored here for
    potential recovery. When new unmatched detections appear, this class
    attempts to match them with terminated tracks based on:
    - Predicted position (using velocity extrapolation)
    - Spatial proximity

    Attributes:
        max_recovery_frames: Max frames a terminated track can be recovered
        max_tracks: Maximum number of terminated tracks to store
        spatial_threshold_factor: Distance threshold as multiple of bbox size
    """

    def __init__(
        self,
        max_recovery_frames: int = 60,
        max_tracks: int = 10,
        spatial_threshold_factor: float = 2.0
    ):
        """Initialize track re-identifier.

        Args:
            max_recovery_frames: Max frames to keep terminated track.
                Default 60 = 2 seconds at 30fps.
            max_tracks: Maximum terminated tracks to store.
                Older tracks are pruned when limit is exceeded.
            spatial_threshold_factor: Distance threshold as multiple of bbox size.
                Default 2.0 means detection must be within 2× bbox size of
                predicted position for recovery.
        """
        self.max_recovery_frames = max_recovery_frames
        self.max_tracks = max_tracks
        self.spatial_threshold_factor = spatial_threshold_factor
        self.terminated_tracks: Dict[int, TerminatedTrack] = {}

    def reset(self):
        """Clear all stored terminated tracks."""
        self.terminated_tracks.clear()

    def store(
        self,
        track_id: int,
        last_bbox: np.ndarray,
        last_velocity: Tuple[float, float],
        frame_id: int,
        last_keypoints: Optional[np.ndarray] = None
    ):
        """Store a terminated track for potential recovery.

        Args:
            track_id: Track ID being terminated
            last_bbox: Last known bbox
            last_velocity: Last known velocity (dx, dy)
            frame_id: Current frame number
            last_keypoints: Optional keypoints for pose matching
        """
        # Prune old tracks first
        self._prune_old_tracks(frame_id)

        # Enforce max_tracks limit
        if len(self.terminated_tracks) >= self.max_tracks:
            # Remove oldest track
            oldest_id = min(
                self.terminated_tracks.keys(),
                key=lambda tid: self.terminated_tracks[tid].termination_frame
            )
            del self.terminated_tracks[oldest_id]

        # Store the terminated track
        self.terminated_tracks[track_id] = TerminatedTrack(
            track_id=track_id,
            last_bbox=last_bbox.copy(),
            last_velocity=last_velocity,
            termination_frame=frame_id,
            last_keypoints=last_keypoints.copy() if last_keypoints is not None else None
        )

    def try_recover(
        self,
        bbox: np.ndarray,
        frame_id: int
    ) -> int:
        """Attempt to recover a terminated track for an unmatched detection.

        Args:
            bbox: Unmatched detection bbox
            frame_id: Current frame number

        Returns:
            Recovered track_id if match found, -1 otherwise
        """
        if not self.terminated_tracks:
            return -1

        # Prune old tracks
        self._prune_old_tracks(frame_id)

        # Calculate detection center and size
        center_curr = np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        best_match = -1
        best_distance = float('inf')
        max_distance = self.spatial_threshold_factor * bbox_size

        for track_id, term in list(self.terminated_tracks.items()):
            # Calculate frame delta
            frame_delta = frame_id - term.termination_frame

            # Skip if too old
            if frame_delta > self.max_recovery_frames:
                continue

            # Predict where track should be based on velocity
            dx, dy = term.last_velocity
            predicted_center = np.array([
                (term.last_bbox[0] + term.last_bbox[2]) / 2 + dx * frame_delta,
                (term.last_bbox[1] + term.last_bbox[3]) / 2 + dy * frame_delta
            ])

            # Calculate distance to predicted position
            distance = np.linalg.norm(center_curr - predicted_center)

            # Check if this is the best match within threshold
            if distance < max_distance and distance < best_distance:
                best_distance = distance
                best_match = track_id

        # If match found, remove from terminated tracks
        if best_match >= 0:
            del self.terminated_tracks[best_match]

        return best_match

    def try_recover_batch(
        self,
        unmatched_bboxes: List[np.ndarray],
        unmatched_indices: List[int],
        frame_id: int
    ) -> Dict[int, int]:
        """Attempt to recover tracks for multiple unmatched detections.

        Args:
            unmatched_bboxes: List of unmatched detection bboxes
            unmatched_indices: Original indices of unmatched detections
            frame_id: Current frame number

        Returns:
            Dict mapping detection index to recovered track_id
        """
        recoveries = {}

        for bbox, idx in zip(unmatched_bboxes, unmatched_indices):
            track_id = self.try_recover(bbox, frame_id)
            if track_id >= 0:
                recoveries[idx] = track_id

        return recoveries

    def _prune_old_tracks(self, current_frame: int):
        """Remove terminated tracks that are too old to recover.

        Args:
            current_frame: Current frame number
        """
        to_remove = [
            track_id for track_id, term in self.terminated_tracks.items()
            if current_frame - term.termination_frame > self.max_recovery_frames
        ]

        for track_id in to_remove:
            del self.terminated_tracks[track_id]

    def get_stored_count(self) -> int:
        """Get number of currently stored terminated tracks.

        Returns:
            Count of stored tracks
        """
        return len(self.terminated_tracks)

    def get_stored_ids(self) -> List[int]:
        """Get list of stored track IDs.

        Returns:
            List of track IDs that can potentially be recovered
        """
        return list(self.terminated_tracks.keys())
