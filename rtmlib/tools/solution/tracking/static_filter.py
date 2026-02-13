"""
Static track detection and suppression for pose tracking.

Filters out detections of static/background people (e.g. billboard ads,
posters, standees) that the detector picks up as real people.  Uses three
layers of motion analysis:

1. **Track confirmation** – new tracks start as *tentative* and must
   demonstrate camera-independent motion within a confirmation window
   to become *confirmed*.  Only confirmed tracks are emitted.

2. **Velocity-variance scoring** – even after confirmation, tracks are
   continuously monitored.  If a track's velocity variance drops below
   a threshold for an extended period it is demoted and suppressed.

3. **Background-motion consensus (cheap CMC)** – each frame the median
   displacement of all active tracks is computed and subtracted.  The
   residual displacement is the *independent motion* of each track.
   Billboard people move in lockstep with the background and have
   near-zero residual.
"""

from collections import deque
from typing import Dict, Set, Tuple

import numpy as np


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Return center (cx, cy) of an [x1, y1, x2, y2] bbox."""
    return np.array([
        (bbox[0] + bbox[2]) * 0.5,
        (bbox[1] + bbox[3]) * 0.5,
    ])


class StaticTrackFilter:
    """Classifies tracks as *confirmed* (moving) or *static* (background).

    Parameters
    ----------
    confirmation_frames : int
        Number of frames a tentative track is observed before judgement.
    min_residual_displacement : float
        Minimum cumulative camera-independent displacement (px) a track
        must accumulate during the confirmation window to be confirmed.
    variance_window : int
        Sliding-window length (frames) for ongoing velocity-variance check.
    min_velocity_variance : float
        Minimum velocity variance (px²/frame²) to stay confirmed.
    enable_background_consensus : bool
        Whether to subtract the estimated background (camera) motion
        from each track's displacement before evaluating.
    """

    # Track statuses
    TENTATIVE = 0
    CONFIRMED = 1
    STATIC = 2

    def __init__(
        self,
        confirmation_frames: int = 30,
        min_residual_displacement: float = 50.0,
        variance_window: int = 60,
        min_velocity_variance: float = 0.5,
        enable_background_consensus: bool = True,
    ):
        self.confirmation_frames = confirmation_frames
        self.min_residual_displacement = min_residual_displacement
        self.variance_window = variance_window
        self.min_velocity_variance = min_velocity_variance
        self.enable_background_consensus = enable_background_consensus

        # Per-track state ------------------------------------------------
        # status: TENTATIVE / CONFIRMED / STATIC
        self.track_status: Dict[int, int] = {}
        # how many frames the track has been observed
        self.track_hits: Dict[int, int] = {}
        # first observed bbox center (for displacement calc)
        self.track_origin: Dict[int, np.ndarray] = {}
        # cumulative *residual* displacement (camera-compensated)
        self.track_cum_residual: Dict[int, float] = {}
        # per-frame residual velocity history for variance scoring
        self.track_vel_history: Dict[int, deque] = {}
        # previous bbox center (for frame-to-frame delta)
        self.track_prev_center: Dict[int, np.ndarray] = {}

    def reset(self):
        """Clear all state."""
        self.track_status.clear()
        self.track_hits.clear()
        self.track_origin.clear()
        self.track_cum_residual.clear()
        self.track_vel_history.clear()
        self.track_prev_center.clear()

    # ------------------------------------------------------------------
    # Layer 3 – background motion consensus
    # ------------------------------------------------------------------
    def _estimate_background_motion(
        self, displacements: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Estimate camera motion as median displacement of all tracks.

        Parameters
        ----------
        displacements : dict
            {track_id: (dx, dy)} raw displacements this frame.

        Returns
        -------
        np.ndarray
            Estimated camera motion (dx, dy).
        """
        if not displacements:
            return np.zeros(2)
        vecs = np.array(list(displacements.values()))
        return np.median(vecs, axis=0)

    # ------------------------------------------------------------------
    # Main per-frame update
    # ------------------------------------------------------------------
    def update(
        self,
        track_bboxes: Dict[int, np.ndarray],
    ) -> Set[int]:
        """Update filter state and return set of track IDs to suppress.

        Call this once per frame *after* the tracker has updated
        ``active_tracks``.  Provide the current bbox for every active
        track.

        Parameters
        ----------
        track_bboxes : dict
            {track_id: bbox} for all active tracks this frame.

        Returns
        -------
        set of int
            Track IDs that should be suppressed (tentative or static).
        """
        suppress: Set[int] = set()

        # --- compute raw per-track displacement this frame ---------------
        raw_displacements: Dict[int, np.ndarray] = {}

        for tid, bbox in track_bboxes.items():
            center = _bbox_center(bbox)

            if tid not in self.track_status:
                # Brand-new track
                self.track_status[tid] = self.TENTATIVE
                self.track_hits[tid] = 0
                self.track_origin[tid] = center.copy()
                self.track_cum_residual[tid] = 0.0
                self.track_vel_history[tid] = deque(
                    maxlen=self.variance_window
                )
                self.track_prev_center[tid] = center.copy()

            prev = self.track_prev_center[tid]
            displacement = center - prev
            raw_displacements[tid] = displacement
            self.track_prev_center[tid] = center.copy()

        # --- Layer 3: estimate background (camera) motion ----------------
        if self.enable_background_consensus and len(raw_displacements) > 1:
            bg_motion = self._estimate_background_motion(raw_displacements)
        else:
            bg_motion = np.zeros(2)

        # --- per-track scoring -------------------------------------------
        for tid, bbox in track_bboxes.items():
            self.track_hits[tid] = self.track_hits.get(tid, 0) + 1
            raw_d = raw_displacements.get(tid, np.zeros(2))

            # Residual displacement (camera-compensated)
            residual = raw_d - bg_motion
            residual_mag = float(np.linalg.norm(residual))
            self.track_cum_residual[tid] = (
                self.track_cum_residual.get(tid, 0.0) + residual_mag
            )
            self.track_vel_history[tid].append(residual)

            status = self.track_status[tid]

            # --- Layer 1: confirmation gate ------------------------------
            if status == self.TENTATIVE:
                if self.track_hits[tid] >= self.confirmation_frames:
                    cum = self.track_cum_residual[tid]
                    if cum >= self.min_residual_displacement:
                        self.track_status[tid] = self.CONFIRMED
                    else:
                        self.track_status[tid] = self.STATIC
                # Always suppress tentative tracks
                suppress.add(tid)

            # --- Layer 2: ongoing variance check -------------------------
            elif status == self.CONFIRMED:
                hist = self.track_vel_history[tid]
                if len(hist) >= self.variance_window:
                    arr = np.array(hist)  # (W, 2)
                    var_sum = float(arr.var(axis=0).sum())
                    if var_sum < self.min_velocity_variance:
                        # Track has become static – demote
                        self.track_status[tid] = self.STATIC
                        suppress.add(tid)

            elif status == self.STATIC:
                suppress.add(tid)

        return suppress

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def remove_track(self, track_id: int):
        """Remove all state for a track that has been deleted."""
        self.track_status.pop(track_id, None)
        self.track_hits.pop(track_id, None)
        self.track_origin.pop(track_id, None)
        self.track_cum_residual.pop(track_id, None)
        self.track_vel_history.pop(track_id, None)
        self.track_prev_center.pop(track_id, None)

    def is_static(self, track_id: int) -> bool:
        """Return True if track is classified as static."""
        return self.track_status.get(track_id) == self.STATIC

    def is_confirmed(self, track_id: int) -> bool:
        """Return True if track is confirmed (moving)."""
        return self.track_status.get(track_id) == self.CONFIRMED

    def get_status(self, track_id: int) -> int:
        """Return track status (TENTATIVE / CONFIRMED / STATIC)."""
        return self.track_status.get(track_id, self.TENTATIVE)
