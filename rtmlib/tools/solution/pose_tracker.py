'''
Example:

import cv2
from functools import partial
from rtmlib import PoseTracker, Wholebody, Custom, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

cap = cv2.VideoCapture('./demo.mp4')

pose_tracker = PoseTracker(Wholebody,
                        det_frequency=10,  # detect every 10 frames
                        to_openpose=openpose_skeleton,
                        backend=backend, device=device)


# # Initialized slightly differently for Custom solution:
# custom = partial(Custom,
#                 to_openpose=openpose_skeleton,
#                 pose_class='RTMO',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-384x384-39e78cc4_20231211.zip', # noqa
#                 pose_input_size=(384,384),
#                 backend=backend,
#                 device=device)
# # or
# custom = partial(
#             Custom,
#             to_openpose=openpose_skeleton,
#             det_class='RFDETRNano',
#             det=None,  # Uses models/rfdetr_nano_person.pt
#             det_input_size=(384, 384),
#             pose_class='RTMPose',
#             pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', # noqa
#             pose_input_size=(192, 256),
#             backend=backend,
#             device=device)
# # then
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)


frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = pose_tracker(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
'''
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tracking import (
    compute_iou,
    compute_iou_matrix,
    hungarian_matching,
    distance_matching,
    VelocityTracker,
    TeleportationDetector,
    TrackReidentifier,
    SCIPY_AVAILABLE
)


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints: Keypoints of person.
        expansion: Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


class PoseTracker:
    """Pose tracker for multi-person pose estimation with enhanced tracking.

    Features:
    - Hungarian algorithm for optimal bbox assignment (reduces ID switches)
    - Velocity prediction for better occlusion handling
    - Position smoothing to reduce trajectory jitter
    - Teleportation detection to reject unrealistic jumps
    - Track re-identification to recover lost tracks

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, Custom, etc.
        det_frequency (int): Frequency of object detection.
        tracking (bool): Whether to enable tracking.
        tracking_thr (float): IoU threshold for track matching.
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
        use_hungarian (bool): Use Hungarian algorithm for optimal matching.
        velocity_prediction (bool): Enable velocity-based prediction.
        velocity_alpha (float): EMA weight for velocity smoothing (0.0-1.0).
        position_smoothing (bool): Enable position smoothing to reduce jitter.
        smoothing_alpha (float): EMA weight for position smoothing (0.0-1.0).
        teleport_detection (bool): Enable teleportation detection.
        max_teleport_factor (float): Max movement as multiple of bbox size.
        enable_reid (bool): Enable track re-identification.
        reid_max_frames (int): Max frames to recover a lost track.
        reid_max_tracks (int): Max terminated tracks to store for recovery.
        max_age (int): Max frames to keep track alive without detection.
    """
    MIN_AREA = 1000

    def __init__(
        self,
        solution: type,
        det_frequency: int = 1,
        tracking: bool = True,
        tracking_thr: float = 0.3,
        mode: str = 'balanced',
        to_openpose: bool = False,
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        # Enhanced tracking options (all backward compatible with defaults)
        use_hungarian: bool = True,
        velocity_prediction: bool = True,
        velocity_alpha: float = 0.8,
        position_smoothing: bool = True,
        smoothing_alpha: float = 0.85,
        teleport_detection: bool = True,
        max_teleport_factor: float = 3.0,
        enable_reid: bool = False,
        reid_max_frames: int = 60,
        reid_max_tracks: int = 10,
        max_age: int = 30,
        # Adaptive pose skip for performance optimization
        adaptive_pose: bool = False,
        motion_threshold: float = 5.0,
        pose_reuse_frames: int = 3
    ):
        model = solution(mode=mode,
                         to_openpose=to_openpose,
                         backend=backend,
                         device=device)

        try:
            self.det_model = model.det_model
        except:  # rtmo
            self.det_model = None
        self.pose_model = model.pose_model

        self.det_frequency = det_frequency
        self.tracking = tracking
        self.tracking_thr = tracking_thr
        self.max_age = max_age

        # Enhanced tracking options
        self.use_hungarian = use_hungarian and SCIPY_AVAILABLE
        self.velocity_prediction = velocity_prediction
        self.position_smoothing = position_smoothing
        self.teleport_detection = teleport_detection
        self.enable_reid = enable_reid

        # Adaptive pose skip options
        self.adaptive_pose = adaptive_pose
        self.motion_threshold = motion_threshold
        self.pose_reuse_frames = pose_reuse_frames

        # Initialize tracking components
        self.velocity_tracker = VelocityTracker(
            velocity_alpha=velocity_alpha,
            smoothing_alpha=smoothing_alpha
        ) if (velocity_prediction or position_smoothing) else None

        self.teleport_detector = TeleportationDetector(
            max_displacement_factor=max_teleport_factor
        ) if teleport_detection else None

        self.reid = TrackReidentifier(
            max_recovery_frames=reid_max_frames,
            max_tracks=reid_max_tracks
        ) if enable_reid else None

        self.reset()

        if self.tracking:
            features = []
            if self.use_hungarian:
                features.append("Hungarian matching")
            if velocity_prediction:
                features.append("velocity prediction")
            if position_smoothing:
                features.append("position smoothing")
            if teleport_detection:
                features.append("teleportation detection")
            if enable_reid:
                features.append("track re-identification")
            if adaptive_pose:
                features.append("adaptive pose skip")

            if features:
                print(f'Enhanced tracking enabled: {", ".join(features)}')
            else:
                print('Basic tracking enabled (greedy IoU matching)')

    def reset(self):
        """Reset pose tracker state."""
        self.frame_cnt = 0
        self.next_id = 0

        # Track state
        self.active_tracks: Dict[int, np.ndarray] = {}  # track_id -> bbox
        self.track_ages: Dict[int, int] = {}  # track_id -> frames since last seen
        self.track_keypoints: Dict[int, np.ndarray] = {}  # track_id -> last keypoints
        self.track_scores: Dict[int, np.ndarray] = {}  # track_id -> last scores

        # Adaptive pose skip state
        self.pose_skip_count: Dict[int, int] = {}  # track_id -> frames since last pose

        # Legacy compatibility
        self.bboxes_last_frame: List[np.ndarray] = []
        self.track_ids_last_frame: List[int] = []

        # Reset components
        if self.velocity_tracker:
            self.velocity_tracker.reset()
        if self.reid:
            self.reid.reset()

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a frame and return tracked keypoints.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Tuple of (keypoints, scores) where:
                keypoints: Shape (num_persons, num_joints, 2)
                scores: Shape (num_persons, num_joints)
        """
        # Run detection
        if self.det_model is not None:
            if self.frame_cnt % self.det_frequency == 0:
                bboxes = self.det_model(image)
            else:
                bboxes = self.bboxes_last_frame

            # Adaptive pose skip: only run pose on tracks that need it
            if self.adaptive_pose and self.tracking and len(bboxes) > 0 and self.active_tracks:
                keypoints, scores = self._adaptive_pose_inference(image, bboxes)
            else:
                keypoints, scores = self.pose_model(image, bboxes=bboxes)
        else:  # rtmo
            keypoints, scores = self.pose_model(image)

        if not self.tracking:
            # Without tracking - just update bboxes for next frame
            bboxes_current_frame = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)
                bboxes_current_frame.append(bbox)
            self.bboxes_last_frame = bboxes_current_frame
            self.frame_cnt += 1
            return keypoints, scores

        # With tracking
        if len(keypoints) == 0:
            self._age_tracks()
            self.frame_cnt += 1
            return np.array([]), np.array([])

        # Convert keypoints to bboxes
        current_bboxes = [pose_to_bbox(kpts) for kpts in keypoints]

        # Match current detections to existing tracks
        track_assignments = self._match_detections(current_bboxes)

        # Build output ordered by track ID
        tracked_keypoints = []
        tracked_scores = []
        new_track_ids = []
        new_bboxes = []

        for det_idx, (kpts, score, bbox) in enumerate(
            zip(keypoints, scores, current_bboxes)
        ):
            track_id = track_assignments.get(det_idx, -1)

            if track_id == -1:
                # Unmatched detection - try ReID or create new track
                if self.enable_reid and self.reid:
                    track_id = self.reid.try_recover(bbox, self.frame_cnt)

                if track_id == -1:
                    # Check minimum area
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area >= self.MIN_AREA:
                        track_id = self.next_id
                        self.next_id += 1

            if track_id >= 0:
                # Update track state
                old_bbox = self.active_tracks.get(track_id)

                # Apply position smoothing if enabled
                if self.position_smoothing and self.velocity_tracker and old_bbox is not None:
                    bbox = self.velocity_tracker.smooth(track_id, bbox, old_bbox)

                # Update velocity if enabled
                if self.velocity_prediction and self.velocity_tracker and old_bbox is not None:
                    self.velocity_tracker.update(track_id, old_bbox, bbox)

                self.active_tracks[track_id] = bbox
                self.track_ages[track_id] = 0
                self.track_keypoints[track_id] = kpts
                self.track_scores[track_id] = score  # Store for adaptive pose skip

                tracked_keypoints.append(kpts)
                tracked_scores.append(score)
                new_track_ids.append(track_id)
                new_bboxes.append(bbox)

        # Age and remove old tracks
        self._age_tracks()

        # Update state for next frame
        self.track_ids_last_frame = new_track_ids
        self.bboxes_last_frame = new_bboxes
        self.frame_cnt += 1

        if len(tracked_keypoints) == 0:
            return np.array([]), np.array([])

        return np.array(tracked_keypoints), np.array(tracked_scores)

    def _match_detections(
        self,
        current_bboxes: List[np.ndarray]
    ) -> Dict[int, int]:
        """Match current detections to existing tracks.

        Args:
            current_bboxes: List of current detection bboxes

        Returns:
            Dict mapping detection index to track_id
        """
        if not self.active_tracks:
            return {}

        # Get previous track bboxes (with optional velocity prediction)
        prev_track_ids = list(self.active_tracks.keys())
        prev_bboxes = []

        for track_id in prev_track_ids:
            bbox = self.active_tracks[track_id]

            # Apply velocity prediction for tracks that weren't seen recently
            if (self.velocity_prediction and
                self.velocity_tracker and
                self.track_ages.get(track_id, 0) > 0):
                frame_delta = self.track_ages[track_id]
                bbox = self.velocity_tracker.predict(track_id, bbox, frame_delta)

            prev_bboxes.append(bbox)

        # Compute matching
        if self.use_hungarian:
            matches, unmatched_prev, unmatched_curr = hungarian_matching(
                compute_iou_matrix(prev_bboxes, current_bboxes),
                min_iou=self.tracking_thr
            )
        else:
            # Fallback to greedy matching (original behavior)
            matches, unmatched_prev, unmatched_curr = self._greedy_match(
                prev_bboxes, current_bboxes
            )

        # Filter teleporting matches
        if self.teleport_detection and self.teleport_detector:
            valid_matches = []
            for prev_idx, curr_idx in matches:
                prev_bbox = self.active_tracks[prev_track_ids[prev_idx]]
                curr_bbox = current_bboxes[curr_idx]

                if self.teleport_detector.validate_match(prev_bbox, curr_bbox):
                    valid_matches.append((prev_idx, curr_idx))
                else:
                    unmatched_prev.append(prev_idx)
                    unmatched_curr.append(curr_idx)

            matches = valid_matches

        # Try distance-based matching for unmatched tracks (fallback)
        if unmatched_prev and unmatched_curr and self.use_hungarian:
            remaining_prev_bboxes = [prev_bboxes[i] for i in unmatched_prev]
            remaining_curr_bboxes = [current_bboxes[i] for i in unmatched_curr]

            dist_matches, _, _ = distance_matching(
                remaining_prev_bboxes,
                remaining_curr_bboxes,
                max_distance=100.0  # pixels
            )

            # Add distance matches to main matches
            for local_prev, local_curr in dist_matches:
                global_prev = unmatched_prev[local_prev]
                global_curr = unmatched_curr[local_curr]

                # Validate teleportation for distance matches too
                if self.teleport_detection and self.teleport_detector:
                    prev_bbox = self.active_tracks[prev_track_ids[global_prev]]
                    curr_bbox = current_bboxes[global_curr]
                    if not self.teleport_detector.validate_match(prev_bbox, curr_bbox):
                        continue

                matches.append((global_prev, global_curr))

        # Build assignment dict
        assignments: Dict[int, int] = {}
        for prev_idx, curr_idx in matches:
            track_id = prev_track_ids[prev_idx]
            assignments[curr_idx] = track_id

        return assignments

    def _greedy_match(
        self,
        prev_bboxes: List[np.ndarray],
        curr_bboxes: List[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU matching (original algorithm).

        Args:
            prev_bboxes: Previous frame bboxes
            curr_bboxes: Current frame bboxes

        Returns:
            (matches, unmatched_prev, unmatched_curr)
        """
        matches = []
        matched_prev = set()
        matched_curr = set()

        for curr_idx, curr_bbox in enumerate(curr_bboxes):
            best_iou = -1
            best_prev_idx = -1

            for prev_idx, prev_bbox in enumerate(prev_bboxes):
                if prev_idx in matched_prev:
                    continue

                iou = compute_iou(prev_bbox, curr_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_idx = prev_idx

            if best_iou >= self.tracking_thr and best_prev_idx >= 0:
                matches.append((best_prev_idx, curr_idx))
                matched_prev.add(best_prev_idx)
                matched_curr.add(curr_idx)

        unmatched_prev = [i for i in range(len(prev_bboxes)) if i not in matched_prev]
        unmatched_curr = [i for i in range(len(curr_bboxes)) if i not in matched_curr]

        return matches, unmatched_prev, unmatched_curr

    def _age_tracks(self):
        """Age tracks and remove/store old ones."""
        to_remove = []

        for track_id in self.active_tracks:
            self.track_ages[track_id] = self.track_ages.get(track_id, 0) + 1

            if self.track_ages[track_id] > self.max_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            # Store for potential re-identification
            if self.enable_reid and self.reid and self.velocity_tracker:
                velocity = self.velocity_tracker.get_velocity(track_id) or (0, 0)
                self.reid.store(
                    track_id=track_id,
                    last_bbox=self.active_tracks[track_id],
                    last_velocity=velocity,
                    frame_id=self.frame_cnt,
                    last_keypoints=self.track_keypoints.get(track_id)
                )

            # Remove from active tracking
            del self.active_tracks[track_id]
            del self.track_ages[track_id]
            self.track_keypoints.pop(track_id, None)

            if self.velocity_tracker:
                self.velocity_tracker.remove_track(track_id)
            self.pose_skip_count.pop(track_id, None)
            self.track_scores.pop(track_id, None)

    def _adaptive_pose_inference(
        self,
        image: np.ndarray,
        bboxes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose inference with adaptive skipping for stationary tracks.

        Args:
            image: Input image
            bboxes: List of detection bounding boxes

        Returns:
            Tuple of (keypoints, scores) for all bboxes
        """
        # Quick IoU matching to determine which bboxes match existing tracks
        prev_track_ids = list(self.active_tracks.keys())
        prev_bboxes = [self.active_tracks[tid] for tid in prev_track_ids]

        iou_matrix = compute_iou_matrix(prev_bboxes, list(bboxes))

        # Find best match for each bbox
        bbox_to_track: Dict[int, int] = {}
        if iou_matrix.size > 0:
            for bbox_idx in range(len(bboxes)):
                if iou_matrix.shape[0] > 0:
                    best_prev_idx = np.argmax(iou_matrix[:, bbox_idx])
                    if iou_matrix[best_prev_idx, bbox_idx] >= self.tracking_thr:
                        bbox_to_track[bbox_idx] = prev_track_ids[best_prev_idx]

        # Determine which bboxes need fresh pose inference
        need_pose_indices = []
        reuse_pose_indices = []

        for bbox_idx, bbox in enumerate(bboxes):
            track_id = bbox_to_track.get(bbox_idx, -1)
            if track_id >= 0 and not self._should_run_pose(track_id, np.array(bbox)):
                reuse_pose_indices.append((bbox_idx, track_id))
            else:
                need_pose_indices.append(bbox_idx)

        # Prepare output arrays
        n_bboxes = len(bboxes)
        if n_bboxes == 0:
            return np.array([]), np.array([])

        # Run pose only on bboxes that need it
        all_keypoints = [None] * n_bboxes
        all_scores = [None] * n_bboxes

        if need_pose_indices:
            pose_bboxes = [bboxes[i] for i in need_pose_indices]
            new_keypoints, new_scores = self.pose_model(image, bboxes=pose_bboxes)

            for i, bbox_idx in enumerate(need_pose_indices):
                all_keypoints[bbox_idx] = new_keypoints[i]
                all_scores[bbox_idx] = new_scores[i]

        # Reuse keypoints for stationary tracks (with position adjustment)
        for bbox_idx, track_id in reuse_pose_indices:
            old_bbox = self.active_tracks[track_id]
            new_bbox = np.array(bboxes[bbox_idx])
            adjusted_kpts = self._adjust_keypoints_for_bbox(track_id, old_bbox, new_bbox)
            all_keypoints[bbox_idx] = adjusted_kpts
            all_scores[bbox_idx] = self.track_scores.get(track_id, np.ones(adjusted_kpts.shape[0]))

        # Stack results
        keypoints = np.array([k for k in all_keypoints if k is not None])
        scores = np.array([s for s in all_scores if s is not None])

        return keypoints, scores

    def _should_run_pose(self, track_id: int, current_bbox: np.ndarray) -> bool:
        """Check if a track needs fresh pose estimation.

        Args:
            track_id: Track identifier
            current_bbox: Current bounding box

        Returns:
            True if pose inference should be run, False to reuse previous
        """
        if not self.adaptive_pose:
            return True

        if track_id not in self.track_keypoints:
            return True

        prev_bbox = self.active_tracks.get(track_id)
        if prev_bbox is None:
            return True

        # Check motion (center displacement)
        center_prev = np.array([
            (prev_bbox[0] + prev_bbox[2]) / 2,
            (prev_bbox[1] + prev_bbox[3]) / 2
        ])
        center_curr = np.array([
            (current_bbox[0] + current_bbox[2]) / 2,
            (current_bbox[1] + current_bbox[3]) / 2
        ])
        displacement = np.linalg.norm(center_curr - center_prev)

        # Check skip count
        skip_count = self.pose_skip_count.get(track_id, 0)

        if displacement < self.motion_threshold and skip_count < self.pose_reuse_frames:
            self.pose_skip_count[track_id] = skip_count + 1
            return False  # Reuse previous keypoints

        self.pose_skip_count[track_id] = 0
        return True

    def _adjust_keypoints_for_bbox(
        self,
        track_id: int,
        old_bbox: np.ndarray,
        new_bbox: np.ndarray
    ) -> np.ndarray:
        """Adjust keypoints from old bbox to new bbox position.

        Args:
            track_id: Track identifier
            old_bbox: Previous bounding box
            new_bbox: Current bounding box

        Returns:
            Adjusted keypoints
        """
        keypoints = self.track_keypoints[track_id].copy()

        # Compute translation
        old_center = np.array([
            (old_bbox[0] + old_bbox[2]) / 2,
            (old_bbox[1] + old_bbox[3]) / 2
        ])
        new_center = np.array([
            (new_bbox[0] + new_bbox[2]) / 2,
            (new_bbox[1] + new_bbox[3]) / 2
        ])
        translation = new_center - old_center

        # Apply translation to keypoints
        keypoints[:, :2] += translation

        return keypoints

    def get_track_info(self) -> dict:
        """Get current tracking statistics.

        Returns:
            Dict with tracking information
        """
        return {
            'active_tracks': len(self.active_tracks),
            'next_id': self.next_id,
            'frame_count': self.frame_cnt,
            'terminated_tracks': self.reid.get_stored_count() if self.reid else 0,
            'track_ages': dict(self.track_ages),
        }

    # Legacy compatibility method
    def track_by_iou(self, bbox: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
        """Legacy method for backward compatibility.

        Use the new tracking system via __call__ instead.
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        max_iou_score = -1
        max_index = -1
        match_result = None

        for index, each_bbox in enumerate(self.bboxes_last_frame):
            iou_score = compute_iou(bbox, each_bbox)
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                max_index = index

        if max_iou_score > self.tracking_thr:
            track_id = self.track_ids_last_frame.pop(max_index)
            match_result = self.bboxes_last_frame.pop(max_index)
        elif area >= self.MIN_AREA:
            track_id = self.next_id
            self.next_id += 1
        else:
            track_id = -1

        return track_id, match_result
