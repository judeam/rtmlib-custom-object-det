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

    keypoints, scores, bboxes = pose_tracker(frame)

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
    - Batch processing for multi-person pose estimation

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, Custom, etc.
        det_frequency (int): Frequency of object detection.
        tracking (bool): Whether to enable tracking.
        tracking_thr (float): IoU threshold for track matching.
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
        batch_size (int): Batch size for detection model.
        pose_batch_size (int): Batch size for pose model (max people per inference).
        use_cuda_graphs (bool): Enable CUDA graphs for kernel replay optimization.
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
        # Batch processing options
        batch_size: int = 1,
        pose_batch_size: int = 8,
        use_cuda_graphs: bool = True,
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
        max_age: int = 30
    ):
        # Try to pass batch params if solution supports them
        try:
            model = solution(mode=mode,
                             to_openpose=to_openpose,
                             backend=backend,
                             device=device,
                             batch_size=batch_size,
                             pose_batch_size=pose_batch_size,
                             use_cuda_graphs=use_cuda_graphs)
        except TypeError:
            # Fallback for solutions that don't support batch params
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

        # Legacy compatibility
        self.bboxes_last_frame: List[np.ndarray] = []
        self.track_ids_last_frame: List[int] = []

        # Reset components
        if self.velocity_tracker:
            self.velocity_tracker.reset()
        if self.reid:
            self.reid.reset()

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a frame and return tracked keypoints and bounding boxes.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Tuple of (keypoints, scores, bboxes) where:
                keypoints: Shape (num_persons, num_joints, 2)
                scores: Shape (num_persons, num_joints)
                bboxes: Shape (num_persons, 4) in [x1, y1, x2, y2] format
        """
        # Run detection
        if self.det_model is not None:
            if self.frame_cnt % self.det_frequency == 0:
                bboxes = self.det_model(image)
            else:
                bboxes = self.bboxes_last_frame
            keypoints, scores = self.pose_model(image, bboxes=bboxes)
        else:  # rtmo
            keypoints, scores = self.pose_model(image)

        # Apply tracking using helper
        tracked_kpts, tracked_scores, tracked_bboxes, _ = (
            self._process_frame_tracking(keypoints, scores)
        )
        return tracked_kpts, tracked_scores, tracked_bboxes

    def _process_frame_tracking(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Apply tracking to a single frame's pose results.

        This is the core tracking logic extracted for reuse in both
        single-frame and batch processing modes.

        Args:
            keypoints: Pose keypoints from pose model, shape (N, J, 2)
            scores: Confidence scores from pose model, shape (N, J)

        Returns:
            Tuple of (tracked_keypoints, tracked_scores, tracked_bboxes, track_ids)
        """
        # Without tracking - just update bboxes for next frame
        if not self.tracking:
            bboxes_current_frame = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)
                bboxes_current_frame.append(bbox)
            self.bboxes_last_frame = bboxes_current_frame
            self.frame_cnt += 1
            # Return with sequential indices as dummy track IDs
            track_ids = list(range(len(keypoints)))
            bboxes_arr = np.array(bboxes_current_frame) if bboxes_current_frame else np.array([])
            return keypoints, scores, bboxes_arr, track_ids

        # With tracking - empty detections
        if len(keypoints) == 0:
            self._age_tracks()
            self.frame_cnt += 1
            return np.array([]), np.array([]), np.array([]), []

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
            return np.array([]), np.array([]), np.array([]), []

        return (
            np.array(tracked_keypoints),
            np.array(tracked_scores),
            np.array(new_bboxes),
            new_track_ids,
        )

    def __call_batch__(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch process frames with optimized detection + sequential tracking.

        Maximizes GPU throughput by batching detection across all frames
        while maintaining tracking state sequentially. This provides
        significant speedup over calling __call__ individually.

        Args:
            images: List of input images (BGR format from OpenCV).
                All images should have the same dimensions for optimal
                batch processing performance.

        Returns:
            List of (keypoints, scores, bboxes) tuples, one per frame.
                - keypoints: shape (num_people, num_joints, 2)
                - scores: shape (num_people, num_joints)
                - bboxes: shape (num_people, 4) in [x1, y1, x2, y2] format

        Note:
            Frames are processed in order. Tracking state persists across
            the batch and continues from any previous __call__ or
            __call_batch__ invocations. Call reset() to start fresh.
        """
        if not images:
            return []

        n_frames = len(images)

        # RTMO single-stage model: no batch detection available
        if self.det_model is None:
            results = []
            for image in images:
                keypoints, scores = self.pose_model(image)
                tracked_kpts, tracked_scores, tracked_bboxes, _ = (
                    self._process_frame_tracking(keypoints, scores)
                )
                results.append((tracked_kpts, tracked_scores, tracked_bboxes))
            return results

        # Step 1: Determine which frames need detection based on det_frequency
        detection_indices = []
        for i in range(n_frames):
            if (self.frame_cnt + i) % self.det_frequency == 0:
                detection_indices.append(i)

        # Force detection on first frame if cache is empty
        if not self.bboxes_last_frame and 0 not in detection_indices:
            detection_indices.insert(0, 0)

        # Step 2: Batch detection (single GPU call for all detection frames)
        detection_results = {}
        if detection_indices:
            det_frames = [images[i] for i in detection_indices]
            batch_bboxes = self.det_model.predict_batch(det_frames)
            detection_results = dict(zip(detection_indices, batch_bboxes))

        # Step 3: Process each frame with pose estimation + tracking
        results = []
        cached_bboxes = self.bboxes_last_frame

        for i, image in enumerate(images):
            # Get bboxes for this frame
            if i in detection_results:
                bboxes = detection_results[i]
                cached_bboxes = bboxes
            else:
                bboxes = cached_bboxes

            # Pose estimation (batches people internally via TensorRT)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

            # Sequential tracking (preserves track state frame-by-frame)
            tracked_kpts, tracked_scores, tracked_bboxes, _ = (
                self._process_frame_tracking(keypoints, scores)
            )
            results.append((tracked_kpts, tracked_scores, tracked_bboxes))

        return results

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
