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
from typing import List, Tuple

import numpy as np
import torch

from .tracking.algorithms import compute_iou_matrix
from ocsort.ocsort import OCSort


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
    """Pose tracker using OC-SORT for multi-person tracking.

    Uses OC-SORT (Observation-Centric SORT) which provides:
    - Kalman filter motion modeling
    - Observation-Centric Momentum (OCM) for velocity direction consistency
    - Observation-Centric Recovery (OCR) for recovering lost tracks

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, Custom, etc.
        det_frequency (int): Frequency of object detection.
        tracking (bool): Whether to enable tracking.
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
        batch_size (int): Batch size for detection model.
        pose_batch_size (int): Batch size for pose model (max people per inference).
        use_cuda_graphs (bool): Enable CUDA graphs for kernel replay optimization.
        det_thresh (float): Min confidence for OC-SORT high-confidence detections.
        max_age (int): Max frames to coast lost tracks.
        min_hits (int): Min hits before track appears (1 = immediate).
        iou_threshold (float): IoU threshold for matching.
        delta_t (int): OCM velocity direction window.
        inertia (float): OCM inertia weight.
        use_byte (bool): ByteTrack-style second matching pass.
    """

    def __init__(
        self,
        solution: type,
        det_frequency: int = 1,
        tracking: bool = True,
        mode: str = 'balanced',
        to_openpose: bool = False,
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        # Batch processing options
        batch_size: int = 1,
        pose_batch_size: int = 8,
        use_cuda_graphs: bool = True,
        # OC-SORT parameters
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = False,
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

        self._ocsort_config = dict(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            inertia=inertia,
            use_byte=use_byte,
        )

        self.reset()

        if self.tracking:
            print('OC-SORT tracking enabled')

    def reset(self):
        """Reset pose tracker state."""
        self.frame_cnt = 0
        if self.tracking:
            self._tracker = OCSort(**self._ocsort_config)
        self.bboxes_last_frame = []

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

        # Apply tracking
        tracked_kpts, tracked_scores, tracked_bboxes, _ = (
            self._process_frame_tracking(keypoints, scores)
        )
        return tracked_kpts, tracked_scores, tracked_bboxes

    def _process_frame_tracking(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Apply OC-SORT tracking to a single frame's pose results.

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
            track_ids = list(range(len(keypoints)))
            bboxes_arr = np.array(bboxes_current_frame) if bboxes_current_frame else np.array([])
            return keypoints, scores, bboxes_arr, track_ids

        # Empty detections - still call tracker to advance state
        if len(keypoints) == 0:
            self._tracker.update(torch.empty((0, 6)), None)
            self.bboxes_last_frame = []
            self.frame_cnt += 1
            return np.array([]), np.array([]), np.array([]), []

        # Convert keypoints to bboxes
        original_bboxes = np.array([pose_to_bbox(kpts) for kpts in keypoints])

        # Compute per-person confidence as mean of joint scores
        confidences = np.mean(scores, axis=1)

        # Build OC-SORT input: [x1, y1, x2, y2, confidence, class]
        classes = np.zeros(len(keypoints))
        dets = np.column_stack([original_bboxes, confidences, classes]).astype(np.float32)

        # OC-SORT expects torch tensors
        tracks = self._tracker.update(torch.from_numpy(dets), None)

        if len(tracks) == 0:
            self.bboxes_last_frame = []
            self.frame_cnt += 1
            return np.array([]), np.array([]), np.array([]), []

        # Extract tracked bboxes and IDs
        tracked_bboxes = tracks[:, :4]
        track_ids = tracks[:, 4].astype(int).tolist()

        # Map tracked bboxes back to original keypoints via IoU
        iou_matrix = compute_iou_matrix(
            list(tracked_bboxes), list(original_bboxes)
        )

        tracked_keypoints = []
        tracked_scores = []
        final_bboxes = []
        final_track_ids = []
        used_dets = set()

        for track_idx in range(len(tracks)):
            # Find best matching detection for this track (greedy)
            best_det_idx = -1
            best_iou = -1.0
            for det_idx in range(len(original_bboxes)):
                if det_idx in used_dets:
                    continue
                if iou_matrix[track_idx, det_idx] > best_iou:
                    best_iou = iou_matrix[track_idx, det_idx]
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                used_dets.add(best_det_idx)
                tracked_keypoints.append(keypoints[best_det_idx])
                tracked_scores.append(scores[best_det_idx])
                final_bboxes.append(tracked_bboxes[track_idx])
                final_track_ids.append(track_ids[track_idx])

        self.bboxes_last_frame = final_bboxes
        self.frame_cnt += 1

        if len(tracked_keypoints) == 0:
            return np.array([]), np.array([]), np.array([]), []

        return (
            np.array(tracked_keypoints),
            np.array(tracked_scores),
            np.array(final_bboxes),
            final_track_ids,
        )

    def __call_batch__(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch process frames with optimized detection + sequential tracking.

        Args:
            images: List of input images (BGR format from OpenCV).

        Returns:
            List of (keypoints, scores, bboxes) tuples, one per frame.
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

        for i, image in enumerate(images):
            # Get bboxes: fresh detection or Kalman-smoothed from last frame
            if i in detection_results:
                bboxes = detection_results[i]
            else:
                bboxes = self.bboxes_last_frame

            # Pose estimation
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

            # Sequential tracking (preserves track state frame-by-frame)
            tracked_kpts, tracked_scores, tracked_bboxes, _ = (
                self._process_frame_tracking(keypoints, scores)
            )
            results.append((tracked_kpts, tracked_scores, tracked_bboxes))

        return results
