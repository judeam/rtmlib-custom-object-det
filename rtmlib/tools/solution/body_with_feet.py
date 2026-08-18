'''
Example:

import cv2
from rtmlib import BodyWithFeet, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body_with_feet = BodyWithFeet(to_openpose=openpose_skeleton,
                  backend=backend,
                  device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body_with_feet(frame)

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

class BodyWithFeet:
    """
    BodyWithFeet class for human pose estimation using the Halpe26 keypoint format.
    This class supports different modes of operation and can output in OpenPose format.
    """

    MODE = {
        'performance': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip',
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip',
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
            'pose_input_size': (192, 256),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (384, 384),
                 det_score_thr: float = 0.5,
                 pose: str = None,
                 pose_input_size: tuple = (192, 256),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 batch_size: int = 1,
                 pose_batch_size: int = 8,
                 use_cuda_graphs: bool = True):
        """
        Initialize the Halpe26 pose estimation model.

        Args:
            det (str, optional): Path to detection model. If None, uses default based on mode.
            det_input_size (tuple, optional): Input size for detection model. Default is (384, 384).
            det_score_thr (float, optional): Detection confidence threshold. Default is 0.5.
            pose (str, optional): Path to pose estimation model. If None, uses default based on mode.
            pose_input_size (tuple, optional): Input size for pose model. Default is (192, 256).
            mode (str, optional): Operation mode ('performance', 'lightweight', or 'balanced'). Default is 'balanced'.
            to_openpose (bool, optional): Whether to convert output to OpenPose format. Default is False.
            backend (str, optional): Backend for inference ('onnxruntime' or 'opencv'). Default is 'onnxruntime'.
            device (str, optional): Device for inference ('cpu' or 'cuda'). Default is 'cpu'.
            batch_size (int, optional): Batch size for detection model. Default is 1.
            pose_batch_size (int, optional): Batch size for pose model (max people per inference). Default is 8.
            use_cuda_graphs (bool, optional): Enable CUDA graphs for kernel replay optimization. Default is True.
        """
        from .. import RFDETRNano, RTMPose

        print(f"[BodyWithFeet DEBUG] Received: backend={backend}, device={device}")

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        self.det_model = RFDETRNano(model_path=det,
                                    model_input_size=det_input_size,
                                    score_thr=det_score_thr,
                                    backend=backend,
                                    device=device,
                                    export_format='engine',
                                    batch_size=batch_size,
                                    use_cuda_graphs=use_cuda_graphs)
        print(f"[BodyWithFeet DEBUG] After RFDETRNano, passing to RTMPose: backend={backend}")
        self.pose_model = RTMPose(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device,
                                  batch_size=pose_batch_size,
                                  use_cuda_graphs=use_cuda_graphs)

    def __call__(self, image: np.ndarray):
        """
        Perform pose estimation on the input image.

        Args:
            image (np.ndarray): Input image for pose estimation.

        Returns:
            tuple: A tuple containing:
                - keypoints (np.ndarray): Estimated keypoint coordinates.
                - scores (np.ndarray): Confidence scores for each keypoint.
                - bboxes (np.ndarray): Person detections in
                  [x1, y1, x2, y2, det_score] format, shape (N, 5).
        """
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)
        return keypoints, scores, bboxes

    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run batch pose estimation on multiple images.

        Optimized for video processing with batch detection.

        Args:
            images: List of input images (BGR format from OpenCV)

        Returns:
            List of (keypoints, scores, bboxes) tuples, one per image.
            keypoints: shape (num_people, 26, 2) for Halpe26 format
            scores: shape (num_people, 26)
            bboxes: shape (num_people, 5) in [x1, y1, x2, y2, det_score] format
        """
        # Batch detection for all frames at once
        all_bboxes = self.det_model.predict_batch(images)

        # Pose estimation packed across the whole frame batch. The pose engine
        # is built at a fixed batch size, so a per-frame call pays for that
        # whole batch to pose the one player a drill frame contains; packing
        # crops from every frame in this batch fills the same execution
        # instead of padding it with zeros. Results are unchanged -- see
        # RTMPose.predict_frames.
        pose_results = self.pose_model.predict_frames(images, all_bboxes)

        return [
            (keypoints, scores, bboxes)
            for (keypoints, scores), bboxes in zip(pose_results, all_bboxes)
        ]
