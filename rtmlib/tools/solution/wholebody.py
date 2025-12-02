'''
Example:

import cv2

from rtmlib import Wholebody, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      backend=backend,
                      device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = wholebody(frame)

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
from typing import List, Optional

import numpy as np

from .. import RFDETRNano, RTMPose
from .utils.types import BodyResult, Keypoint, PoseResult


class Wholebody:

    MODE = {
        'performance': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',  # noqa
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip',  # noqa
            'pose_input_size': (192, 256),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (384, 384),
                 det_score_thr: float = 0.5,
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 batch_size: int = 1,
                 pose_batch_size: int = 8,
                 use_cuda_graphs: bool = True):
        """Initialize Wholebody pose estimation pipeline.

        Args:
            det: Detection model path (None for default)
            det_input_size: Detection model input size
            det_score_thr: Detection confidence threshold
            pose: Pose model path (None for default)
            pose_input_size: Pose model input size
            mode: 'performance', 'balanced', or 'lightweight'
            to_openpose: Convert keypoints to OpenPose format
            backend: 'onnxruntime' or 'tensorrt'
            device: 'cpu' or 'cuda'
            batch_size: Batch size for detection model
            pose_batch_size: Batch size for pose model (max people per inference)
            use_cuda_graphs: Enable CUDA graphs for kernel replay optimization
        """
        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        self.det_model = RFDETRNano(model_path=det,
                                    model_input_size=det_input_size,
                                    score_thr=det_score_thr,
                                    backend=backend,
                                    device=device,
                                    export_format='engine',
                                    batch_size=batch_size,
                                    use_cuda_graphs=use_cuda_graphs)
        self.pose_model = RTMPose(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device,
                                  batch_size=pose_batch_size,
                                  use_cuda_graphs=use_cuda_graphs)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores

    @staticmethod
    def format_result(keypoints_info: np.ndarray) -> List[PoseResult]:

        def format_keypoint_part(
                part: np.ndarray) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (None if all(keypoint is None
                                for keypoint in keypoints) else keypoints)

        def total_score(
                keypoints: Optional[List[Optional[Keypoint]]]) -> float:
            return (sum(
                keypoint.score for keypoint in keypoints
                if keypoint is not None) if keypoints is not None else 0.0)

        pose_results = []

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(
                instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while RTMPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(body_keypoints, total_score(body_keypoints),
                              len(body_keypoints))
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results
