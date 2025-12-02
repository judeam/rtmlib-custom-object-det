'''
Example:

import cv2

from rtmlib import Body, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

body = Body(to_openpose=openpose_skeleton,
                      backend=backend,
                      device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body(frame)

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
import numpy as np


class Body:
    MODE = {
        'performance': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',  # noqa
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det': None,  # Use RFDETRNano default model
            'det_input_size': (384, 384),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',  # noqa
            'pose_input_size': (192, 256),
        }
    }

    RTMO_MODE = {
        'performance': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-384x384-b37118ce_20231211.zip',  # noqa
            'pose_input_size': (384, 384),
        },
        'lightweight': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-384x384-dac2bf74_20231211.zip',  # noqa
            'pose_input_size': (384, 384),
        },
        'balanced': {
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-384x384-39e78cc4_20231211.zip',  # noqa
            'pose_input_size': (384, 384),
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
        """Initialize Body pose estimation pipeline.

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
        if pose is not None and 'rtmo' in pose:
            from .. import RTMO

            self.one_stage = True

            pose = self.RTMO_MODE[mode]['pose']
            pose_input_size = self.RTMO_MODE[mode]['pose_input_size']
            self.pose_model = RTMO(pose,
                                   model_input_size=pose_input_size,
                                   to_openpose=to_openpose,
                                   backend=backend,
                                   device=device)
        else:
            from .. import RFDETRNano, RTMPose

            self.one_stage = False

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
            self.pose_model = RTMPose(pose,
                                      model_input_size=pose_input_size,
                                      to_openpose=to_openpose,
                                      backend=backend,
                                      device=device,
                                      batch_size=pose_batch_size,
                                      use_cuda_graphs=use_cuda_graphs)

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores

    def predict_batch(self, images: list):
        """Run batch pose estimation on multiple images.

        Uses batch detection for improved throughput when processing
        multiple frames (e.g., video processing).

        Args:
            images: List of input images (BGR format from OpenCV)

        Returns:
            List of (keypoints, scores) tuples, one per image
        """
        if self.one_stage:
            # RTMO doesn't support batch processing, fall back to sequential
            return [self.pose_model(img) for img in images]

        # Batch detection for all frames
        all_bboxes = self.det_model.predict_batch(images)

        # Run pose estimation per frame (pose model processes per-person)
        results = []
        for img, bboxes in zip(images, all_bboxes):
            keypoints, scores = self.pose_model(img, bboxes=bboxes)
            results.append((keypoints, scores))

        return results
