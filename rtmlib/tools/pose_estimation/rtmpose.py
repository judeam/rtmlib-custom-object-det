from typing import List, Tuple

import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine


class RTMPose(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose
        # Pre-compute normalization arrays for faster preprocessing
        if mean is not None:
            self._mean_arr = np.array(mean, dtype=np.float32)
            self._std_arr = np.array(std, dtype=np.float32)
        else:
            self._mean_arr = None
            self._std_arr = None

    # Fixed batch size to avoid TensorRT engine rebuilds
    MAX_BATCH_SIZE = 4

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        n_boxes = len(bboxes)
        if n_boxes == 0:
            return np.array([]).reshape(0, 17, 2), np.array([]).reshape(0, 17)

        # Process in fixed-size batches to avoid TensorRT rebuilds
        all_keypoints = []
        all_scores = []

        for batch_start in range(0, n_boxes, self.MAX_BATCH_SIZE):
            batch_end = min(batch_start + self.MAX_BATCH_SIZE, n_boxes)
            batch_bboxes = bboxes[batch_start:batch_end]
            actual_batch_size = len(batch_bboxes)

            # Batch preprocess
            h, w = self.model_input_size[1], self.model_input_size[0]
            # Always allocate MAX_BATCH_SIZE for consistent TensorRT engine
            batch = np.zeros((self.MAX_BATCH_SIZE, 3, h, w), dtype=np.float32)
            centers = np.zeros((self.MAX_BATCH_SIZE, 2), dtype=np.float32)
            scales = np.zeros((self.MAX_BATCH_SIZE, 2), dtype=np.float32)

            for i, bbox in enumerate(batch_bboxes):
                img, center, scale = self.preprocess(image, bbox)
                batch[i] = img.transpose(2, 0, 1)
                centers[i] = center
                scales[i] = scale

            # Pad remaining slots with copies of first (will be discarded)
            if actual_batch_size < self.MAX_BATCH_SIZE:
                for i in range(actual_batch_size, self.MAX_BATCH_SIZE):
                    batch[i] = batch[0]
                    centers[i] = centers[0]
                    scales[i] = scales[0]

            # Single batched inference with fixed batch size
            outputs = self._batch_inference(batch)

            # Batch postprocess (only take actual results)
            keypoints, scores = self._batch_postprocess(outputs, centers, scales)
            all_keypoints.append(keypoints[:actual_batch_size])
            all_scores.append(scores[:actual_batch_size])

        # Combine all batches
        keypoints = np.concatenate(all_keypoints, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def _batch_inference(self, batch: np.ndarray):
        """Run inference on entire batch at once.

        Args:
            batch: Batched input images of shape (N, 3, H, W)

        Returns:
            Model outputs (batched)
        """
        batch = np.ascontiguousarray(batch, dtype=np.float32)

        if self.backend == 'opencv':
            outNames = self.session.getUnconnectedOutLayersNames()
            self.session.setInput(batch)
            outputs = self.session.forward(outNames)
        elif self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: batch}
            sess_output = [out.name for out in self.session.get_outputs()]
            outputs = self.session.run(sess_output, sess_input)
        elif self.backend == 'openvino':
            results = self.compiled_model(batch)
            output0 = results[self.output_layer0]
            output1 = results[self.output_layer1]
            outputs = [output0, output1]

        return outputs

    def _batch_postprocess(
            self,
            outputs: List[np.ndarray],
            centers: np.ndarray,
            scales: np.ndarray,
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess batched RTMPose model output.

        Args:
            outputs: Batched model outputs [simcc_x, simcc_y]
            centers: Batch centers of shape (N, 2)
            scales: Batch scales of shape (N, 2)
            simcc_split_ratio: Split ratio of simcc

        Returns:
            keypoints: Batched keypoints of shape (N, K, 2)
            scores: Batched scores of shape (N, K)
        """
        # Decode simcc (already batched from model)
        simcc_x, simcc_y = outputs  # (N, K, Wx), (N, K, Wy)
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)  # Handles batch
        keypoints = locs / simcc_split_ratio  # (N, K, 2)

        # Vectorized rescaling: (N, K, 2) * (N, 1, 2) / (2,) + (N, 1, 2) - (N, 1, 2) / 2
        model_size = np.array(self.model_input_size, dtype=np.float32)
        keypoints = keypoints / model_size * scales[:, np.newaxis, :]
        keypoints = keypoints + centers[:, np.newaxis, :] - scales[:, np.newaxis, :] / 2

        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image (using pre-computed arrays)
        if self._mean_arr is not None:
            resized_img = (resized_img - self._mean_arr) / self._std_arr

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores
