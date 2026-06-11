"""Unit tests for RTMPose fixes.

Covers:
- Empty-detection behavior (no phantom full-frame person).
- TRT SimCC batch decode parity with the ONNX path (get_simcc_maximum).
- _warp_mat_to_theta half-pixel correctness vs cv2.warpAffine.
- Bbox tolerance for trailing detection-score column (xyxys).
"""

import numpy as np
import pytest

from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
from rtmlib.tools.pose_estimation.pre_processings import get_warp_matrix
from rtmlib.tools.pose_estimation.rtmpose import RTMPose


def _bare_rtmpose(model_input_size=(192, 256)):
    """Create an RTMPose instance without loading any model."""
    pose = RTMPose.__new__(RTMPose)
    pose.model_input_size = model_input_size
    pose.to_openpose = False
    pose._trt_engine = None
    return pose


class TestEmptyDetections:
    def test_call_with_no_bboxes_returns_empty(self):
        pose = _bare_rtmpose()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        keypoints, scores = pose(image, bboxes=[])

        assert len(keypoints) == 0
        assert len(scores) == 0

    def test_call_with_empty_array_returns_empty(self):
        pose = _bare_rtmpose()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        keypoints, scores = pose(
            image, bboxes=np.zeros((0, 5), dtype=np.float32)
        )

        assert len(keypoints) == 0
        assert len(scores) == 0


class TestSimccDecodeParity:
    def test_batch_decode_matches_onnx_semantics(self):
        """TRT batch decode must produce the same scores/locs as the ONNX path."""
        rng = np.random.default_rng(42)
        n, k = 3, 26
        simcc_x = rng.normal(0.3, 0.5, size=(n, k, 384)).astype(np.float32)
        simcc_y = rng.normal(0.3, 0.5, size=(n, k, 512)).astype(np.float32)
        # Force some keypoints to be entirely non-positive (invalid)
        simcc_x[0, 0] = -1.0
        simcc_y[0, 0] = -1.0

        pose = _bare_rtmpose(model_input_size=(192, 256))
        centers = rng.uniform(100, 500, size=(n, 2)).astype(np.float32)
        scales = rng.uniform(50, 300, size=(n, 2)).astype(np.float32)

        keypoints, scores = pose._postprocess_batch(
            simcc_x, simcc_y, centers, scales
        )

        ref_locs, ref_vals = get_simcc_maximum(simcc_x, simcc_y)
        ref_keypoints = ref_locs / 2.0
        ref_keypoints = (
            ref_keypoints / np.array([192, 256], dtype=np.float32)
            * scales[:, None, :]
        )
        ref_keypoints = ref_keypoints + centers[:, None, :] - scales[:, None, :] / 2

        np.testing.assert_allclose(scores, ref_vals, rtol=1e-5)
        np.testing.assert_allclose(keypoints, ref_keypoints, rtol=1e-4)

    def test_invalid_responses_masked(self):
        """Non-positive SimCC maxima must be masked like the ONNX path."""
        n, k = 1, 2
        simcc_x = np.full((n, k, 100), -1.0, dtype=np.float32)
        simcc_y = np.full((n, k, 100), -1.0, dtype=np.float32)
        simcc_x[0, 1, 50] = 2.0
        simcc_y[0, 1, 60] = 2.0

        pose = _bare_rtmpose(model_input_size=(192, 256))
        centers = np.zeros((n, 2), dtype=np.float32)
        scales = np.array([[192.0, 256.0]], dtype=np.float32)

        keypoints, scores = pose._postprocess_batch(
            simcc_x, simcc_y, centers, scales
        )

        # Keypoint 0 is invalid: score is the mean of negatives, loc masked to -1
        assert scores[0, 0] < 0
        ref_locs, _ = get_simcc_maximum(simcc_x, simcc_y)
        assert np.all(ref_locs[0, 0] == -1)
        # Valid keypoint has positive mean score
        assert scores[0, 1] == pytest.approx(2.0)


class TestWarpMatToTheta:
    @pytest.mark.parametrize(
        "center,scale",
        [
            ((320.0, 240.0), (200.0, 266.67)),
            ((100.0, 400.0), (150.0, 200.0)),
        ],
    )
    def test_grid_sample_matches_warp_affine(self, center, scale):
        """grid_sample with the converted theta must match cv2.warpAffine."""
        torch = pytest.importorskip("torch")
        import cv2
        import torch.nn.functional as F

        src_h, src_w = 480, 640
        dst_w, dst_h = 192, 256

        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(src_h, src_w, 3)).astype(np.uint8)

        warp_mat = get_warp_matrix(
            np.array(center), np.array(scale), 0, output_size=(dst_w, dst_h)
        )
        expected = cv2.warpAffine(
            image, warp_mat, (dst_w, dst_h), flags=cv2.INTER_LINEAR
        ).astype(np.float32)

        pose = _bare_rtmpose(model_input_size=(dst_w, dst_h))
        theta = pose._warp_mat_to_theta(warp_mat, src_w, src_h, dst_w, dst_h)

        img_tensor = (
            torch.from_numpy(image.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        theta_tensor = torch.from_numpy(theta).unsqueeze(0).float()
        grid = F.affine_grid(
            theta_tensor, (1, 3, dst_h, dst_w), align_corners=False
        )
        warped = F.grid_sample(
            img_tensor, grid, mode="bilinear", align_corners=False
        )
        actual = warped.squeeze(0).permute(1, 2, 0).numpy()

        # Interior pixels must match closely (borders differ slightly due to
        # padding behavior between warpAffine and grid_sample).
        diff = np.abs(actual[4:-4, 4:-4] - expected[4:-4, 4:-4])
        assert np.median(diff) < 1.0
        assert np.mean(diff) < 2.0


class TestBboxScoreTolerance:
    def test_compute_centers_scales_accepts_xyxys(self):
        pose = _bare_rtmpose()
        bboxes4 = np.array([[10.0, 20.0, 110.0, 220.0]], dtype=np.float32)
        bboxes5 = np.array([[10.0, 20.0, 110.0, 220.0, 0.9]], dtype=np.float32)

        c4, s4 = pose._compute_centers_scales_batch(bboxes4, 0.75)
        c5, s5 = pose._compute_centers_scales_batch(bboxes5, 0.75)

        np.testing.assert_allclose(c4, c5)
        np.testing.assert_allclose(s4, s5)

    def test_preprocess_accepts_xyxys(self):
        pose = _bare_rtmpose(model_input_size=(192, 256))
        pose._mean_arr = None
        pose._std_arr = None
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        img4, c4, s4 = pose.preprocess(image, [10, 20, 110, 220])
        img5, c5, s5 = pose.preprocess(image, [10, 20, 110, 220, 0.9])

        np.testing.assert_allclose(c4, c5)
        np.testing.assert_allclose(s4, s5)
        assert img4.shape == img5.shape
