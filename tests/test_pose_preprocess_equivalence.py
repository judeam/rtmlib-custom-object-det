"""Packing crops across frames must not change the crops themselves.

`_preprocess_batch_multi` is `_preprocess_batch` with a per-crop source image
instead of one shared image. If the two disagree for the single-image case,
cross-frame packing is not a refactor but a silent change to every keypoint,
so this is the test that licenses the whole optimisation.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="preprocessing runs on CUDA"
)

from rtmlib.tools.pose_estimation.rtmpose import RTMPose


def _pose():
    pose = RTMPose.__new__(RTMPose)
    pose.batch_size = 8
    pose.model_input_size = (288, 384)
    pose.to_openpose = False
    pose._trt_engine = object()
    pose._gpu_norm_mean = torch.tensor(
        [123.675, 116.28, 103.53], device="cuda"
    ).view(1, 3, 1, 1)
    pose._gpu_norm_std = torch.tensor(
        [58.395, 57.12, 57.375], device="cuda"
    ).view(1, 3, 1, 1)
    return pose


def _image(seed, h=720, w=1280):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


BOXES = [
    [100.0, 80.0, 260.0, 520.0],
    [640.0, 200.0, 900.0, 700.0],
    [12.0, 4.0, 300.0, 640.0],
]


def test_single_image_path_matches_multi_image_path():
    pose = _pose()
    image = _image(0)

    one, centers_one, scales_one = pose._preprocess_batch(image, BOXES)
    many, centers_many, scales_many = pose._preprocess_batch_multi(
        [image], [0, 0, 0], BOXES
    )

    assert torch.equal(one, many)
    np.testing.assert_array_equal(centers_one, centers_many)
    np.testing.assert_array_equal(scales_one, scales_many)


def test_crops_follow_their_own_frame():
    """A crop packed beside other frames must sample the frame it came from."""
    pose = _pose()
    images = [_image(1), _image(2), _image(3)]

    packed, _, _ = pose._preprocess_batch_multi(
        images, [0, 1, 2], [BOXES[0], BOXES[0], BOXES[0]]
    )

    # Same box on three different frames: each packed crop must equal the crop
    # the single-image path produces for that frame, and they must differ from
    # each other (otherwise the wrong source was indexed).
    for i, image in enumerate(images):
        alone, _, _ = pose._preprocess_batch(image, [BOXES[0]])
        assert torch.equal(packed[i:i + 1], alone)
    assert not torch.equal(packed[0], packed[1])


def test_gpu_tensor_frames_match_numpy_frames():
    """NVDEC hands over CUDA tensors; they must preprocess identically."""
    pose = _pose()
    image = _image(4)
    gpu_image = torch.from_numpy(image).cuda()

    from_numpy, _, _ = pose._preprocess_batch_multi([image], [0], [BOXES[0]])
    from_tensor, _, _ = pose._preprocess_batch_multi(
        [gpu_image], [0], [BOXES[0]]
    )

    assert torch.equal(from_numpy, from_tensor)


def test_repeated_frame_is_uploaded_once_but_crops_are_distinct():
    pose = _pose()
    image = _image(5)

    packed, _, _ = pose._preprocess_batch_multi(
        [image, image], [0, 0, 1], [BOXES[0], BOXES[1], BOXES[2]]
    )

    assert packed.shape[0] == 3
    reference, _, _ = pose._preprocess_batch(image, [BOXES[0], BOXES[1]])
    assert torch.equal(packed[:2], reference)


def test_single_image_preprocess_also_accepts_a_gpu_tensor():
    """Both preprocessors share one upload path, so both take CUDA frames."""
    pose = _pose()
    image = _image(6)

    from_numpy, _, _ = pose._preprocess_batch(image, BOXES)
    from_tensor, _, _ = pose._preprocess_batch(torch.from_numpy(image).cuda(), BOXES)

    assert torch.equal(from_numpy, from_tensor)


EDGE_BOXES = [
    [-60.0, -40.0, 220.0, 420.0],      # hangs off the top-left corner
    [1100.0, 500.0, 1400.0, 900.0],    # hangs off the bottom-right corner
    [0.0, 0.0, 1280.0, 720.0],         # the whole frame
    [600.0, 300.0, 604.0, 330.0],      # tiny
]


def test_windowed_conversion_matches_single_image_path_at_the_edges():
    """Only a window of each frame is converted; crops that touch the frame
    border, leave it, or cover all of it must still match the full-frame path."""
    pose = _pose()
    images = [_image(10 + i) for i in range(len(EDGE_BOXES))]

    packed, _, _ = pose._preprocess_batch_multi(
        images, list(range(len(EDGE_BOXES))), EDGE_BOXES
    )
    for i, (image, box) in enumerate(zip(images, EDGE_BOXES)):
        alone, _, _ = pose._preprocess_batch(image, [box])
        assert torch.equal(packed[i:i + 1], alone), f"box {i}"


def test_stale_staging_planes_are_never_read():
    """The fp32 planes are reused: a later call must not see an earlier frame."""
    pose = _pose()
    first = [_image(20), _image(21)]
    pose._preprocess_batch_multi(first, [0, 1], [EDGE_BOXES[2], EDGE_BOXES[2]])

    second = [_image(22), _image(23)]
    packed, _, _ = pose._preprocess_batch_multi(
        second, [0, 1], [BOXES[0], BOXES[1]]
    )
    for i in range(2):
        alone, _, _ = pose._preprocess_batch(second[i], [[BOXES[0], BOXES[1]][i]])
        assert torch.equal(packed[i:i + 1], alone)


def test_stacked_warp_thetas_match_the_per_crop_functions():
    """``_warp_thetas`` is the per-crop get_warp_matrix + _warp_mat_to_theta, vectorised."""
    from rtmlib.tools.pose_estimation.pre_processings import get_warp_matrix

    pose = RTMPose.__new__(RTMPose)
    w, h = 288, 384
    rng = np.random.default_rng(3)
    boxes = np.concatenate([
        np.array(BOXES + EDGE_BOXES, dtype=np.float32),
        (rng.random((40, 4)) * [900, 500, 300, 300] + [0, 0, 5, 5]).astype(np.float32)
        * [1, 1, 1, 1],
    ])
    boxes[:, 2:] = boxes[:, :2] + np.abs(boxes[:, 2:]) + 2
    centers, scales = pose._compute_centers_scales_batch(boxes, w / h)

    stacked = pose._warp_thetas(centers, scales, 1280, 720, w, h)
    for i in range(len(boxes)):
        warp = get_warp_matrix(centers[i], scales[i], 0, output_size=(w, h))
        np.testing.assert_array_equal(
            stacked[i], pose._warp_mat_to_theta(warp, 1280, 720, w, h)
        )
