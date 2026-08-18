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
