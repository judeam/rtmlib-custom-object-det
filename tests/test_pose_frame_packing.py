"""Cross-frame packing of the fixed-batch RTMPose engine.

The engine is built with min == opt == max batch, so an execution costs a full
batch whatever it is handed. These tests pin the two properties that make
packing crops from several frames a safe substitute for a per-frame loop:
every frame gets its own crops back, in order, and a frame with no detections
still gets an entry.
"""

import numpy as np
import pytest

from rtmlib.tools.pose_estimation.rtmpose import RTMPose


class _StubPose(RTMPose):
    """RTMPose with the engine and preprocessing replaced by bookkeeping."""

    def __init__(self, batch_size=8, trt=True):
        self.batch_size = batch_size
        self.to_openpose = False
        self._trt_engine = object() if trt else None
        self.executions = []
        self.seen_pairs = []

    def _infer_flat_crops(self, images, frame_ids, bboxes):
        # One execution per engine-batch of crops, not one per frame.
        self.executions.append(
            (len(bboxes) + self.batch_size - 1) // self.batch_size
        )
        self.seen_pairs.append(list(zip(frame_ids, [tuple(b) for b in bboxes])))
        n = len(bboxes)
        keypoints = np.stack([
            np.full((26, 2), float(i), dtype=np.float32) for i in range(n)
        ])
        scores = np.stack([
            np.full((26,), float(i), dtype=np.float32) for i in range(n)
        ])
        return keypoints, scores


def _box(v):
    return [v, v, v + 10, v + 20]


def test_every_frame_gets_its_own_crops_back():
    pose = _StubPose(batch_size=8)
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    bboxes = [[_box(0)], [_box(1), _box(2)], [_box(3)]]

    results = pose.predict_frames(images, bboxes)

    assert len(results) == 3
    # Frame 0 got crop 0; frame 1 got crops 1 and 2; frame 2 got crop 3.
    assert results[0][0].shape == (1, 26, 2)
    assert results[1][0].shape == (2, 26, 2)
    assert results[2][0].shape == (1, 26, 2)
    assert results[0][0][0, 0, 0] == 0.0
    assert list(results[1][0][:, 0, 0]) == [1.0, 2.0]
    assert results[2][0][0, 0, 0] == 3.0


def test_four_single_person_frames_cost_one_execution():
    pose = _StubPose(batch_size=8)
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    bboxes = [[_box(i)] for i in range(4)]

    pose.predict_frames(images, bboxes)

    # The point of the change: four frames, one engine execution -- not four.
    assert pose.executions == [1]


def test_frames_without_detections_take_no_engine_slot():
    pose = _StubPose(batch_size=8)
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    bboxes = [[], [_box(1)], []]

    results = pose.predict_frames(images, bboxes)

    assert len(results) == 3
    assert results[0][0].shape[0] == 0
    assert results[2][0].shape[0] == 0
    assert results[1][0].shape[0] == 1
    # Only the one real box was ever sent to the engine.
    assert [fid for fid, _ in pose.seen_pairs[0]] == [1]


def test_no_detections_at_all_skips_the_engine():
    pose = _StubPose(batch_size=8)
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]

    results = pose.predict_frames(images, [[], []])

    assert pose.executions == []
    assert all(kp.shape[0] == 0 for kp, _ in results)


def test_non_tensorrt_backend_falls_back_to_per_frame():
    calls = []

    class _Fallback(_StubPose):
        def __call__(self, image, bboxes=()):
            calls.append(len(bboxes))
            return np.zeros((len(bboxes), 26, 2)), np.zeros((len(bboxes), 26))

    pose = _Fallback(trt=False)
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]

    results = pose.predict_frames(images, [[_box(0)], [], [_box(2)]])

    assert len(results) == 3
    assert calls == [1, 0, 1]


def test_mismatched_lengths_are_rejected():
    pose = _StubPose()
    with pytest.raises(ValueError, match="same length"):
        pose.predict_frames([np.zeros((4, 4, 3), dtype=np.uint8)], [[], []])
