"""Regression tests for BatchVideoProcessor result unpacking.

BodyWithFeet.predict_batch returns 3-tuples (keypoints, scores, bboxes),
while older solutions return 2-tuples. BatchVideoProcessor must handle both
without raising ValueError (it previously unpacked 2-tuples directly).
"""

from unittest.mock import patch

import numpy as np
import pytest

from rtmlib import BatchVideoProcessor


class FakeLoader:
    """Stand-in for BatchFrameLoader yielding synthetic frame batches."""

    fps = 30.0
    width = 64
    height = 48
    frame_count = 4

    def __init__(self, *args, **kwargs):
        self._batches = [
            ([0, 1], [self._frame(), self._frame()]),
            ([2, 3], [self._frame(), self._frame()]),
        ]

    @staticmethod
    def _frame():
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __iter__(self):
        return iter(self._batches)


class FakeSolution3Tuple:
    """Solution whose predict_batch returns (keypoints, scores, bboxes)."""

    def predict_batch(self, frames):
        results = []
        for _ in frames:
            keypoints = np.zeros((1, 26, 2), dtype=np.float32)
            scores = np.ones((1, 26), dtype=np.float32)
            bboxes = np.array([[0.0, 0.0, 10.0, 20.0, 0.9]], dtype=np.float32)
            results.append((keypoints, scores, bboxes))
        return results


class FakeSolution2Tuple:
    """Solution whose predict_batch returns legacy (keypoints, scores)."""

    def predict_batch(self, frames):
        results = []
        for _ in frames:
            keypoints = np.zeros((1, 26, 2), dtype=np.float32)
            scores = np.ones((1, 26), dtype=np.float32)
            results.append((keypoints, scores))
        return results


@pytest.mark.parametrize(
    "solution_cls", [FakeSolution3Tuple, FakeSolution2Tuple]
)
def test_process_video_handles_2_and_3_tuples(solution_cls):
    processor = BatchVideoProcessor(solution_cls(), batch_size=2)

    with patch(
        "rtmlib.tools.batch_video_processor.BatchFrameLoader", FakeLoader
    ):
        results = processor.process_video("fake.mp4")

    assert sorted(results.keys()) == [0, 1, 2, 3]
    keypoints, scores = results[0]
    assert keypoints.shape == (1, 26, 2)
    assert scores.shape == (1, 26)


@pytest.mark.parametrize(
    "solution_cls", [FakeSolution3Tuple, FakeSolution2Tuple]
)
def test_iter_video_yields_bboxes(solution_cls):
    processor = BatchVideoProcessor(solution_cls(), batch_size=2)

    with patch(
        "rtmlib.tools.batch_video_processor.BatchFrameLoader", FakeLoader
    ):
        items = list(processor.iter_video("fake.mp4"))

    assert len(items) == 4
    for idx, frame, kpts, scores, bboxes in items:
        assert kpts.shape == (1, 26, 2)
        assert scores.shape == (1, 26)
        if solution_cls is FakeSolution3Tuple:
            assert bboxes is not None
            assert bboxes.shape == (1, 5)
        else:
            assert bboxes is None


def test_process_stream_handles_3_tuples():
    processor = BatchVideoProcessor(FakeSolution3Tuple(), batch_size=2)
    seen = []

    def callback(idx, frame, kpts, scores):
        seen.append(idx)
        return True

    with patch(
        "rtmlib.tools.batch_video_processor.BatchFrameLoader", FakeLoader
    ):
        processor.process_stream("fake.mp4", frame_callback=callback)

    assert seen == [0, 1, 2, 3]
