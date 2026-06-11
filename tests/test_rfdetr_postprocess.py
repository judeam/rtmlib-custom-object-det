"""Unit tests for RFDETRNano detection postprocessing helpers."""

import numpy as np

from rtmlib.tools.object_detection.rfdetr_nano import RFDETRNano


def _bare_detector():
    """Create an RFDETRNano instance without loading any model."""
    return RFDETRNano.__new__(RFDETRNano)


class TestDedupAndFilter:
    def test_empty_input_passthrough(self):
        det = _bare_detector()
        empty = np.zeros((0, 5), dtype=np.float32)
        result = det._dedup_and_filter_detections(empty)
        assert len(result) == 0

    def test_near_duplicate_dropped(self):
        det = _bare_detector()
        detections = np.array(
            [
                [10.0, 10.0, 110.0, 210.0, 0.95],
                [11.0, 11.0, 111.0, 211.0, 0.60],  # IoU ~0.97 vs first
                [300.0, 10.0, 400.0, 210.0, 0.80],
            ],
            dtype=np.float32,
        )
        result = det._dedup_and_filter_detections(detections)

        assert len(result) == 2
        # Higher-scoring duplicate kept, sorted by score descending
        assert result[0][4] == np.float32(0.95)
        assert result[1][4] == np.float32(0.80)

    def test_moderate_overlap_kept(self):
        det = _bare_detector()
        # Two people standing close: IoU well below 0.9 must be preserved
        detections = np.array(
            [
                [10.0, 10.0, 110.0, 210.0, 0.95],
                [60.0, 10.0, 160.0, 210.0, 0.90],
            ],
            dtype=np.float32,
        )
        result = det._dedup_and_filter_detections(detections)
        assert len(result) == 2

    def test_tiny_box_dropped(self):
        det = _bare_detector()
        detections = np.array(
            [
                [10.0, 10.0, 110.0, 210.0, 0.95],
                [50.0, 50.0, 51.0, 51.0, 0.99],  # 1 px^2: degenerate
                [70.0, 70.0, 70.0, 90.0, 0.99],  # zero width
            ],
            dtype=np.float32,
        )
        result = det._dedup_and_filter_detections(detections)
        assert len(result) == 1
        assert result[0][4] == np.float32(0.95)

    def test_output_sorted_by_score(self):
        det = _bare_detector()
        detections = np.array(
            [
                [10.0, 10.0, 60.0, 110.0, 0.55],
                [200.0, 10.0, 260.0, 110.0, 0.90],
                [400.0, 10.0, 460.0, 110.0, 0.70],
            ],
            dtype=np.float32,
        )
        result = det._dedup_and_filter_detections(detections)
        assert list(result[:, 4]) == sorted(result[:, 4], reverse=True)
        assert len(result) == 3


class TestEmptyBoxesShape:
    def test_empty_boxes_have_score_column(self):
        assert RFDETRNano._EMPTY_BOXES.shape == (0, 5)
