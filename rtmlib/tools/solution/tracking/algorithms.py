"""
Core tracking algorithms for pose tracking.

Provides:
- IoU (Intersection over Union) for bounding box overlap
- Hungarian algorithm for optimal assignment
- Distance-based matching as fallback
"""

import warnings
from typing import List, Tuple, Set
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not available. Falling back to greedy matching. "
        "Install scipy for optimal Hungarian algorithm: pip install scipy"
    )


def compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        bbox_a: First bbox as [x1, y1, x2, y2]
        bbox_b: Second bbox as [x1, y1, x2, y2]

    Returns:
        IoU score in [0, 1], where 1 is perfect overlap
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union_area = float(area_a + area_b - inter_area)

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(
    bboxes_prev: List[np.ndarray],
    bboxes_curr: List[np.ndarray]
) -> np.ndarray:
    """Compute IoU matrix between two sets of bounding boxes (vectorized).

    Args:
        bboxes_prev: List of bboxes from previous frame
        bboxes_curr: List of bboxes from current frame

    Returns:
        IoU matrix of shape (len(prev), len(curr))
    """
    n_prev = len(bboxes_prev)
    n_curr = len(bboxes_curr)

    if n_prev == 0 or n_curr == 0:
        return np.zeros((n_prev, n_curr), dtype=np.float32)

    # Convert to arrays for vectorized computation
    prev = np.array(bboxes_prev, dtype=np.float32)  # (M, 4)
    curr = np.array(bboxes_curr, dtype=np.float32)  # (N, 4)

    # Expand dims for broadcasting: prev (M,1,4), curr (1,N,4)
    prev_exp = prev[:, np.newaxis, :]  # (M, 1, 4)
    curr_exp = curr[np.newaxis, :, :]  # (1, N, 4)

    # Intersection coordinates
    x1 = np.maximum(prev_exp[..., 0], curr_exp[..., 0])
    y1 = np.maximum(prev_exp[..., 1], curr_exp[..., 1])
    x2 = np.minimum(prev_exp[..., 2], curr_exp[..., 2])
    y2 = np.minimum(prev_exp[..., 3], curr_exp[..., 3])

    # Intersection area
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Individual areas
    area_prev = (prev[:, 2] - prev[:, 0]) * (prev[:, 3] - prev[:, 1])
    area_curr = (curr[:, 2] - curr[:, 0]) * (curr[:, 3] - curr[:, 1])

    # Union
    union = area_prev[:, np.newaxis] + area_curr[np.newaxis, :] - inter

    # IoU with division safety
    iou = np.where(union > 0, inter / union, 0)
    return iou.astype(np.float32)


def compute_distance_matrix(
    bboxes_prev: List[np.ndarray],
    bboxes_curr: List[np.ndarray]
) -> np.ndarray:
    """Compute Euclidean distance matrix between bbox centers (vectorized).

    Args:
        bboxes_prev: List of bboxes from previous frame
        bboxes_curr: List of bboxes from current frame

    Returns:
        Distance matrix of shape (len(prev), len(curr))
    """
    n_prev = len(bboxes_prev)
    n_curr = len(bboxes_curr)

    if n_prev == 0 or n_curr == 0:
        return np.zeros((n_prev, n_curr), dtype=np.float32)

    # Convert to arrays
    prev = np.array(bboxes_prev, dtype=np.float32)  # (M, 4)
    curr = np.array(bboxes_curr, dtype=np.float32)  # (N, 4)

    # Compute centers vectorized
    centers_prev = np.stack([
        (prev[:, 0] + prev[:, 2]) / 2,
        (prev[:, 1] + prev[:, 3]) / 2
    ], axis=1)  # (M, 2)

    centers_curr = np.stack([
        (curr[:, 0] + curr[:, 2]) / 2,
        (curr[:, 1] + curr[:, 3]) / 2
    ], axis=1)  # (N, 2)

    # Compute pairwise distances using broadcasting
    # (M, 1, 2) - (1, N, 2) = (M, N, 2)
    diff = centers_prev[:, np.newaxis, :] - centers_curr[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    return dist_matrix.astype(np.float32)


def hungarian_matching(
    iou_matrix: np.ndarray,
    min_iou: float = 0.3
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve assignment problem using Hungarian algorithm.

    Args:
        iou_matrix: IoU matrix of shape (n_prev, n_curr)
        min_iou: Minimum IoU threshold for a valid match

    Returns:
        Tuple of:
        - matches: List of (prev_idx, curr_idx) matched pairs
        - unmatched_prev: Indices of unmatched from previous frame
        - unmatched_curr: Indices of unmatched from current frame
    """
    n_prev, n_curr = iou_matrix.shape

    if n_prev == 0:
        return [], [], list(range(n_curr))
    if n_curr == 0:
        return [], list(range(n_prev)), []

    if SCIPY_AVAILABLE:
        # Optimal Hungarian algorithm
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        matched_prev: Set[int] = set()
        matched_curr: Set[int] = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= min_iou:
                matches.append((int(row), int(col)))
                matched_prev.add(row)
                matched_curr.add(col)
    else:
        # Fallback: greedy matching
        matches, matched_prev, matched_curr = _greedy_matching(iou_matrix, min_iou)

    unmatched_prev = [i for i in range(n_prev) if i not in matched_prev]
    unmatched_curr = [j for j in range(n_curr) if j not in matched_curr]

    return matches, unmatched_prev, unmatched_curr


def _greedy_matching(
    iou_matrix: np.ndarray,
    min_iou: float
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    """Greedy matching fallback when scipy is not available.

    Args:
        iou_matrix: IoU matrix
        min_iou: Minimum IoU threshold

    Returns:
        matches, matched_prev set, matched_curr set
    """
    matches = []
    matched_prev: Set[int] = set()
    matched_curr: Set[int] = set()

    n_prev, n_curr = iou_matrix.shape

    # Find matches greedily by highest IoU first
    while True:
        # Find maximum IoU among unmatched pairs
        max_iou = -1
        max_i, max_j = -1, -1

        for i in range(n_prev):
            if i in matched_prev:
                continue
            for j in range(n_curr):
                if j in matched_curr:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_i, max_j = i, j

        if max_iou < min_iou or max_i < 0:
            break

        matches.append((max_i, max_j))
        matched_prev.add(max_i)
        matched_curr.add(max_j)

    return matches, matched_prev, matched_curr


def distance_matching(
    bboxes_prev: List[np.ndarray],
    bboxes_curr: List[np.ndarray],
    max_distance: float = 100.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match bboxes using center distance (fallback for IoU failures).

    Args:
        bboxes_prev: Bboxes from previous frame
        bboxes_curr: Bboxes from current frame
        max_distance: Maximum distance in pixels for a valid match

    Returns:
        Tuple of (matches, unmatched_prev, unmatched_curr)
    """
    n_prev = len(bboxes_prev)
    n_curr = len(bboxes_curr)

    if n_prev == 0:
        return [], [], list(range(n_curr))
    if n_curr == 0:
        return [], list(range(n_prev)), []

    dist_matrix = compute_distance_matrix(bboxes_prev, bboxes_curr)

    if SCIPY_AVAILABLE:
        row_indices, col_indices = linear_sum_assignment(dist_matrix)

        matches = []
        matched_prev: Set[int] = set()
        matched_curr: Set[int] = set()

        for row, col in zip(row_indices, col_indices):
            if dist_matrix[row, col] <= max_distance:
                matches.append((int(row), int(col)))
                matched_prev.add(row)
                matched_curr.add(col)
    else:
        # Greedy distance matching
        matches = []
        matched_prev: Set[int] = set()
        matched_curr: Set[int] = set()

        while True:
            min_dist = float('inf')
            min_i, min_j = -1, -1

            for i in range(n_prev):
                if i in matched_prev:
                    continue
                for j in range(n_curr):
                    if j in matched_curr:
                        continue
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        min_i, min_j = i, j

            if min_dist > max_distance or min_i < 0:
                break

            matches.append((min_i, min_j))
            matched_prev.add(min_i)
            matched_curr.add(min_j)

    unmatched_prev = [i for i in range(n_prev) if i not in matched_prev]
    unmatched_curr = [j for j in range(n_curr) if j not in matched_curr]

    return matches, unmatched_prev, unmatched_curr
