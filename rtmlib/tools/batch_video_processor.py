"""High-throughput video processor with batch inference.

Coordinates threaded frame loading with batch pose estimation.
Handles the full pipeline: decode -> batch -> infer -> collect results.

Example:
    from rtmlib import Wholebody, BatchVideoProcessor

    wholebody = Wholebody(
        backend='tensorrt', device='cuda',
        batch_size=8, pose_batch_size=16
    )

    processor = BatchVideoProcessor(wholebody, batch_size=8)

    # Process file with visualization
    results = processor.process_video('input.mp4', output_path='output.mp4')

    # Process stream with callback
    def on_frame(idx, frame, kpts, scores):
        cv2.imshow('stream', draw_skeleton(frame, kpts, scores))
        return cv2.waitKey(1) != ord('q')

    processor.process_stream(0, frame_callback=on_frame)
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Tuple, List, Union

from .batch_frame_loader import BatchFrameLoader


class BatchVideoProcessor:
    """High-throughput video processor with batch inference.

    Coordinates threaded frame loading with batch pose estimation.
    Automatically matches batch size to detector configuration.

    Args:
        solution: Pose estimation solution (Body, Wholebody, BodyWithFeet, Custom)
        batch_size: Frames per inference batch. If None, auto-detect from detector.
        buffer_multiplier: Frame buffer size = batch_size * buffer_multiplier

    Example:
        processor = BatchVideoProcessor(Wholebody(...), batch_size=8)

        # Process file
        results = processor.process_video('input.mp4', 'output.mp4')

        # Process stream
        processor.process_stream(0, callback=my_callback)
    """

    def __init__(
        self,
        solution,
        batch_size: Optional[int] = None,
        buffer_multiplier: int = 2
    ):
        self.solution = solution

        # Auto-detect batch size from detector if not specified
        if batch_size is None:
            if hasattr(solution, 'det_model') and hasattr(solution.det_model, 'batch_size'):
                batch_size = solution.det_model.batch_size
            else:
                batch_size = 8  # Default

        self.batch_size = batch_size
        self.buffer_multiplier = buffer_multiplier

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        kpt_thr: float = 0.3,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_frames: bool = False
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Process entire video file.

        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video
            kpt_thr: Keypoint confidence threshold for visualization
            progress_callback: Optional callback(processed_frames, total_frames)
            return_frames: If True, include frames in results dict

        Returns:
            Dict mapping frame_index -> (keypoints, scores)
            If return_frames=True: frame_index -> (keypoints, scores, frame)
        """
        from ..visualization.draw import draw_skeleton

        with BatchFrameLoader(
            video_path,
            batch_size=self.batch_size,
            buffer_multiplier=self.buffer_multiplier
        ) as loader:

            results = {}
            writer = None

            try:
                for frame_indices, frames in loader:
                    # Batch inference
                    if hasattr(self.solution, 'predict_batch'):
                        batch_results = self.solution.predict_batch(frames)
                    else:
                        # Fallback for solutions without predict_batch
                        batch_results = [self.solution(f) for f in frames]

                    # Store results with frame indices
                    for idx, frame, (keypoints, scores) in zip(
                        frame_indices, frames, batch_results
                    ):
                        if return_frames:
                            results[idx] = (keypoints, scores, frame)
                        else:
                            results[idx] = (keypoints, scores)

                    # Optional visualization / video writing
                    if output_path:
                        for frame, (kpts, scores) in zip(frames, batch_results):
                            vis_frame = draw_skeleton(
                                frame.copy(), kpts, scores, kpt_thr=kpt_thr
                            )
                            if writer is None:
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                writer = cv2.VideoWriter(
                                    output_path, fourcc, loader.fps,
                                    (loader.width, loader.height)
                                )
                            writer.write(vis_frame)

                    if progress_callback:
                        progress_callback(max(frame_indices) + 1, loader.frame_count)

            finally:
                if writer:
                    writer.release()

        return results

    def process_stream(
        self,
        source: Union[int, str],
        frame_callback: Callable[[int, np.ndarray, np.ndarray, np.ndarray], bool],
        max_frames: Optional[int] = None
    ):
        """Process live stream with callback per frame.

        Args:
            source: Camera index (0, 1, ...) or stream URL
            frame_callback: Callback(frame_idx, frame, keypoints, scores) -> continue
                           Return False to stop processing
            max_frames: Optional maximum number of frames to process

        Note:
            Callback is called for each frame in batch order.
            Latency = batch_size * frame_time (frames wait for batch to fill).
            For lowest latency, use batch_size=1 (but lower throughput).
        """
        frames_processed = 0

        with BatchFrameLoader(
            source,
            batch_size=self.batch_size,
            buffer_multiplier=self.buffer_multiplier
        ) as loader:
            for frame_indices, frames in loader:
                # Batch inference
                if hasattr(self.solution, 'predict_batch'):
                    batch_results = self.solution.predict_batch(frames)
                else:
                    batch_results = [self.solution(f) for f in frames]

                # Call callback for each frame
                for idx, frame, (kpts, scores) in zip(
                    frame_indices, frames, batch_results
                ):
                    if not frame_callback(idx, frame, kpts, scores):
                        return

                    frames_processed += 1
                    if max_frames and frames_processed >= max_frames:
                        return

    def iter_video(
        self,
        video_path: str
    ):
        """Iterate over video frames with batch inference.

        Yields:
            Tuple of (frame_index, frame, keypoints, scores)

        Example:
            for idx, frame, kpts, scores in processor.iter_video('video.mp4'):
                # Process each frame
                vis = draw_skeleton(frame, kpts, scores)
                cv2.imshow('frame', vis)
        """
        with BatchFrameLoader(
            video_path,
            batch_size=self.batch_size,
            buffer_multiplier=self.buffer_multiplier
        ) as loader:
            for frame_indices, frames in loader:
                if hasattr(self.solution, 'predict_batch'):
                    batch_results = self.solution.predict_batch(frames)
                else:
                    batch_results = [self.solution(f) for f in frames]

                for idx, frame, (kpts, scores) in zip(
                    frame_indices, frames, batch_results
                ):
                    yield idx, frame, kpts, scores

    def to_dataframe(
        self,
        results: Dict[int, Tuple[np.ndarray, np.ndarray]],
        keypoint_names: Optional[List[str]] = None
    ):
        """Convert results to pandas DataFrame.

        Args:
            results: Results dict from process_video()
            keypoint_names: Optional list of keypoint names

        Returns:
            pandas DataFrame with columns: frame, person, keypoint, x, y, score

        Note:
            Requires pandas to be installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe(). Install with: pip install pandas")

        rows = []
        for frame_idx in sorted(results.keys()):
            # Handle both (kpts, scores) and (kpts, scores, frame) formats
            result = results[frame_idx]
            keypoints = result[0]
            scores = result[1]

            for person_idx, (person_kpts, person_scores) in enumerate(
                zip(keypoints, scores)
            ):
                for kpt_idx, ((x, y), score) in enumerate(
                    zip(person_kpts, person_scores)
                ):
                    kpt_name = (
                        keypoint_names[kpt_idx]
                        if keypoint_names
                        else f"kpt_{kpt_idx}"
                    )
                    rows.append({
                        'frame': frame_idx,
                        'person': person_idx,
                        'keypoint': kpt_name,
                        'x': float(x),
                        'y': float(y),
                        'score': float(score)
                    })

        return pd.DataFrame(rows)

    def to_parquet(
        self,
        results: Dict[int, Tuple[np.ndarray, np.ndarray]],
        output_path: str,
        keypoint_names: Optional[List[str]] = None
    ):
        """Save results to parquet file.

        Args:
            results: Results dict from process_video()
            output_path: Path for output parquet file
            keypoint_names: Optional list of keypoint names
        """
        df = self.to_dataframe(results, keypoint_names)
        df.to_parquet(output_path, index=False)
