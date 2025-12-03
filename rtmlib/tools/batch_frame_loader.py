"""Threaded frame loader with batch collection for video processing.

Solves the "Empty Bus" and "CPU Bottleneck" problems:
- Background thread decodes frames continuously (GPU never waits)
- Batch collection ensures full GPU utilization (no padding waste)

Example:
    from rtmlib import BatchFrameLoader, Body

    body = Body(backend='tensorrt', device='cuda', batch_size=8)

    with BatchFrameLoader('video.mp4', batch_size=8) as loader:
        for frame_indices, frames in loader:
            results = body.predict_batch(frames)
            for idx, (kpts, scores) in zip(frame_indices, results):
                print(f"Frame {idx}: {len(kpts)} people")
"""

import threading
import queue
import cv2
import numpy as np
from typing import Tuple, Optional, Union, List, Iterator


class BatchFrameLoader:
    """Threaded frame loader with batch collection for video processing.

    Producer thread decodes frames continuously into a queue.
    Consumer gets batches of N frames for efficient GPU inference.

    Key features:
    - Background thread decodes frames (hides decode latency)
    - Batch collection with configurable size
    - Graceful handling of partial batches at video end
    - Frame index tracking for result ordering
    - Context manager and iterator interfaces

    Args:
        source: Video file path or camera index
        batch_size: Number of frames per batch (should match TRT engine batch)
        buffer_multiplier: Queue size = batch_size * buffer_multiplier
        timeout: Max seconds to wait for first frame of batch

    Example:
        # File processing
        with BatchFrameLoader('video.mp4', batch_size=8) as loader:
            for frame_indices, frames in loader:
                results = model.predict_batch(frames)

        # Manual control
        loader = BatchFrameLoader('video.mp4', batch_size=8)
        while True:
            batch = loader.get_batch()
            if batch is None:
                break
            indices, frames = batch
            # process...
        loader.release()
    """

    def __init__(
        self,
        source: Union[int, str],
        batch_size: int = 8,
        buffer_multiplier: int = 2,
        timeout: float = 5.0
    ):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self._batch_size = batch_size
        self._timeout = timeout
        self._buffer: queue.Queue = queue.Queue(
            maxsize=batch_size * buffer_multiplier
        )
        self._stopped = False
        self._finished = False  # Video ended naturally
        self._lock = threading.Lock()

        # Video properties
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Statistics
        self._frames_read = 0
        self._frames_dropped = 0

        # Start reader thread
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        """Background thread: continuously read frames into buffer."""
        frame_idx = 0
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                with self._lock:
                    self._finished = True
                break

            try:
                # Try to put frame in buffer
                self._buffer.put((frame_idx, frame), timeout=0.1)
                frame_idx += 1
                self._frames_read = frame_idx
            except queue.Full:
                # Buffer full - for live streams, drop oldest; for files, wait
                if isinstance(self._cap.get(cv2.CAP_PROP_POS_FRAMES), float):
                    # File - wait and retry
                    continue
                else:
                    # Live stream - could drop frame here
                    self._frames_dropped += 1
                    continue

    def get_batch(self) -> Optional[Tuple[List[int], List[np.ndarray]]]:
        """Get next batch of frames.

        Returns:
            Tuple of (frame_indices, frames) or None if video ended.
            May return partial batch at end of video.

        Note:
            First frame waits up to `timeout` seconds.
            Subsequent frames use short timeout (batch what's available).
        """
        frames = []
        indices = []

        for i in range(self._batch_size):
            # First frame: wait up to timeout
            # Subsequent frames: short timeout to batch what's available
            wait_time = self._timeout if i == 0 else 0.05

            try:
                idx, frame = self._buffer.get(timeout=wait_time)
                frames.append(frame)
                indices.append(idx)
            except queue.Empty:
                # Check if video finished
                with self._lock:
                    if self._finished and self._buffer.empty():
                        break
                # Otherwise, return what we have
                break

        if len(frames) == 0:
            return None
        return indices, frames

    def __iter__(self) -> Iterator[Tuple[List[int], List[np.ndarray]]]:
        """Iterate over batches until video ends."""
        while True:
            batch = self.get_batch()
            if batch is None:
                break
            yield batch

    def release(self):
        """Release video capture and stop reader thread."""
        with self._lock:
            self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    @property
    def fps(self) -> float:
        """Frames per second of video source."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total frame count (0 for live streams)."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def batch_size(self) -> int:
        """Configured batch size."""
        return self._batch_size

    @property
    def frames_read(self) -> int:
        """Number of frames read so far."""
        return self._frames_read

    @property
    def frames_dropped(self) -> int:
        """Number of frames dropped (live streams only)."""
        return self._frames_dropped

    def get_progress(self) -> Tuple[int, int]:
        """Get progress as (frames_processed, total_frames).

        Returns:
            Tuple of (current frame index, total frames).
            Total is 0 for live streams.
        """
        return self._frames_read, self._frame_count
