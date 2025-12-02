"""
Asynchronous video capture for high-performance inference pipelines.

Provides non-blocking video capture using a producer thread with frame buffering
to prevent GPU idle time waiting for frame reads.
"""

import threading
import queue
import cv2
from typing import Tuple, Optional, Union


class AsyncVideoCapture:
    """Non-blocking video capture using producer thread with frame buffering.

    This class reads frames in a background thread and buffers them,
    allowing the main thread to continue processing without waiting
    for the next frame to be captured.

    Args:
        source: Video source - can be camera index (int) or video file path (str)
        buffer_size: Maximum number of frames to buffer. Higher values provide
            more resilience to processing spikes but increase latency.
            Default is 2 for low-latency applications.

    Example:
        cap = AsyncVideoCapture(0, buffer_size=2)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Process frame...
        cap.release()
    """

    def __init__(self, source: Union[int, str], buffer_size: int = 2):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self.buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self._lock = threading.Lock()

        # Get video properties
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Start reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        """Background thread that continuously reads frames into the buffer."""
        while not self.stopped:
            if not self.buffer.full():
                ret, frame = self.cap.read()
                if not ret:
                    with self._lock:
                        self.stopped = True
                    break
                try:
                    self.buffer.put(frame, timeout=0.1)
                except queue.Full:
                    pass  # Drop frame if buffer is full
            else:
                # Buffer full - grab frame to stay current but don't decode
                self.cap.grab()

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """Read the next frame from the buffer.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and no more frames are available.
        """
        with self._lock:
            if self.buffer.empty() and self.stopped:
                return False, None

        try:
            frame = self.buffer.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None

    def isOpened(self) -> bool:
        """Check if the video capture is still active."""
        with self._lock:
            return not self.stopped or not self.buffer.empty()

    def release(self):
        """Release the video capture and stop the reader thread."""
        with self._lock:
            self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, prop_id: int) -> float:
        """Get a video capture property.

        Args:
            prop_id: OpenCV property ID (e.g., cv2.CAP_PROP_FPS)

        Returns:
            Property value
        """
        return self.cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        """Set a video capture property.

        Args:
            prop_id: OpenCV property ID
            value: Property value to set

        Returns:
            True if successful
        """
        return self.cap.set(prop_id, value)

    @property
    def fps(self) -> float:
        """Get the video frame rate."""
        return self._fps

    @property
    def width(self) -> int:
        """Get the frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get the frame height."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Get the total frame count (-1 for live streams)."""
        return self._frame_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
