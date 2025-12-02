from typing import List, Tuple, Optional
import os
from pathlib import Path

import numpy as np

from ..base import BaseTool
from ..file import download_checkpoint
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine, get_warp_matrix


class RTMPose(BaseTool):
    """RTMPose model with native TensorRT support for fast inference.

    When device='cuda', automatically builds and uses a TensorRT engine
    with GPU preprocessing for optimal performance.

    Supports batch processing for multiple people detection - significantly
    faster than sequential processing when 2+ people are detected.
    """

    # Pre-allocated empty results for zero-detection frames
    _EMPTY_KEYPOINTS = np.zeros((0, 17, 2), dtype=np.float32)
    _EMPTY_SCORES = np.zeros((0, 17), dtype=np.float32)

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 batch_size: int = 8,
                 use_cuda_graphs: bool = True):

        # Batch processing config
        self.batch_size = batch_size
        self.use_cuda_graphs = use_cuda_graphs

        # TensorRT-specific attributes (init before super() for tensorrt backend)
        self._trt_engine = None
        self._trt_context = None
        self._trt_stream = None
        self._trt_input_tensors = {}
        self._trt_output_tensors = {}
        self._trt_input_name = None
        self._cuda_graph = None
        self._graph_captured = False
        self._graph_input_buffer = None
        self._gpu_norm_mean = None
        self._gpu_norm_std = None

        # Batch preprocessing buffers (allocated lazily)
        self._batch_gpu_buffer = None
        self._batch_centers = None
        self._batch_scales = None

        # For tensorrt backend, skip BaseTool init (it doesn't know tensorrt)
        # and handle everything ourselves
        if backend == 'tensorrt' and device == 'cuda':
            # Download model if it's a URL (like BaseTool does)
            if onnx_model.startswith('http'):
                onnx_model = download_checkpoint(onnx_model)

            # Set attributes that BaseTool would set
            self.model_input_size = model_input_size
            self.mean = mean
            self.std = std
            self.backend = backend
            self.device = device
            self.session = None  # Not used for tensorrt
            self.onnx_model_path = onnx_model
            self.to_openpose = to_openpose

            # Pre-compute normalization arrays for CPU fallback
            if mean is not None:
                self._mean_arr = np.array(mean, dtype=np.float32)
                self._std_arr = np.array(std, dtype=np.float32)
            else:
                self._mean_arr = None
                self._std_arr = None

            # Initialize native TensorRT
            self._try_init_tensorrt()
        else:
            # Use standard BaseTool initialization for other backends
            super().__init__(onnx_model, model_input_size, mean, std, backend,
                             device)
            self.to_openpose = to_openpose
            self.onnx_model_path = onnx_model

            # Pre-compute normalization arrays for CPU preprocessing
            if mean is not None:
                self._mean_arr = np.array(mean, dtype=np.float32)
                self._std_arr = np.array(std, dtype=np.float32)
            else:
                self._mean_arr = None
                self._std_arr = None

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        # Route to TensorRT fast path if available
        if self._trt_engine is not None:
            keypoints, scores = self._inference_tensorrt_batch(image, bboxes)
        else:
            # Fallback to ONNX Runtime (CPU or CUDA EP)
            keypoints, scores = self._inference_sequential(image, bboxes)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def _inference_sequential(self, image: np.ndarray, bboxes: list):
        """Sequential inference using ONNX Runtime (fallback path)."""
        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)
        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image (using pre-computed arrays)
        if self._mean_arr is not None:
            resized_img = (resized_img - self._mean_arr) / self._std_arr

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores

    # ==================== TensorRT Methods ====================

    def _try_init_tensorrt(self):
        """Attempt TensorRT initialization with graceful fallback."""
        try:
            import tensorrt as trt
            import torch

            if not torch.cuda.is_available():
                print("[RTMPose] CUDA not available, using ONNX Runtime")
                return

            engine_path = self._get_or_build_engine()
            if engine_path and os.path.exists(engine_path):
                self._load_tensorrt_engine(engine_path)
                print(f"[RTMPose] Using native TensorRT inference")
            else:
                print("[RTMPose] TensorRT engine build failed, using ONNX Runtime")

        except ImportError as e:
            print(f"[RTMPose] TensorRT/PyTorch not installed: {e}, using ONNX Runtime")
        except Exception as e:
            print(f"[RTMPose] TensorRT init failed: {e}, using ONNX Runtime")

    def _get_or_build_engine(self) -> Optional[str]:
        """Get existing TensorRT engine or build a new one."""
        # Generate engine path based on model and config
        model_path = Path(self.onnx_model_path)
        model_stem = model_path.stem
        w, h = self.model_input_size
        size_str = f"{w}x{h}"

        # Cache directory
        cache_dir = Path.home() / ".cache" / "rtmlib" / "trt_engines"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Include batch size in engine name for batch-specific optimization
        engine_name = f"{model_stem}_{size_str}_b{self.batch_size}_fp16.engine"
        engine_path = str(cache_dir / engine_name)

        # Check if engine already exists
        if os.path.exists(engine_path):
            print(f"[RTMPose] TensorRT engine found: {engine_path}")
            return engine_path

        # Build new engine
        print(f"[RTMPose] Building TensorRT engine: {engine_path}")
        return self._build_tensorrt_engine(self.onnx_model_path, engine_path)

    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str) -> Optional[str]:
        """Build TensorRT engine from ONNX model with FP16 precision and batch support."""
        try:
            import tensorrt as trt

            print(f"[RTMPose] Building TensorRT engine from {onnx_path}")
            print(f"[RTMPose] Target batch size: {self.batch_size}")

            # Create builder and network
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, logger)

            # Parse ONNX model
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"[RTMPose] ONNX parse error: {parser.get_error(i)}")
                    return None

            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4GB

            # Enable FP16 for faster inference
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[RTMPose] FP16 enabled")

            # Set optimization profile with configured batch size
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            input_name = input_tensor.name

            # Get shape from ONNX (should be something like [1, 3, 256, 192])
            onnx_shape = input_tensor.shape
            c = onnx_shape[1] if onnx_shape[1] > 0 else 3
            h = onnx_shape[2] if onnx_shape[2] > 0 else self.model_input_size[1]
            w = onnx_shape[3] if onnx_shape[3] > 0 else self.model_input_size[0]

            # Use configured batch size (override ONNX batch=1)
            input_shape = (self.batch_size, c, h, w)
            profile.set_shape(input_name, input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

            print(f"[RTMPose] Input shape: {input_shape}")

            # Build engine
            print("[RTMPose] Building serialized network (this may take a few minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                print("[RTMPose] Failed to build TensorRT engine")
                return None

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            print(f"[RTMPose] Successfully built TensorRT engine: {engine_path}")
            return engine_path

        except ImportError:
            print("[RTMPose] TensorRT not available")
            return None
        except Exception as e:
            print(f"[RTMPose] Failed to build TensorRT engine: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_tensorrt_engine(self, engine_path: str):
        """Load TensorRT engine and allocate GPU buffers."""
        try:
            import tensorrt as trt
            import torch

            print(f"[RTMPose] Loading TensorRT engine: {engine_path}")

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(engine_path, "rb") as f:
                engine_data = f.read()

            self._trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self._trt_context = self._trt_engine.create_execution_context()

            # Allocate buffers
            self._allocate_trt_buffers()

            # Create CUDA stream
            self._trt_stream = torch.cuda.Stream()

            # Pre-compute GPU normalization constants
            self._gpu_norm_mean = torch.tensor(
                [123.675, 116.28, 103.53], device="cuda"
            ).view(1, 3, 1, 1)
            self._gpu_norm_std = torch.tensor(
                [58.395, 57.12, 57.375], device="cuda"
            ).view(1, 3, 1, 1)

            print("[RTMPose] TensorRT engine loaded successfully")

            # Set up CUDA graph after warm-up
            self._setup_cuda_graph()

        except Exception as e:
            print(f"[RTMPose] Failed to load TensorRT engine: {e}")
            import traceback
            traceback.print_exc()
            self._trt_engine = None

    def _allocate_trt_buffers(self):
        """Allocate GPU buffers for TensorRT inference."""
        import torch
        import tensorrt as trt

        self._trt_input_tensors = {}
        self._trt_output_tensors = {}

        for i in range(self._trt_engine.num_io_tensors):
            name = self._trt_engine.get_tensor_name(i)
            dtype = trt.nptype(self._trt_engine.get_tensor_dtype(name))
            shape = tuple(self._trt_engine.get_tensor_shape(name))

            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            elif dtype == np.int64:
                torch_dtype = torch.int64
            else:
                torch_dtype = torch.float32

            # Allocate contiguous GPU tensor
            tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")

            if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._trt_input_tensors[name] = tensor
                self._trt_context.set_tensor_address(name, tensor.data_ptr())
            else:
                self._trt_output_tensors[name] = tensor
                self._trt_context.set_tensor_address(name, tensor.data_ptr())

        # Store input tensor name
        self._trt_input_name = list(self._trt_input_tensors.keys())[0]

        # Find output tensor names (simcc_x and simcc_y)
        output_names = list(self._trt_output_tensors.keys())
        print(f"[RTMPose] TensorRT outputs: {output_names}")

    def _setup_cuda_graph(self):
        """Set up CUDA graph for kernel replay optimization."""
        import torch

        try:
            # Warm-up runs to stabilize kernels
            print("[RTMPose] Warming up for CUDA graph capture...")
            for _ in range(10):
                self._trt_context.execute_async_v3(
                    stream_handle=self._trt_stream.cuda_stream
                )
                self._trt_stream.synchronize()

            # Capture CUDA graph (only if enabled)
            if self.use_cuda_graphs:
                self._cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._cuda_graph, stream=self._trt_stream):
                    self._trt_context.execute_async_v3(
                        stream_handle=self._trt_stream.cuda_stream
                    )
                self._graph_captured = True
                print("[RTMPose] CUDA graph captured successfully")
            else:
                print("[RTMPose] CUDA graphs disabled by config")

        except Exception as e:
            print(f"[RTMPose] CUDA graph capture failed: {e}, using standard execution")
            self._cuda_graph = None
            self._graph_captured = False

    def _inference_tensorrt_batch(self, image: np.ndarray, bboxes: list):
        """True batch TensorRT inference - processes all people in parallel.

        Key optimizations:
        - Single image upload to GPU
        - Batched affine transforms via grid_sample
        - Single TensorRT execution for all people
        - Vectorized postprocessing
        """
        import torch
        import torch.nn.functional as F

        num_people = len(bboxes)
        if num_people == 0:
            return self._EMPTY_KEYPOINTS.copy(), self._EMPTY_SCORES.copy()

        # Batch preprocess - all crops in one pass
        batch_tensor, centers, scales = self._preprocess_batch(image, bboxes)

        # Process in chunks if more people than engine batch size
        all_simcc_x = []
        all_simcc_y = []

        for chunk_start in range(0, num_people, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_people)
            chunk_size = chunk_end - chunk_start
            chunk_tensor = batch_tensor[chunk_start:chunk_end]

            # Pad if needed to match engine batch size
            if chunk_size < self.batch_size:
                pad_size = self.batch_size - chunk_size
                padding = torch.zeros(
                    (pad_size,) + chunk_tensor.shape[1:],
                    dtype=chunk_tensor.dtype,
                    device=chunk_tensor.device
                )
                chunk_tensor = torch.cat([chunk_tensor, padding], dim=0)

            # Copy to TensorRT input buffer
            self._trt_input_tensors[self._trt_input_name].copy_(chunk_tensor)

            # Execute batch inference
            if self._cuda_graph is not None and self._graph_captured:
                self._cuda_graph.replay()
            else:
                self._trt_context.execute_async_v3(
                    stream_handle=self._trt_stream.cuda_stream
                )

            # Get outputs (extract only valid results, not padding)
            output_names = list(self._trt_output_tensors.keys())
            simcc_x = self._trt_output_tensors[output_names[0]][:chunk_size].cpu().numpy()
            simcc_y = self._trt_output_tensors[output_names[1]][:chunk_size].cpu().numpy()

            all_simcc_x.append(simcc_x)
            all_simcc_y.append(simcc_y)

        # Concatenate all chunks
        simcc_x = np.concatenate(all_simcc_x, axis=0)
        simcc_y = np.concatenate(all_simcc_y, axis=0)

        # Batch postprocess
        keypoints, scores = self._postprocess_batch(simcc_x, simcc_y, centers, scales)

        return keypoints, scores

    def _preprocess_batch(self, image: np.ndarray, bboxes: list):
        """Batch GPU preprocessing - all crops in one pass.

        Optimizations:
        - Single image upload to GPU
        - Vectorized center/scale computation
        - Batched affine_grid + grid_sample
        - Single normalization pass
        """
        import torch
        import torch.nn.functional as F

        num_people = len(bboxes)
        w, h = self.model_input_size
        aspect_ratio = w / h
        src_h, src_w = image.shape[:2]

        # Compute centers and scales for all bboxes (CPU - vectorized)
        bboxes_arr = np.array(bboxes, dtype=np.float32)
        centers, scales = self._compute_centers_scales_batch(bboxes_arr, aspect_ratio)

        # Compute all theta matrices (CPU)
        thetas = np.zeros((num_people, 2, 3), dtype=np.float32)
        for i in range(num_people):
            warp_mat = get_warp_matrix(centers[i], scales[i], 0, output_size=(w, h))
            thetas[i] = self._warp_mat_to_theta(warp_mat, src_w, src_h, w, h)

        # Upload image to GPU once
        img_tensor = torch.from_numpy(image).cuda(non_blocking=True)
        img_tensor = img_tensor.permute(2, 0, 1).float()  # (3, H, W)

        # Expand to batch dimension for grid_sample
        # Each person gets the same source image
        img_batch = img_tensor.unsqueeze(0).expand(num_people, -1, -1, -1)

        # Batch affine transforms on GPU
        theta_tensor = torch.from_numpy(thetas).cuda()
        grid = F.affine_grid(theta_tensor, (num_people, 3, h, w), align_corners=False)
        warped = F.grid_sample(img_batch, grid, mode='bilinear', align_corners=False)

        # Batch normalization
        warped = warped.sub_(self._gpu_norm_mean).div_(self._gpu_norm_std)

        return warped, centers, scales

    def _compute_centers_scales_batch(self, bboxes: np.ndarray, aspect_ratio: float):
        """Vectorized center/scale computation for all bboxes.

        Args:
            bboxes: (N, 4) array of xyxy bboxes
            aspect_ratio: w/h of model input

        Returns:
            centers: (N, 2) array
            scales: (N, 2) array
        """
        # bbox_xyxy2cs logic vectorized
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        centers = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], axis=1)

        # Scale with padding=1.25
        bbox_w = (x2 - x1) * 1.25
        bbox_h = (y2 - y1) * 1.25

        # Adjust for aspect ratio
        scales = np.zeros((len(bboxes), 2), dtype=np.float32)
        wider = bbox_w > bbox_h * aspect_ratio
        scales[wider, 0] = bbox_w[wider]
        scales[wider, 1] = bbox_w[wider] / aspect_ratio
        scales[~wider, 0] = bbox_h[~wider] * aspect_ratio
        scales[~wider, 1] = bbox_h[~wider]

        return centers, scales

    def _postprocess_batch(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        centers: np.ndarray,
        scales: np.ndarray,
        simcc_split_ratio: float = 2.0
    ):
        """Vectorized batch postprocessing.

        Args:
            simcc_x: (N, num_keypoints, W*2) SimCC x predictions
            simcc_y: (N, num_keypoints, H*2) SimCC y predictions
            centers: (N, 2) bbox centers
            scales: (N, 2) bbox scales

        Returns:
            keypoints: (N, num_keypoints, 2)
            scores: (N, num_keypoints)
        """
        # Decode simcc - get max positions and scores
        # simcc_x shape: (N, 17, W*2), simcc_y shape: (N, 17, H*2)
        x_locs = np.argmax(simcc_x, axis=2)  # (N, 17)
        y_locs = np.argmax(simcc_y, axis=2)  # (N, 17)

        # Get scores from max values
        N, num_kpts = x_locs.shape
        x_scores = np.take_along_axis(simcc_x, x_locs[:, :, None], axis=2).squeeze(2)
        y_scores = np.take_along_axis(simcc_y, y_locs[:, :, None], axis=2).squeeze(2)
        scores = np.minimum(x_scores, y_scores)  # (N, 17)

        # Stack to (N, 17, 2) and apply split ratio
        locs = np.stack([x_locs, y_locs], axis=2).astype(np.float32)
        keypoints = locs / simcc_split_ratio

        # Rescale keypoints: keypoints / model_size * scale + center - scale/2
        model_size = np.array(self.model_input_size, dtype=np.float32)  # (w, h)
        keypoints = keypoints / model_size * scales[:, None, :]
        keypoints = keypoints + centers[:, None, :] - scales[:, None, :] / 2

        return keypoints, scores

    def _gpu_preprocess(self, image: np.ndarray, bbox: list):
        """GPU-accelerated preprocessing using PyTorch."""
        import torch
        import torch.nn.functional as F

        bbox = np.array(bbox)

        # Get center and scale (CPU - fast matrix math)
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # Adjust scale for aspect ratio (same as top_down_affine)
        w, h = self.model_input_size
        aspect_ratio = w / h
        b_w, b_h = scale[0], scale[1]
        if b_w > b_h * aspect_ratio:
            scale = np.array([b_w, b_w / aspect_ratio])
        else:
            scale = np.array([b_h * aspect_ratio, b_h])

        # Get warp matrix (CPU - fast)
        warp_mat = get_warp_matrix(center, scale, 0, output_size=(w, h))

        # Convert warp matrix to theta format for grid_sample
        # warp_mat is 2x3 affine matrix that maps src -> dst
        # grid_sample needs theta that maps normalized dst coords to normalized src coords
        theta = self._warp_mat_to_theta(warp_mat, image.shape[1], image.shape[0], w, h)

        # Move image to GPU
        img_tensor = torch.from_numpy(image).cuda(non_blocking=True)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

        # Create sampling grid and apply affine transform on GPU
        theta_tensor = torch.from_numpy(theta).cuda().unsqueeze(0).float()
        grid = F.affine_grid(theta_tensor, (1, 3, h, w), align_corners=False)
        warped = F.grid_sample(img_tensor, grid, mode='bilinear', align_corners=False)

        # GPU normalization (in-place operations)
        warped = warped.sub_(self._gpu_norm_mean).div_(self._gpu_norm_std)

        return warped, center, scale

    def _warp_mat_to_theta(self, warp_mat: np.ndarray,
                           src_w: int, src_h: int,
                           dst_w: int, dst_h: int) -> np.ndarray:
        """Convert cv2 warp matrix to PyTorch affine_grid theta format.

        The warp_mat maps source coordinates to destination coordinates.
        affine_grid expects theta that maps normalized destination coords
        to normalized source coords, so we need to invert and normalize.
        """
        # Invert the affine transformation
        # warp_mat: src -> dst, we need dst -> src
        # Extend to 3x3 matrix
        M = np.vstack([warp_mat, [0, 0, 1]])
        M_inv = np.linalg.inv(M)[:2]  # Take 2x3 part

        # Convert to normalized coordinates
        # Source normalization: x' = 2*x/src_w - 1, y' = 2*y/src_h - 1
        # Dest normalization: x' = 2*x/dst_w - 1, y' = 2*y/dst_h - 1
        #
        # theta should transform normalized dest coords to normalized src coords

        # Scaling matrices for normalization/denormalization
        # T_dst_denorm: maps [-1,1] to [0,dst_w-1]
        # T_src_norm: maps [0,src_w-1] to [-1,1]

        # theta = T_src_norm @ M_inv @ T_dst_denorm
        # T_dst_denorm: scale by dst_w/2, dst_h/2, then translate by dst_w/2, dst_h/2
        # T_src_norm: translate by -src_w/2, -src_h/2, then scale by 2/src_w, 2/src_h

        # Simplified: combine all transformations
        theta = np.zeros((2, 3), dtype=np.float32)

        # The transformation is:
        # x_src = M_inv[0,0]*x_dst + M_inv[0,1]*y_dst + M_inv[0,2]
        # y_src = M_inv[1,0]*x_dst + M_inv[1,1]*y_dst + M_inv[1,2]
        #
        # In normalized coords:
        # x_src_norm = 2*x_src/src_w - 1
        # x_dst = (x_dst_norm + 1) * dst_w / 2
        #
        # Substituting and simplifying:
        theta[0, 0] = M_inv[0, 0] * dst_w / src_w
        theta[0, 1] = M_inv[0, 1] * dst_h / src_w
        theta[0, 2] = (M_inv[0, 0] * dst_w / 2 + M_inv[0, 1] * dst_h / 2 + M_inv[0, 2]) * 2 / src_w - 1

        theta[1, 0] = M_inv[1, 0] * dst_w / src_h
        theta[1, 1] = M_inv[1, 1] * dst_h / src_h
        theta[1, 2] = (M_inv[1, 0] * dst_w / 2 + M_inv[1, 1] * dst_h / 2 + M_inv[1, 2]) * 2 / src_h - 1

        return theta
