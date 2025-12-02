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
    """

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        # TensorRT-specific attributes (init before super() for tensorrt backend)
        self._trt_engine = None
        self._trt_context = None
        self._trt_stream = None
        self._trt_input_tensors = {}
        self._trt_output_tensors = {}
        self._trt_input_name = None
        self._cuda_graph = None
        self._graph_input_buffer = None
        self._gpu_norm_mean = None
        self._gpu_norm_std = None

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

        engine_name = f"{model_stem}_{size_str}_fp16.engine"
        engine_path = str(cache_dir / engine_name)

        # Check if engine already exists
        if os.path.exists(engine_path):
            print(f"[RTMPose] TensorRT engine found: {engine_path}")
            return engine_path

        # Build new engine
        print(f"[RTMPose] Building TensorRT engine: {engine_path}")
        return self._build_tensorrt_engine(self.onnx_model_path, engine_path)

    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str) -> Optional[str]:
        """Build TensorRT engine from ONNX model with FP16 precision."""
        try:
            import tensorrt as trt

            print(f"[RTMPose] Building TensorRT engine from {onnx_path}")

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

            # Set optimization profile for static batch size
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            input_name = input_tensor.name

            # Get shape from ONNX (should be something like [1, 3, 256, 192])
            onnx_shape = input_tensor.shape
            batch_size = onnx_shape[0] if onnx_shape[0] > 0 else 1
            c = onnx_shape[1] if onnx_shape[1] > 0 else 3
            h = onnx_shape[2] if onnx_shape[2] > 0 else self.model_input_size[1]
            w = onnx_shape[3] if onnx_shape[3] > 0 else self.model_input_size[0]

            input_shape = (batch_size, c, h, w)
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

            # Capture CUDA graph
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph, stream=self._trt_stream):
                self._trt_context.execute_async_v3(
                    stream_handle=self._trt_stream.cuda_stream
                )

            print("[RTMPose] CUDA graph captured successfully")

        except Exception as e:
            print(f"[RTMPose] CUDA graph capture failed: {e}, using standard execution")
            self._cuda_graph = None

    def _inference_tensorrt_batch(self, image: np.ndarray, bboxes: list):
        """Fast TensorRT inference with GPU preprocessing."""
        import torch
        import torch.nn.functional as F

        keypoints_list, scores_list = [], []

        for bbox in bboxes:
            # GPU preprocessing
            img_tensor, center, scale = self._gpu_preprocess(image, bbox)

            # Copy to TensorRT input buffer
            self._trt_input_tensors[self._trt_input_name].copy_(img_tensor)

            # Execute inference (use CUDA graph if available)
            if self._cuda_graph is not None:
                self._cuda_graph.replay()
            else:
                self._trt_context.execute_async_v3(
                    stream_handle=self._trt_stream.cuda_stream
                )
            self._trt_stream.synchronize()

            # Get outputs and convert to numpy
            output_names = list(self._trt_output_tensors.keys())
            simcc_x = self._trt_output_tensors[output_names[0]].cpu().numpy()
            simcc_y = self._trt_output_tensors[output_names[1]].cpu().numpy()

            # Postprocess
            kpts, score = self.postprocess([simcc_x, simcc_y], center, scale)
            keypoints_list.append(kpts)
            scores_list.append(score)

        keypoints = np.concatenate(keypoints_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
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
