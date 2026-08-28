"""RFDETRNano object detector with person filtering and TensorRT support.

Supports batch processing with CUDA Graphs for maximum throughput.
"""

from typing import Optional, List, Tuple
import numpy as np
import glob
import os
import shutil
from pathlib import Path

from ..base import BaseTool


class RFDETRNano(BaseTool):
    """RFDETRNano object detector with person filtering and TensorRT engine support.

    This detector uses RF-DETR Nano model for person detection, with support for
    TensorRT acceleration. The model is automatically exported to TensorRT engine
    format on first use for optimal inference performance.
    """

    PERSON_CLASS = 5  # Person class in the fine-tuned model (class 5 based on debug output)
    _EMPTY_BOXES = np.array([], dtype=np.float32).reshape(0, 5)  # Pre-allocated empty result
    _DEDUP_IOU_THR = 0.9  # Drop boxes overlapping a higher-scoring box above this IoU
    _MIN_BOX_AREA = 4.0  # Minimum box area in px^2 (sanity filter)

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_input_size: tuple = (384, 384),
        score_thr: float = 0.5,
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        export_format: str = 'engine',
        batch_size: int = 1,
        use_cuda_graphs: bool = True,
    ):
        """Initialize RFDETRNano detector.

        Args:
            model_path: Path to model file. If None, will use default model.
            model_input_size: Input size for the model (height, width).
            score_thr: Score threshold for filtering detections.
            backend: Backend for inference ('onnxruntime', 'pytorch', or 'tensorrt').
            device: Device for inference ('cpu' or 'cuda').
            export_format: Format for export ('engine' for TensorRT, 'onnx' for ONNX).
            batch_size: Batch size for TensorRT inference (default: 1).
            use_cuda_graphs: Whether to use CUDA Graphs for faster inference (default: True).
        """
        self.model_input_size = model_input_size
        self.score_thr = score_thr
        self.backend = backend
        self.device = device
        self.export_format = export_format
        self.batch_size = batch_size
        self.use_cuda_graphs = use_cuda_graphs
        self.model = None
        self.engine_path = None
        self._engine = None
        self._cuda_graph = None
        self._graph_captured = False
        self._preprocess_buffers = {}
        self._pad_buffers = {}

        # Resolve model path
        if model_path is None:
            model_path = self._resolve_model_path()

        self.model_path = model_path
        print(f"[RFDETRNano] Model path: {self.model_path}")

        # Initialize model based on backend
        if backend in ('onnxruntime', 'tensorrt') and device == 'cuda':
            # Try to use TensorRT engine
            self.engine_path = self._get_or_build_engine()
            if self.engine_path and os.path.exists(self.engine_path):
                self._load_tensorrt_engine()
            else:
                # Fallback to PyTorch
                print("TensorRT engine not available, falling back to PyTorch")
                self.backend = 'pytorch'
                self._load_pytorch_model()
        else:
            # Use PyTorch model directly
            self._load_pytorch_model()

    def _candidate_model_paths(self) -> "list[tuple[str, str]]":
        """(models_dir, model_path) pairs to look for the checkpoint in."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        tools_models_dir = os.path.abspath(
            os.path.join(current_dir, '..', 'models')
        )
        # current_dir is rtmlib/tools/object_detection/, so 3 levels up is
        # the project root (for development checkouts).
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(current_dir))
        )
        dev_models_dir = os.path.join(project_root, 'models')

        return [
            (tools_models_dir,
             os.path.join(tools_models_dir, 'rfdetr_nano_person.pt')),
            (dev_models_dir,
             os.path.join(dev_models_dir, 'rfdetr_nano_person.pt')),
        ]

    def _resolve_model_path(self) -> str:
        """Where the checkpoint is (or would be), WITHOUT decompressing it.

        Deliberately does no work beyond path arithmetic. The engine filename
        is derived from this path's stem and parent, so construction needs the
        name long before -- and usually instead of -- the bytes: on the
        TensorRT path a cached engine is loaded and the checkpoint is never
        read at all.

        It used to decompress here, which meant every cold container joined
        three xz parts and wrote 115MB (9.1s measured on an L4 Cloud Run
        instance) to produce a file nothing then opened. Decompression now
        happens in _ensure_model_file, called only where the checkpoint is
        actually about to be loaded.
        """
        candidates = self._candidate_model_paths()

        for _models_dir, model_path in candidates:
            if os.path.exists(model_path):
                return model_path

        # Nothing decompressed yet: name the location the compressed parts
        # would decompress into, so the engine path stays stable either way.
        for models_dir, model_path in candidates:
            if glob.glob(os.path.join(
                    models_dir, 'rfdetr_nano_person.pt.xz.part_*')):
                return model_path

        tools_models_dir = candidates[0][0]
        os.makedirs(tools_models_dir, exist_ok=True)
        raise FileNotFoundError(
            f"Model file not found. Please place 'rfdetr_nano_person.pt' "
            f"or compressed parts 'rfdetr_nano_person.pt.xz.part_*' in: {tools_models_dir}"
        )

    def _ensure_model_file(self) -> str:
        """Materialise the checkpoint, decompressing the xz parts if needed.

        Call this immediately before reading the checkpoint -- never during
        construction. See _resolve_model_path for why.
        """
        if os.path.exists(self.model_path):
            return self.model_path

        models_dir = os.path.dirname(self.model_path)
        if self._decompress_model_if_needed(models_dir, self.model_path):
            return self.model_path

        raise FileNotFoundError(
            f"Model file not found and could not be decompressed: "
            f"{self.model_path}"
        )

    def _decompress_model_if_needed(self, models_dir: str, output_path: str) -> bool:
        """Decompress model from split xz parts if available.

        The model is distributed as split xz-compressed parts to fit within
        GitHub's 100MB file size limit. This method joins and decompresses
        the parts on first run.

        Args:
            models_dir: Directory containing the compressed parts
            output_path: Path where the decompressed .pt file should be written

        Returns:
            True if decompression succeeded or file already exists, False otherwise
        """
        import lzma
        import glob

        if os.path.exists(output_path):
            return True

        # Look for split xz parts
        part_pattern = os.path.join(models_dir, 'rfdetr_nano_person.pt.xz.part_*')
        parts = sorted(glob.glob(part_pattern))

        if not parts:
            return False

        print(f"[RFDETRNano] Found {len(parts)} compressed model parts, decompressing...")
        print(f"[RFDETRNano] This is a one-time operation on first run.")

        try:
            # Join parts and decompress
            compressed_data = b''
            for part_path in parts:
                print(f"[RFDETRNano] Reading: {os.path.basename(part_path)}")
                with open(part_path, 'rb') as f:
                    compressed_data += f.read()

            print(f"[RFDETRNano] Decompressing {len(compressed_data) / (1024*1024):.1f} MB...")
            decompressed_data = lzma.decompress(compressed_data)

            print(f"[RFDETRNano] Writing {len(decompressed_data) / (1024*1024):.1f} MB to {output_path}")
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)

            print(f"[RFDETRNano] Model decompression complete!")
            return True

        except Exception as e:
            print(f"[RFDETRNano] Failed to decompress model: {e}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

    def _load_pytorch_model(self):
        """Load the PyTorch RF-DETR model."""
        try:
            from rfdetr import RFDETRNano as RFDETRNanoModel

            # The one path that genuinely needs the checkpoint bytes, so this
            # is where the xz parts are joined if that has not happened yet.
            self._ensure_model_file()

            resolution = self.model_input_size[0]  # Assuming square input
            # trust_checkpoint=True is required from rfdetr's hardened loader
            # onwards: _safe_torch_load defaults to weights_only=True and raises
            # on a checkpoint carrying custom Python objects, which
            # rfdetr_nano_person.pt does. Without it the whole pose stage dies at
            # construction. Trusting it is correct here and nowhere else — the
            # checkpoint is not user input, it ships in this repository as the
            # rfdetr_nano_person.pt.xz.part_* files that _ensure_model_file()
            # joins immediately above.
            self.model = RFDETRNanoModel(
                pretrain_weights=self.model_path,
                resolution=resolution,
                trust_checkpoint=True,
            )
            print(f"Loaded RF-DETR Nano model from {self.model_path}")
        except ImportError:
            raise ImportError(
                "rfdetr package not installed. Please install with: pip install rfdetr"
            )

    def _get_or_build_engine(self) -> Optional[str]:
        """Get existing TensorRT engine or build a new one."""
        # Generate engine path based on model and config
        model_stem = Path(self.model_path).stem
        size_str = f"{self.model_input_size[0]}x{self.model_input_size[1]}"
        # Use BF16 for better dynamic range than FP16
        precision_str = "bf16" if self.device == 'cuda' else "fp32"

        # Include batch size in engine filename for caching different configurations
        engine_name = f"{model_stem}_{size_str}_b{self.batch_size}_cuda_{precision_str}.engine"
        engine_path = str(Path(self.model_path).parent / engine_name)

        print(f"[RFDETRNano] Checking for engine at: {engine_path}")
        print(f"[RFDETRNano] Engine exists: {os.path.exists(engine_path)}")

        # Check if engine already exists
        if os.path.exists(engine_path):
            print(f"TensorRT engine already exists: {engine_path}")
            return engine_path

        # Need to build engine: PyTorch -> ONNX -> TensorRT
        print(f"Building TensorRT engine: {engine_path}")

        # First, export to ONNX
        onnx_path = self._export_to_onnx()
        if onnx_path is None:
            return None

        # Then build TensorRT engine from ONNX
        engine_path = self._build_tensorrt_engine(onnx_path, engine_path)
        return engine_path

    def _export_to_onnx(self) -> Optional[str]:
        """Export RF-DETR model to ONNX format with batch size support.

        Uses RF-DETR fork's batch-capable export API from:
        git+https://github.com/jude-mingay-bps/rf-detr-tensorrt-batch.git
        """
        try:
            # Load PyTorch model if not already loaded
            if self.model is None:
                self._load_pytorch_model()

            model_stem = Path(self.model_path).stem
            # Include batch size in ONNX filename for caching different configurations
            onnx_path = str(Path(self.model_path).parent / f"{model_stem}_b{self.batch_size}.onnx")

            # Check if ONNX already exists
            if os.path.exists(onnx_path):
                print(f"ONNX model already exists: {onnx_path}")
                return onnx_path

            print(f"Exporting to ONNX with batch_size={self.batch_size}: {onnx_path}")

            # Remember current working directory
            cwd = os.getcwd()
            output_dir = Path(onnx_path).parent

            # Use RF-DETR fork's new API with batch_size parameter
            # Note: Don't pass shape parameter - the model uses its configured resolution
            # and RF-DETR validates shape % 14 == 0 (DINOv2 patch size requirement)
            self.model.export(
                output_dir=str(output_dir),
                opset_version=17,  # Enables INormalizationLayer for FP16/BF16
                batch_size=self.batch_size,
            )

            # RF-DETR exports to 'output/inference_model.onnx' or similar
            # regardless of output_dir parameter - need to find and rename
            if not os.path.exists(onnx_path):
                # Check common RF-DETR export locations
                possible_paths = [
                    os.path.join(str(output_dir), "inference_model.onnx"),
                    os.path.join(cwd, "output", "inference_model.onnx"),
                    os.path.join(cwd, "inference_model.onnx"),
                    "output/inference_model.onnx",
                    "inference_model.onnx",
                ]
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        print(f"Found exported ONNX at: {possible_path}, moving to: {onnx_path}")
                        shutil.move(possible_path, onnx_path)
                        # Clean up empty output directory if created
                        parent_dir = os.path.dirname(possible_path)
                        if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                        break

            if os.path.exists(onnx_path):
                print(f"Successfully exported to ONNX: {onnx_path}")
                return onnx_path
            else:
                print("ONNX export failed - output file not found")
                print(f"Checked: {possible_paths}")
                return None

        except Exception as e:
            print(f"Failed to export to ONNX: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str) -> Optional[str]:
        """Build TensorRT engine from ONNX model with BF16 precision."""
        try:
            import tensorrt as trt

            print(f"Building TensorRT engine from {onnx_path}")

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
                        print(f"ONNX parse error: {parser.get_error(i)}")
                    return None

            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4GB

            # Enable BF16 for CUDA (better dynamic range than FP16)
            # Fall back to FP16 if BF16 not available
            if self.device == 'cuda':
                if hasattr(trt.BuilderFlag, 'BF16') and builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.BF16)
                    print("[RFDETRNano] Using BF16 precision")
                elif builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("[RFDETRNano] BF16 not available, using FP16 precision")

            # Set optimization profile for static batch size
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            input_name = input_tensor.name

            # Use configured batch size (ONNX should already have correct shape from export)
            onnx_shape = input_tensor.shape
            batch_size = onnx_shape[0] if onnx_shape[0] > 0 else self.batch_size
            resolution = self.model_input_size[0]

            # Validate ONNX batch size matches configuration
            if batch_size != self.batch_size:
                print(f"[RFDETRNano] Warning: ONNX batch_size={batch_size} differs from config batch_size={self.batch_size}")
                print(f"[RFDETRNano] Using ONNX batch_size={batch_size}")

            input_shape = (batch_size, 3, resolution, resolution)
            profile.set_shape(input_name, input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

            # Build engine
            print("Building serialized network (this may take a few minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                print("Failed to build TensorRT engine")
                return None

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            print(f"Successfully built TensorRT engine: {engine_path}")
            return engine_path

        except ImportError:
            print("TensorRT not available. Please install tensorrt package.")
            return None
        except Exception as e:
            print(f"Failed to build TensorRT engine: {e}")
            return None

    def _load_tensorrt_engine(self):
        """Load TensorRT engine for inference."""
        try:
            import tensorrt as trt
            import torch

            print(f"Loading TensorRT engine: {self.engine_path}")

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(self.engine_path, "rb") as f:
                engine_data = f.read()

            self._engine = runtime.deserialize_cuda_engine(engine_data)
            self._context = self._engine.create_execution_context()

            # Allocate buffers
            self._allocate_buffers()

            # Create CUDA stream
            self._stream = torch.cuda.Stream()

            # Pre-compute normalization constants (avoid allocation each inference)
            self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
            self._norm_std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

            # Set up CUDA graph for faster kernel replay
            self._setup_cuda_graph()

            print("TensorRT engine loaded successfully")

        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            self._engine = None
            # Fallback to PyTorch
            self.backend = 'pytorch'
            self._load_pytorch_model()

    def _allocate_buffers(self):
        """Allocate GPU buffers for TensorRT inference."""
        import torch
        import tensorrt as trt

        self._input_tensors = {}
        self._output_tensors = {}

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            shape = tuple(self._engine.get_tensor_shape(name))

            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            else:
                torch_dtype = torch.float32

            # Allocate contiguous GPU tensor
            tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_tensors[name] = tensor
                self._context.set_tensor_address(name, tensor.data_ptr())
            else:
                self._output_tensors[name] = tensor
                self._context.set_tensor_address(name, tensor.data_ptr())

        # Store input tensor name
        self._input_name = list(self._input_tensors.keys())[0]

        # Find output tensors by name pattern (like reference implementation)
        self._boxes_name = None
        self._scores_name = None
        for name in self._output_tensors.keys():
            name_lower = name.lower()
            if "det" in name_lower or "box" in name_lower:
                self._boxes_name = name
            elif "label" in name_lower or "score" in name_lower or "class" in name_lower:
                self._scores_name = name

        # Fallback to index order if names don't match patterns
        output_names = list(self._output_tensors.keys())
        if self._boxes_name is None and len(output_names) > 0:
            self._boxes_name = output_names[0]
        if self._scores_name is None and len(output_names) > 1:
            self._scores_name = output_names[1]

        print(f"TensorRT outputs - boxes: {self._boxes_name}, scores: {self._scores_name}")

    def _setup_cuda_graph(self):
        """Set up CUDA graph for kernel replay optimization.

        CUDA Graphs capture the entire inference kernel sequence and replay it
        with minimal CPU overhead, providing ~10-20% performance gains.
        """
        import torch

        if not self.use_cuda_graphs:
            print("[RFDETRNano] CUDA graphs disabled by configuration")
            return

        try:
            print("[RFDETRNano] Warming up for CUDA graph capture...")

            # Warm-up runs to stabilize kernels (required before capture)
            for _ in range(10):
                self._context.execute_async_v3(
                    stream_handle=self._stream.cuda_stream
                )
                self._stream.synchronize()

            # Capture CUDA graph
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph, stream=self._stream):
                self._context.execute_async_v3(
                    stream_handle=self._stream.cuda_stream
                )

            self._graph_captured = True
            print("[RFDETRNano] CUDA graph captured successfully")

        except Exception as e:
            print(f"[RFDETRNano] CUDA graph capture failed: {e}, using standard execution")
            self._cuda_graph = None
            self._graph_captured = False

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run detection on input image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Filtered person detections in format [[x1, y1, x2, y2, score], ...].
        """
        if self.backend == 'pytorch' or self._engine is None:
            return self._inference_pytorch(image)
        else:
            return self._inference_tensorrt(image)

    def _dedup_and_filter_detections(self, detections: np.ndarray) -> np.ndarray:
        """Drop degenerate and near-duplicate detections.

        Applies a minimum-area sanity filter and a conservative IoU-based
        dedup: any box with IoU > _DEDUP_IOU_THR against a higher-scoring
        box is dropped (greedy, score-descending).

        Args:
            detections: Array of shape (N, 5) in [x1, y1, x2, y2, score] format.

        Returns:
            Filtered detections sorted by score (descending), shape (M, 5).
        """
        if len(detections) == 0:
            return detections

        widths = np.maximum(detections[:, 2] - detections[:, 0], 0.0)
        heights = np.maximum(detections[:, 3] - detections[:, 1], 0.0)
        areas = widths * heights
        detections = detections[areas >= self._MIN_BOX_AREA]

        if len(detections) <= 1:
            return detections

        # Greedy dedup in score-descending order
        order = np.argsort(-detections[:, 4], kind='stable')
        detections = detections[order]
        x1, y1 = detections[:, 0], detections[:, 1]
        x2, y2 = detections[:, 2], detections[:, 3]
        areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)

        keep = []
        for i in range(len(detections)):
            duplicate = False
            for j in keep:
                ix1 = max(x1[i], x1[j])
                iy1 = max(y1[i], y1[j])
                ix2 = min(x2[i], x2[j])
                iy2 = min(y2[i], y2[j])
                inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
                union = areas[i] + areas[j] - inter
                if union > 0 and inter / union > self._DEDUP_IOU_THR:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(i)

        return detections[keep]

    def _inference_pytorch(self, image: np.ndarray) -> np.ndarray:
        """Run inference using PyTorch model."""
        if self.model is None:
            self._load_pytorch_model()

        # Convert BGR to RGB
        import cv2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        detections = self.model.predict(image_rgb, threshold=self.score_thr)

        # Filter for person class and convert to expected format
        return self._filter_person_detections(detections)

    def _inference_tensorrt(self, image: np.ndarray) -> np.ndarray:
        """Run inference using TensorRT engine."""
        import torch
        import torch.nn.functional as F

        # Get original image size for coordinate scaling
        orig_h, orig_w = image.shape[:2]
        resolution = self.model_input_size[0]

        # Preprocess image - upload BGR directly, convert on GPU
        # Check contiguous to avoid unnecessary copy
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # Convert to tensor and move to GPU (still BGR)
        img_tensor = torch.from_numpy(image).cuda(non_blocking=True)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

        # BGR→RGB channel swap on GPU (faster than CPU cv2.cvtColor)
        img_tensor = img_tensor[:, [2, 1, 0], :, :]

        # Resize to model input size
        if img_tensor.shape[2] != resolution or img_tensor.shape[3] != resolution:
            img_tensor = F.interpolate(
                img_tensor,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet normalization (using pre-computed constants)
        img_tensor = img_tensor.mul_(1.0 / 255.0)
        img_tensor = img_tensor.sub_(self._norm_mean).div_(self._norm_std)

        # Copy input data
        self._input_tensors[self._input_name].copy_(img_tensor)

        # Execute inference (use CUDA graph if available for ~10-20% speedup)
        if self._cuda_graph is not None and self._graph_captured:
            self._cuda_graph.replay()
            # No explicit sync - implicit sync happens on .cpu() transfer
        else:
            # Make the private stream wait for the input copy issued on the
            # current stream, then sync so outputs are complete before reads.
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            self._stream.synchronize()  # Only sync if not using CUDA graph

        # Parse outputs using name-based lookup
        boxes = self._output_tensors[self._boxes_name]
        scores = self._output_tensors[self._scores_name] if self._scores_name else None

        return self._postprocess_tensorrt(boxes, scores, orig_w, orig_h)

    def _postprocess_tensorrt(
        self, boxes, scores, orig_w: int, orig_h: int
    ) -> np.ndarray:
        """Postprocess TensorRT outputs.

        Returns:
            Detections in [x1, y1, x2, y2, score] format, shape (N, 5).
        """
        import torch

        # Apply sigmoid to scores
        if scores is not None:
            scores = torch.sigmoid(scores)
            # Get max score and class per box
            max_scores, class_ids = scores.max(dim=2)
            max_scores = max_scores[0]  # Remove batch dimension
            class_ids = class_ids[0]
        else:
            max_scores = torch.ones(boxes.shape[1], device="cuda")
            class_ids = torch.zeros(boxes.shape[1], device="cuda", dtype=torch.int64)

        boxes = boxes[0]  # Remove batch dimension (300, 4)

        # Filter by confidence and person class
        valid_mask = (max_scores >= self.score_thr) & (class_ids == self.PERSON_CLASS)

        if not valid_mask.any():
            return self._EMPTY_BOXES.copy()

        # Get valid boxes and convert from cxcywh to xyxy
        valid_boxes = boxes[valid_mask]
        valid_scores = max_scores[valid_mask]
        cx, cy, w, h = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]

        # Scale to original image size (coordinates are normalized 0-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h

        # Stack [x1, y1, x2, y2, score] and convert to numpy
        detections = torch.stack([x1, y1, x2, y2, valid_scores], dim=1)
        return self._dedup_and_filter_detections(detections.cpu().numpy())

    def _filter_person_detections(self, detections) -> np.ndarray:
        """Filter RF-DETR detections to only include person class.

        Args:
            detections: Detection results from RF-DETR model.

        Returns:
            Filtered person detections in format [[x1, y1, x2, y2, score], ...].
        """
        if detections is None:
            return self._EMPTY_BOXES.copy()

        # Handle RF-DETR detection format
        if hasattr(detections, "xyxy") and len(detections.xyxy) > 0:
            boxes = detections.xyxy
            confidences = detections.confidence
            class_ids = detections.class_id

            # Filter for person class and confidence threshold
            person_mask = (class_ids == self.PERSON_CLASS) & (confidences >= self.score_thr)

            if not person_mask.any():
                return self._EMPTY_BOXES.copy()

            # Extract person boxes and scores
            person_boxes = boxes[person_mask]
            person_scores = confidences[person_mask]

            # Convert to numpy if needed
            if hasattr(person_boxes, 'cpu'):
                person_boxes = person_boxes.cpu().numpy()
            elif not isinstance(person_boxes, np.ndarray):
                person_boxes = np.array(person_boxes)
            if hasattr(person_scores, 'cpu'):
                person_scores = person_scores.cpu().numpy()
            elif not isinstance(person_scores, np.ndarray):
                person_scores = np.array(person_scores)

            person_detections = np.concatenate(
                [person_boxes.astype(np.float32),
                 person_scores.astype(np.float32).reshape(-1, 1)],
                axis=1,
            )
            return self._dedup_and_filter_detections(person_detections)

        return self._EMPTY_BOXES.copy()

    # ============================================================
    # Batch Processing Methods (for high-throughput inference)
    # ============================================================

    def predict_batch(
        self,
        frames: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Run batch detection on multiple frames.

        This method is optimized for processing multiple frames simultaneously
        using TensorRT with CUDA Graphs for maximum throughput.

        Args:
            frames: List of input images as numpy arrays (BGR format from OpenCV).
            threshold: Optional confidence threshold override.

        Returns:
            List of detection arrays, one per frame. Each array has shape (N, 5)
            with format [[x1, y1, x2, y2, score], ...] for person detections.
        """
        if self.backend == 'pytorch' or self._engine is None:
            # Fallback to sequential for non-TensorRT
            return [self.__call__(frame) for frame in frames]

        return self._inference_tensorrt_batch(frames, threshold)

    def _preprocess_batch_for_tensorrt(
        self, frames: List[np.ndarray]
    ) -> "torch.Tensor":
        """Preprocess frames for TensorRT inference using GPU.

        Optimized process (3.5x faster than numpy.stack approach):
        1. Use pre-allocated GPU buffer for fast CPU->GPU transfer
        2. Transfer frames directly without numpy.stack
        3. BGR→RGB conversion on GPU (faster than CPU cv2.cvtColor)
        4. Resize + normalize on GPU

        Args:
            frames: List of BGR frames as numpy arrays (from OpenCV)

        Returns:
            Preprocessed batch as GPU tensor (batch, 3, H, W) in RGB format
        """
        import torch
        import torch.nn.functional as F

        resolution = self.model_input_size[0]
        batch_size = len(frames)
        frame_h, frame_w = frames[0].shape[:2]

        # Use pre-allocated GPU buffer (avoid allocation per batch)
        buffer_key = (batch_size, frame_h, frame_w)
        if buffer_key not in self._preprocess_buffers:
            self._preprocess_buffers[buffer_key] = torch.empty(
                (batch_size, frame_h, frame_w, 3),
                dtype=torch.uint8,
                device='cuda'
            )
            print(f"[RFDETRNano] Allocated preprocess buffer: {buffer_key}")

        gpu_buffer = self._preprocess_buffers[buffer_key]

        # Fast path: copy frames directly to GPU (avoids numpy.stack bottleneck).
        # A frame that is already a CUDA tensor -- which is what an NVDEC
        # decode loop holds -- is copied device to device, so the caller does
        # not have to stage it through host memory to hand it back.
        for i, frame in enumerate(frames):
            if isinstance(frame, torch.Tensor):
                gpu_buffer[i].copy_(frame, non_blocking=True)
                continue
            # Check contiguous to avoid unnecessary copy
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            gpu_buffer[i].copy_(
                torch.from_numpy(frame),
                non_blocking=True
            )

        # Convert to float and reorder dimensions on GPU: (B, H, W, C) -> (B, C, H, W)
        batch_tensor = gpu_buffer[:batch_size].permute(0, 3, 1, 2).float()

        # BGR→RGB channel swap on GPU (faster than CPU cv2.cvtColor)
        batch_tensor = batch_tensor[:, [2, 1, 0], :, :]

        # Resize on GPU if needed
        if frame_h != resolution or frame_w != resolution:
            batch_tensor = F.interpolate(
                batch_tensor,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )

        # Fused normalize on GPU: [0-255] -> [0-1] -> ImageNet normalized
        batch_tensor = batch_tensor.mul_(1.0 / 255.0)
        batch_tensor = batch_tensor.sub_(self._norm_mean).div_(self._norm_std)

        return batch_tensor

    def _pad_batch(self, batch: "torch.Tensor", target_size: int) -> "torch.Tensor":
        """Pad batch to target size using pre-allocated buffer.

        Args:
            batch: Input GPU tensor
            target_size: Target batch size

        Returns:
            Padded batch tensor
        """
        import torch

        current_size = batch.shape[0]
        if current_size >= target_size:
            return batch

        # Use pre-allocated buffer for efficiency
        buffer_key = (target_size, tuple(batch.shape[1:]), batch.dtype)
        if buffer_key not in self._pad_buffers:
            buffer_shape = (target_size,) + tuple(batch.shape[1:])
            self._pad_buffers[buffer_key] = torch.zeros(
                buffer_shape, dtype=batch.dtype, device='cuda'
            )
            print(f"[RFDETRNano] Allocated padding buffer: {buffer_shape}")

        buffer = self._pad_buffers[buffer_key]
        buffer[:current_size].copy_(batch)
        buffer[current_size:].zero_()
        return buffer

    def _inference_tensorrt_batch(
        self,
        frames: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Run batch inference using TensorRT engine with CUDA Graphs.

        Key optimizations:
        - CUDA Graph replay (no explicit sync needed)
        - Zero-copy views with narrow()
        - Chunked processing for batches > engine batch size

        Args:
            frames: List of BGR images
            threshold: Confidence threshold

        Returns:
            List of detection arrays per frame
        """
        import torch

        if threshold is None:
            threshold = self.score_thr

        # Get original sizes for coordinate scaling
        frame_sizes = [(f.shape[1], f.shape[0]) for f in frames]

        # Frames stay as BGR - channel swap happens on GPU in _preprocess_batch_for_tensorrt
        actual_batch = len(frames)
        engine_batch = self.batch_size

        all_results = []

        # Process in chunks matching engine batch size
        for chunk_start in range(0, actual_batch, engine_batch):
            chunk_end = min(chunk_start + engine_batch, actual_batch)
            chunk_frames = frames[chunk_start:chunk_end]
            chunk_sizes = frame_sizes[chunk_start:chunk_end]

            # Preprocess on GPU
            preprocessed = self._preprocess_batch_for_tensorrt(chunk_frames)

            # Pad if needed
            if len(chunk_frames) < engine_batch:
                preprocessed = self._pad_batch(preprocessed, engine_batch)

            # Copy to TensorRT input buffer
            self._input_tensors[self._input_name].copy_(preprocessed)

            # Execute inference (use CUDA graph if available)
            if self._cuda_graph is not None and self._graph_captured:
                self._cuda_graph.replay()
                # No explicit sync - implicit sync happens on .cpu() transfer
            else:
                # Make the private stream wait for the input copy issued on
                # the current stream, then sync so outputs are complete
                # before postprocessing reads them.
                self._stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._stream):
                    self._context.execute_async_v3(
                        stream_handle=self._stream.cuda_stream
                    )
                self._stream.synchronize()

            # Parse outputs using zero-copy views (narrow())
            actual_count = len(chunk_frames)
            boxes = self._output_tensors[self._boxes_name].narrow(0, 0, actual_count)
            scores = (
                self._output_tensors[self._scores_name].narrow(0, 0, actual_count)
                if self._scores_name
                else None
            )

            # Postprocess on GPU, then transfer
            chunk_results = self._postprocess_tensorrt_batch(
                boxes, scores, chunk_sizes, threshold
            )
            all_results.extend(chunk_results)

        return all_results

    def _postprocess_tensorrt_batch(
        self,
        boxes: "torch.Tensor",
        scores: "torch.Tensor",
        frame_sizes: List[Tuple[int, int]],
        threshold: float,
    ) -> List[np.ndarray]:
        """Postprocess TensorRT outputs with GPU acceleration.

        Key optimizations:
        - GPU filtering before CPU transfer
        - Vectorized coordinate conversion
        - Single batched CPU transfer

        Args:
            boxes: Shape (batch, 300, 4) in cxcywh format, normalized (0-1)
            scores: Shape (batch, 300, num_classes) as logits
            frame_sizes: List of (width, height) for each frame
            threshold: Confidence threshold

        Returns:
            List of detection arrays per frame, each with shape (N, 5)
            in [x1, y1, x2, y2, score] format
        """
        import torch

        num_frames = len(frame_sizes)

        # Apply sigmoid to all scores at once (GPU)
        if scores is not None:
            scores = torch.sigmoid(scores)
            max_scores, class_ids = scores.max(dim=2)
        else:
            max_scores = torch.ones(boxes.shape[0], boxes.shape[1], device="cuda")
            class_ids = torch.zeros(
                boxes.shape[0], boxes.shape[1], device="cuda", dtype=torch.int64
            )

        # Filter on GPU before CPU transfer - only transfer valid detections
        valid_mask = (max_scores >= threshold) & (class_ids == self.PERSON_CLASS)

        # Create frame sizes tensor for vectorized scaling
        frame_sizes_t = torch.as_tensor(frame_sizes, device="cuda", dtype=torch.float32)
        orig_w = frame_sizes_t[:, 0:1]  # (num_frames, 1)
        orig_h = frame_sizes_t[:, 1:2]  # (num_frames, 1)

        # Convert boxes from cxcywh to xyxy format - fused operations
        half_w = boxes[:, :, 2] * 0.5
        half_h = boxes[:, :, 3] * 0.5
        cx = boxes[:, :, 0]
        cy = boxes[:, :, 1]

        # Scale and stack - vectorized for all frames
        boxes_xyxy = torch.stack(
            [
                (cx - half_w) * orig_w,
                (cy - half_h) * orig_h,
                (cx + half_w) * orig_w,
                (cy + half_h) * orig_h,
            ],
            dim=2,
        )  # (num_frames, 300, 4)

        # Single batched CPU transfer - .cpu() implicitly syncs with GPU
        boxes_cpu = boxes_xyxy.cpu().numpy()
        scores_cpu = max_scores.cpu().numpy()
        valid_cpu = valid_mask.cpu().numpy()

        # Create results per frame using pre-computed GPU filter mask
        results = []
        for batch_idx in range(num_frames):
            valid_indices = np.nonzero(valid_cpu[batch_idx])[0]
            if len(valid_indices) == 0:
                results.append(self._EMPTY_BOXES)
            else:
                frame_detections = np.concatenate(
                    [
                        boxes_cpu[batch_idx, valid_indices].astype(np.float32),
                        scores_cpu[batch_idx, valid_indices]
                        .astype(np.float32)
                        .reshape(-1, 1),
                    ],
                    axis=1,
                )
                results.append(self._dedup_and_filter_detections(frame_detections))

        return results
