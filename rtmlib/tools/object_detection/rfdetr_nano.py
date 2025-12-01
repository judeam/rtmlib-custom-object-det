"""RFDETRNano object detector with person filtering and TensorRT support."""

from typing import Optional
import numpy as np
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

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_input_size: tuple = (384, 384),
        score_thr: float = 0.5,
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        export_format: str = 'engine',
    ):
        """Initialize RFDETRNano detector.

        Args:
            model_path: Path to model file. If None, will use default model.
            model_input_size: Input size for the model (height, width).
            score_thr: Score threshold for filtering detections.
            backend: Backend for inference ('onnxruntime', 'pytorch', or 'tensorrt').
            device: Device for inference ('cpu' or 'cuda').
            export_format: Format for export ('engine' for TensorRT, 'onnx' for ONNX).
        """
        self.model_input_size = model_input_size
        self.score_thr = score_thr
        self.backend = backend
        self.device = device
        self.export_format = export_format
        self.model = None
        self.engine_path = None
        self._engine = None

        # Resolve model path
        if model_path is None:
            model_path = self._resolve_model_path()

        self.model_path = model_path

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

    def _resolve_model_path(self) -> str:
        """Resolve the default model path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Try in tools/models directory
        tools_models_path = os.path.join(
            current_dir, '..', 'models', 'rfdetr_nano_person.pt'
        )
        tools_models_path = os.path.abspath(tools_models_path)

        # Try relative to project root (for development)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        )
        dev_model_path = os.path.join(project_root, 'models', 'rfdetr_nano_person.pt')

        # Choose the path that exists
        if os.path.exists(tools_models_path):
            return tools_models_path
        elif os.path.exists(dev_model_path):
            return dev_model_path
        else:
            # Fallback: create models directory
            models_dir = os.path.join(os.path.dirname(current_dir), 'models')
            os.makedirs(models_dir, exist_ok=True)
            expected_path = os.path.join(models_dir, 'rfdetr_nano_person.pt')
            raise FileNotFoundError(
                f"Model file not found. Please place 'rfdetr_nano_person.pt' in: {models_dir}"
            )

    def _load_pytorch_model(self):
        """Load the PyTorch RF-DETR model."""
        try:
            from rfdetr import RFDETRNano as RFDETRNanoModel

            resolution = self.model_input_size[0]  # Assuming square input
            self.model = RFDETRNanoModel(
                pretrain_weights=self.model_path,
                resolution=resolution
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
        precision_str = "fp16" if self.device == 'cuda' else "fp32"

        engine_name = f"{model_stem}_{size_str}_cuda_{precision_str}.engine"
        engine_path = str(Path(self.model_path).parent / engine_name)

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
        """Export RF-DETR model to ONNX format."""
        try:
            # Load PyTorch model if not already loaded
            if self.model is None:
                self._load_pytorch_model()

            model_stem = Path(self.model_path).stem
            onnx_path = str(Path(self.model_path).parent / f"{model_stem}.onnx")

            # Check if ONNX already exists
            if os.path.exists(onnx_path):
                print(f"ONNX model already exists: {onnx_path}")
                return onnx_path

            print(f"Exporting to ONNX: {onnx_path}")

            # Use RF-DETR's built-in export method
            # dynamic=False is critical for TensorRT CUDA Graphs
            self.model.export(
                format="onnx",
                output_path=onnx_path,
                dynamic=False,
            )

            # RF-DETR may export to a different path, handle this
            if not os.path.exists(onnx_path):
                # Check common RF-DETR export locations
                possible_paths = [
                    "output/inference_model.onnx",
                    "inference_model.onnx",
                ]
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        shutil.move(possible_path, onnx_path)
                        break

            if os.path.exists(onnx_path):
                print(f"Successfully exported to ONNX: {onnx_path}")
                return onnx_path
            else:
                print("ONNX export failed - output file not found")
                return None

        except Exception as e:
            print(f"Failed to export to ONNX: {e}")
            return None

    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str) -> Optional[str]:
        """Build TensorRT engine from ONNX model."""
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

            # Enable FP16 for CUDA
            if self.device == 'cuda' and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # Set optimization profile for static batch size
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            input_name = input_tensor.name

            # Get shape from ONNX or use config
            onnx_shape = input_tensor.shape
            batch_size = onnx_shape[0] if onnx_shape[0] > 0 else 1
            resolution = self.model_input_size[0]

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

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run detection on input image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Filtered bounding boxes for person class in format [[x1, y1, x2, y2], ...].
        """
        if self.backend == 'pytorch' or self._engine is None:
            return self._inference_pytorch(image)
        else:
            return self._inference_tensorrt(image)

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
        import cv2

        # Get original image size for coordinate scaling
        orig_h, orig_w = image.shape[:2]
        resolution = self.model_input_size[0]

        # Preprocess image
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor and move to GPU
        img_tensor = torch.from_numpy(image_rgb).cuda(non_blocking=True)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

        # Resize to model input size
        if img_tensor.shape[2] != resolution or img_tensor.shape[3] != resolution:
            img_tensor = F.interpolate(
                img_tensor,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        img_tensor = img_tensor.mul_(1.0 / 255.0)
        img_tensor = img_tensor.sub_(mean).div_(std)

        # Copy input data
        self._input_tensors[self._input_name].copy_(img_tensor)

        # Execute inference
        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        torch.cuda.synchronize()

        # Parse outputs using name-based lookup
        boxes = self._output_tensors[self._boxes_name]
        scores = self._output_tensors[self._scores_name] if self._scores_name else None

        return self._postprocess_tensorrt(boxes, scores, orig_w, orig_h)

    def _postprocess_tensorrt(
        self, boxes, scores, orig_w: int, orig_h: int
    ) -> np.ndarray:
        """Postprocess TensorRT outputs."""
        import torch

        # Debug: Print shapes on first call
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] boxes shape: {boxes.shape}, scores shape: {scores.shape if scores is not None else None}")
            print(f"[DEBUG] boxes sample: {boxes[0, :3, :]}")
            if scores is not None:
                print(f"[DEBUG] scores sample (pre-sigmoid): {scores[0, :3, :]}")
            self._debug_printed = True

        # Apply sigmoid to scores
        if scores is not None:
            scores = torch.sigmoid(scores)
            # Get max score and class per box
            max_scores, class_ids = scores.max(dim=2)
            max_scores = max_scores[0]  # Remove batch dimension
            class_ids = class_ids[0]

            # Debug: Print score stats on first call
            if not hasattr(self, '_debug_scores_printed'):
                top_scores, top_indices = max_scores.topk(min(10, len(max_scores)))
                top_classes = class_ids[top_indices]
                print(f"[DEBUG] Top 10 scores (post-sigmoid): {top_scores.cpu().numpy()}")
                print(f"[DEBUG] Top 10 class IDs: {top_classes.cpu().numpy()}")
                print(f"[DEBUG] Score threshold: {self.score_thr}, Person class: {self.PERSON_CLASS}")
                self._debug_scores_printed = True
        else:
            max_scores = torch.ones(boxes.shape[1], device="cuda")
            class_ids = torch.zeros(boxes.shape[1], device="cuda", dtype=torch.int64)

        boxes = boxes[0]  # Remove batch dimension (300, 4)

        # Filter by confidence and person class
        valid_mask = (max_scores >= self.score_thr) & (class_ids == self.PERSON_CLASS)

        if not valid_mask.any():
            return np.array([]).reshape(0, 4)

        # Get valid boxes and convert from cxcywh to xyxy
        valid_boxes = boxes[valid_mask]
        cx, cy, w, h = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]

        # Scale to original image size (coordinates are normalized 0-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h

        # Stack and convert to numpy
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes_xyxy.cpu().numpy()

    def _filter_person_detections(self, detections) -> np.ndarray:
        """Filter RF-DETR detections to only include person class.

        Args:
            detections: Detection results from RF-DETR model.

        Returns:
            Filtered bounding boxes for person class in format [[x1, y1, x2, y2], ...].
        """
        if detections is None:
            return np.array([]).reshape(0, 4)

        # Handle RF-DETR detection format
        if hasattr(detections, "xyxy") and len(detections.xyxy) > 0:
            boxes = detections.xyxy
            confidences = detections.confidence
            class_ids = detections.class_id

            # Filter for person class and confidence threshold
            person_mask = (class_ids == self.PERSON_CLASS) & (confidences >= self.score_thr)

            if not person_mask.any():
                return np.array([]).reshape(0, 4)

            # Extract person boxes
            person_boxes = boxes[person_mask]

            # Convert to numpy if needed
            if hasattr(person_boxes, 'cpu'):
                person_boxes = person_boxes.cpu().numpy()
            elif not isinstance(person_boxes, np.ndarray):
                person_boxes = np.array(person_boxes)

            return person_boxes

        return np.array([]).reshape(0, 4)
