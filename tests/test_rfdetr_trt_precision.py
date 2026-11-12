"""The person detector's TensorRT precision is a choice, and bf16 stays the default.

bf16 is what the deployed engines were built at. fp16 is faster on the GPUs
measured and closer to fp32, but the boxes are not the same numbers, so it is
opt-in: an argument, or RFDETR_NANO_TRT_PRECISION for a caller that builds the
detector through one of the solutions. Each precision has its own engine file.
"""

import sys
import types

import pytest

from rtmlib.tools.object_detection.rfdetr_nano import RFDETRNano

ENV = "RFDETR_NANO_TRT_PRECISION"


def _detector(precision=None, device="cuda", batch_size=16):
    det = RFDETRNano.__new__(RFDETRNano)
    det.model_path = "/models/rfdetr_nano_person.pt"
    det.model_input_size = (384, 384)
    det.batch_size = batch_size
    det.device = device
    if precision is not None:
        det.trt_precision = precision
    return det


class TestResolve:
    def test_default_is_bf16(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        assert RFDETRNano._resolve_trt_precision(None) == "bf16"

    @pytest.mark.parametrize("value, expected", [
        ("fp16", "fp16"), ("FP16", "fp16"), (" fp32 ", "fp32"), ("bf16", "bf16"), ("", "bf16"),
    ])
    def test_environment_chooses(self, monkeypatch, value, expected):
        monkeypatch.setenv(ENV, value)
        assert RFDETRNano._resolve_trt_precision(None) == expected

    def test_argument_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv(ENV, "fp16")
        assert RFDETRNano._resolve_trt_precision("fp32") == "fp32"

    @pytest.mark.parametrize("value", ["int8", "half", "fp-16"])
    def test_unknown_precision_is_an_error(self, monkeypatch, value):
        monkeypatch.setenv(ENV, value)
        with pytest.raises(ValueError, match="bf16, fp16, fp32"):
            RFDETRNano._resolve_trt_precision(None)


class TestEngineName:
    def test_default_name_is_the_deployed_one(self):
        assert _detector()._engine_name() == "rfdetr_nano_person_384x384_b16_cuda_bf16.engine"

    @pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
    def test_each_precision_has_its_own_file(self, precision):
        name = _detector(precision)._engine_name()
        assert name == f"rfdetr_nano_person_384x384_b16_cuda_{precision}.engine"

    def test_cpu_is_fp32_whatever_was_asked(self):
        assert _detector("fp16", device="cpu")._engine_name().endswith("_cuda_fp32.engine")


class _Config:
    def __init__(self):
        self.flags = []

    def set_flag(self, flag):
        self.flags.append(flag)

    def set_memory_pool_limit(self, *args):
        pass

    def add_optimization_profile(self, profile):
        pass


class _Logger:
    WARNING = 0

    def __init__(self, level):
        pass


def _fake_tensorrt(monkeypatch, config, has_bf16=True, fast_fp16=True):
    """A tensorrt module that records the builder flags and builds nothing."""
    flags = {"FP16": "FP16"}
    if has_bf16:
        flags["BF16"] = "BF16"
    builder = types.SimpleNamespace(
        platform_has_fast_fp16=fast_fp16,
        create_network=lambda flags: types.SimpleNamespace(
            get_input=lambda i: types.SimpleNamespace(name="input", shape=(16, 3, 384, 384))),
        create_builder_config=lambda: config,
        create_optimization_profile=lambda: types.SimpleNamespace(set_shape=lambda *a: None),
        build_serialized_network=lambda network, cfg: None,
    )
    trt = types.SimpleNamespace(
        Logger=_Logger,
        Builder=lambda logger: builder,
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0),
        OnnxParser=lambda network, logger: types.SimpleNamespace(parse=lambda data: True),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE=0),
        BuilderFlag=types.SimpleNamespace(**flags),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)


class TestBuilderFlags:
    def _build(self, monkeypatch, tmp_path, precision, **trt):
        config = _Config()
        _fake_tensorrt(monkeypatch, config, **trt)
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"onnx")
        det = _detector(precision)
        assert det._build_tensorrt_engine(str(onnx), str(tmp_path / "out.engine")) is None
        return config.flags

    def test_default_sets_bf16_only(self, monkeypatch, tmp_path):
        assert self._build(monkeypatch, tmp_path, None) == ["BF16"]

    def test_fp16_sets_fp16_only(self, monkeypatch, tmp_path):
        assert self._build(monkeypatch, tmp_path, "fp16") == ["FP16"]

    def test_fp32_sets_no_flag(self, monkeypatch, tmp_path):
        assert self._build(monkeypatch, tmp_path, "fp32") == []

    def test_bf16_falls_back_to_fp16_where_tensorrt_has_none(self, monkeypatch, tmp_path):
        assert self._build(monkeypatch, tmp_path, "bf16", has_bf16=False) == ["FP16"]

    def test_no_reduced_precision_without_fast_fp16(self, monkeypatch, tmp_path):
        assert self._build(monkeypatch, tmp_path, "fp16", fast_fp16=False) == []


class TestExistingOnnx:
    def test_an_existing_onnx_is_used_without_loading_the_checkpoint(self, tmp_path):
        """Building an engine at another precision starts from the ONNX already exported."""
        onnx = tmp_path / "rfdetr_nano_person_b16.onnx"
        onnx.write_bytes(b"onnx")
        det = _detector("fp16")
        det.model_path = str(tmp_path / "rfdetr_nano_person.pt")
        det.model = None

        def no_load():
            raise AssertionError("the checkpoint was loaded")

        det._load_pytorch_model = no_load
        assert det._export_to_onnx() == str(onnx)
