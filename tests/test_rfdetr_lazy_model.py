"""The checkpoint must not be decompressed to construct a detector.

Engine filenames are derived from the checkpoint's stem and parent, so
construction needs the path, not the bytes. On the TensorRT path a cached
engine is loaded and the checkpoint is never read at all -- decompressing it
during __init__ cost 9.1s of every cold Cloud Run start to produce a file
nothing opened.
"""

import lzma
import os

import pytest

from rtmlib.tools.object_detection.rfdetr_nano import RFDETRNano


def _detector(models_dir):
    """An RFDETRNano with only the path machinery wired up."""
    det = RFDETRNano.__new__(RFDETRNano)
    det._candidate_model_paths = lambda: [
        (str(models_dir), os.path.join(str(models_dir), "rfdetr_nano_person.pt"))
    ]
    return det


def _write_parts(models_dir, payload=b"weights" * 1000):
    blob = lzma.compress(payload)
    half = len(blob) // 2
    (models_dir / "rfdetr_nano_person.pt.xz.part_aa").write_bytes(blob[:half])
    (models_dir / "rfdetr_nano_person.pt.xz.part_ab").write_bytes(blob[half:])
    return payload


class TestResolveDoesNotDecompress:
    def test_parts_present_resolves_without_writing_the_checkpoint(self, tmp_path):
        _write_parts(tmp_path)
        det = _detector(tmp_path)

        path = det._resolve_model_path()

        assert path == os.path.join(str(tmp_path), "rfdetr_nano_person.pt")
        # The whole point: naming it must not materialise it.
        assert not os.path.exists(path)

    def test_existing_checkpoint_is_returned_as_is(self, tmp_path):
        target = tmp_path / "rfdetr_nano_person.pt"
        target.write_bytes(b"already here")
        det = _detector(tmp_path)

        assert det._resolve_model_path() == str(target)

    def test_neither_checkpoint_nor_parts_raises(self, tmp_path):
        det = _detector(tmp_path)

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            det._resolve_model_path()


class TestEnsureModelFile:
    def test_decompresses_on_demand(self, tmp_path):
        payload = _write_parts(tmp_path)
        det = _detector(tmp_path)
        det.model_path = det._resolve_model_path()
        assert not os.path.exists(det.model_path)

        returned = det._ensure_model_file()

        assert returned == det.model_path
        assert open(det.model_path, "rb").read() == payload

    def test_is_a_no_op_when_the_checkpoint_exists(self, tmp_path):
        target = tmp_path / "rfdetr_nano_person.pt"
        target.write_bytes(b"already here")
        det = _detector(tmp_path)
        det.model_path = str(target)

        det._ensure_model_file()

        assert target.read_bytes() == b"already here"

    def test_raises_when_it_cannot_materialise(self, tmp_path):
        det = _detector(tmp_path)
        det.model_path = os.path.join(str(tmp_path), "rfdetr_nano_person.pt")

        with pytest.raises(FileNotFoundError):
            det._ensure_model_file()
