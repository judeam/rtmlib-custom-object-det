"""Regression tests for download_checkpoint zip handling.

The end2end.onnx search must be confined to the zip's own extraction dir:
a stale end2end.onnx elsewhere under the checkpoints dir (e.g. a Docker
image pre-download left in raw zip layout) used to be picked instead and
renamed to the requested model's filename — serving a 256x192 rtmpose-m
network under the rtmpose-x 384x288 name and breaking every TensorRT
inference with a 192-vs-288 shape mismatch.
"""

import zipfile
from pathlib import Path

import pytest

from rtmlib.tools.file import download_checkpoint


def _make_zip(path: Path, inner_dir: str, content: bytes) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{inner_dir}/end2end.onnx", content)


def test_zip_extraction_ignores_stale_onnx_in_checkpoints(tmp_path):
    # Stale model in raw zip layout, sorts BEFORE the tmp extraction dir
    stale = tmp_path / "20230831" / "rtmpose_onnx" / "rtmpose-m_fake"
    stale.mkdir(parents=True)
    (stale / "end2end.onnx").write_bytes(b"WRONG-MODEL")

    zpath = tmp_path / "rtmpose-x_fake_20230606.zip"
    _make_zip(zpath, "20230831/rtmpose_onnx/rtmpose-x_fake", b"RIGHT-MODEL")

    out = download_checkpoint(
        "file://" + str(zpath), dst_dir=str(tmp_path), filename=zpath.name
    )

    assert out.endswith("rtmpose-x_fake_20230606.onnx")
    assert Path(out).read_bytes() == b"RIGHT-MODEL"


def test_zip_without_onnx_raises(tmp_path):
    zpath = tmp_path / "model.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("deploy.json", "{}")

    with pytest.raises(FileNotFoundError):
        download_checkpoint(
            "file://" + str(zpath), dst_dir=str(tmp_path), filename=zpath.name
        )


def test_existing_onnx_short_circuits(tmp_path):
    onnx = tmp_path / "rtmpose-x_fake_20230606.onnx"
    onnx.write_bytes(b"CACHED")

    out = download_checkpoint(
        "https://example.invalid/rtmpose-x_fake_20230606.zip",
        dst_dir=str(tmp_path),
    )

    assert out == str(onnx)
    assert Path(out).read_bytes() == b"CACHED"
