"""ONNX Runtime inference for the local fine-tuned transformers (sequence classifiers and the event encoder).

Models are exported once with `scripts/export_onnx.py`: `model.onnx` (fp32), `model.fp16.onnx` (fp16), and
`model.int8.onnx` only with `--int8`, each with a `.data` weight sidecar. Callers check `onnx_path()` and fall back to
PyTorch when it returns None. `WHAT_MATTERS_ONNX=0` forces PyTorch. Without an override, fp16 is served when CUDA is
available and fp32 otherwise. `WHAT_MATTERS_ONNX_PRECISION` (fp16, fp32, int8) overrides that choice.
Int8 is lossy on the mdeberta and e5 models (see the export script's parity report), so it is opt-in only.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

FP32_NAME = "model.onnx"
FP16_NAME = "model.fp16.onnx"
INT8_NAME = "model.int8.onnx"
FILE_NAMES = {"fp32": FP32_NAME, "fp16": FP16_NAME, "int8": INT8_NAME}


def _precision_order() -> list[str]:
    forced = os.environ.get("WHAT_MATTERS_ONNX_PRECISION")
    if forced:
        return [forced]
    return ["fp16", "fp32"] if "CUDAExecutionProvider" in _providers() else ["fp32"]


def onnx_path(model_dir: Path | str) -> Path | None:
    """The ONNX file to serve for `model_dir`, or None when PyTorch should be used."""
    if os.environ.get("WHAT_MATTERS_ONNX") == "0":
        return None
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    model_dir = Path(model_dir)
    for precision in _precision_order():
        name = FILE_NAMES.get(precision)
        if name and (model_dir / name).exists():
            return model_dir / name
    return None


def _providers() -> list[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]


def _session_providers() -> list:
    """Providers with the CUDA arena grown per request, not doubled, so several models fit in a small GPU."""
    return [(p, {"arena_extend_strategy": "kSameAsRequested"}) if p == "CUDAExecutionProvider" else p
            for p in _providers()]


@lru_cache(maxsize=None)
def _runtime(path_str: str):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    path = Path(path_str)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(path_str, sess_options=options, providers=_session_providers())
    except Exception as exc:  # e.g. GPU out of memory: serve this model from CPU instead of failing
        if "CUDAExecutionProvider" not in _providers():
            raise
        warnings.warn(f"CUDA init failed for {path.parent.name} ({path.name}), using CPU: {exc}", RuntimeWarning)
        session = ort.InferenceSession(path_str, sess_options=options, providers=["CPUExecutionProvider"])
    return AutoTokenizer.from_pretrained(path.parent), session


def _feeds(session, encoded) -> dict[str, np.ndarray]:
    wanted = {i.name for i in session.get_inputs()}
    return {k: np.asarray(v, dtype=np.int64) for k, v in encoded.items() if k in wanted}


def seq_cls_probs(model_dir: Path | str, first: list[str], second: list[str] | None = None, *,
                  max_length: int, batch_size: int = 64) -> np.ndarray:
    """Softmax class probabilities (n, classes) for single texts or text pairs."""
    path = onnx_path(model_dir)
    if path is None:
        raise RuntimeError(f"No ONNX model in {model_dir}")
    tokenizer, session = _runtime(str(path))
    chunks = []
    for start in range(0, len(first), batch_size):
        b = second[start:start + batch_size] if second is not None else None
        enc = tokenizer(first[start:start + batch_size], b, truncation=True, max_length=max_length,
                        padding=True, return_tensors="np")
        logits = session.run(None, _feeds(session, enc))[0].astype(np.float32)
        logits -= logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        chunks.append(exp / exp.sum(axis=-1, keepdims=True))
    return np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


class OnnxEncoder:
    """Stand-in for SentenceTransformer.encode: mean pooling over the ONNX last hidden state, L2-normalised."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self._dim = int(json.loads((self.model_dir / "config.json").read_text())["hidden_size"])

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 64, normalize_embeddings: bool = True,
               show_progress_bar: bool = False) -> np.ndarray:
        path = onnx_path(self.model_dir)
        if path is None:
            raise RuntimeError(f"No ONNX model in {self.model_dir}")
        tokenizer, session = _runtime(str(path))
        max_length = min(int(tokenizer.model_max_length), 512)
        chunks = []
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start + batch_size], truncation=True, max_length=max_length,
                            padding=True, return_tensors="np")
            hidden = session.run(None, _feeds(session, enc))[0].astype(np.float32)
            mask = enc["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            if normalize_embeddings:
                pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            chunks.append(pooled)
        return np.concatenate(chunks) if chunks else np.zeros((0, self._dim), dtype=np.float32)


def _remove_sidecar(graph: Path) -> None:
    """onnx appends to an existing external-data file, so delete the old weights before re-saving."""
    graph.with_name(graph.name + ".data").unlink(missing_ok=True)


def export(model_dir: Path | str, kind: str, *, quantize: bool = False, fp16: bool = True, opset: int = 17) -> dict[str, Path]:
    """Export a local PyTorch model to ONNX. `kind` is "seq_cls" or "encoder"."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if kind == "seq_cls":
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).eval()
    elif kind == "encoder":
        model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    else:
        raise ValueError(f"unknown kind: {kind}")

    probe = tokenizer(["export probe text"], ["second segment"] if kind == "seq_cls" else None,
                      return_tensors="pt")
    names = [n for n in ("input_ids", "token_type_ids", "attention_mask") if n in probe]
    args = tuple(probe[n] for n in names)
    axes = {n: {0: "batch", 1: "seq"} for n in names}
    out_axes = {0: "batch"} if kind == "seq_cls" else {0: "batch", 1: "seq"}

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, *inputs):
            out = self.model(**dict(zip(names, inputs)))
            return out.logits if kind == "seq_cls" else out.last_hidden_state

    import onnx

    fp32 = model_dir / FP32_NAME
    # The exporter scatters weights into loose files; export to a temp dir, then re-save the graph with all
    # weights in one `model.onnx.data` sidecar (a single file, and required for graphs over the 2 GB protobuf limit).
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        exported = Path(tmp) / FP32_NAME
        torch.onnx.export(Wrapper(), args, str(exported), input_names=names, output_names=["out"],
                          dynamic_axes={**axes, "out": out_axes}, opset_version=opset, dynamo=False)
        _remove_sidecar(fp32)
        onnx.save(onnx.load(str(exported), load_external_data=True), str(fp32), save_as_external_data=True,
                  all_tensors_to_one_file=True, location=f"{FP32_NAME}.data")
    paths = {"fp32": fp32}
    if fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        # disable_shape_infer: the shape pass serializes the whole model, which fails above the 2 GB protobuf limit.
        half = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True, disable_shape_infer=True)
        _remove_sidecar(model_dir / FP16_NAME)
        onnx.save(half, str(model_dir / FP16_NAME), save_as_external_data=True, all_tensors_to_one_file=True,
                  location=f"{FP16_NAME}.data")
        paths["fp16"] = model_dir / FP16_NAME
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8 = model_dir / INT8_NAME
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True)
        paths["int8"] = int8
    return paths
