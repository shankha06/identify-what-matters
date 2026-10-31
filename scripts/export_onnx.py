"""Export the local fine-tuned transformers to ONNX (fp32 + fp16, optionally int8) and check parity against PyTorch.

Usage: uv run python scripts/export_onnx.py [--int8] [--only stance tone political encoder]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identify_what_matters import onnx_backend  # noqa: E402
from identify_what_matters.encoder import MODEL_CANDIDATES  # noqa: E402

STANCE_ROOT = ROOT / "artifacts" / "stance-models"
POLITICAL_ROOT = ROOT / "artifacts" / "political_models"
PROBES = [
    "Modi government is doing great work on roads and railways",
    "This scheme is just a publicity stunt, nothing reaches the farmers",
    "Lol the opposition meme of the day 😂",
    "कांग्रेस ने फिर से वादे तोड़े हैं",
    "Election commission should act on the voter roll issue",
    "Fact check: this video is from 2019, not from the recent protest",
]


def _sessions(model_dir: Path) -> dict[str, object]:
    """One session per exported precision. fp16 runs on the serving provider (GPU when present, to fit VRAM).
    fp32 and int8 are references, so they run on CPU to keep GPU memory free for fp16."""
    import onnxruntime as ort
    out = {}
    for label, name in onnx_backend.FILE_NAMES.items():
        path = model_dir / name
        if path.exists():
            providers = onnx_backend._session_providers() if label == "fp16" else ["CPUExecutionProvider"]
            out[label] = ort.InferenceSession(str(path), providers=providers)
    return out


def _check_seq_cls(model_dir: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).eval()
    enc = tok(PROBES, truncation=True, max_length=160, padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = torch.softmax(model(**enc).logits.float(), dim=-1).numpy()
    for label, sess in _sessions(model_dir).items():
        wanted = {i.name for i in sess.get_inputs()}
        feeds = {k: v.numpy() for k, v in enc.items() if k in wanted}
        logits = sess.run(None, feeds)[0]
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        agree = float((probs.argmax(-1) == ref.argmax(-1)).mean())
        print(f"  {label}: max|dp|={np.abs(probs - ref).max():.5f} argmax_agree={agree:.2f}")


def _check_encoder(model_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer
    ref = SentenceTransformer(str(model_dir)).encode(PROBES, normalize_embeddings=True)
    for label in onnx_backend.FILE_NAMES:
        if not (model_dir / onnx_backend.FILE_NAMES[label]).exists():
            continue
        os.environ["WHAT_MATTERS_ONNX_PRECISION"] = label
        vecs = onnx_backend.OnnxEncoder(model_dir).encode(PROBES)
        cos = (vecs * ref).sum(-1)
        print(f"  {label}: min cosine vs torch={cos.min():.5f}")
        os.environ.pop("WHAT_MATTERS_ONNX_PRECISION", None)
        onnx_backend._runtime.cache_clear()
        gc.collect()


def _timing(model_dir: Path, kind: str) -> None:
    if kind == "seq_cls":
        run = lambda: onnx_backend.seq_cls_probs(model_dir, PROBES * 10, max_length=160)  # noqa: E731
    else:
        run = lambda: onnx_backend.OnnxEncoder(model_dir).encode(PROBES * 10)  # noqa: E731
    run()
    start = time.perf_counter()
    run()
    print(f"  onnx serving: {(time.perf_counter() - start) * 1000:.0f} ms for 60 texts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", action="store_true", help="also write int8 files (lossy on mdeberta/e5)")
    parser.add_argument("--only", nargs="*", default=None, help="subset of: stance tone political encoder")
    args = parser.parse_args()
    wanted = set(args.only) if args.only else {"stance", "tone", "political", "encoder"}

    jobs: list[tuple[str, Path, str]] = []
    if "stance" in wanted:
        jobs.append(("stance", STANCE_ROOT / "stance", "seq_cls"))
    if "tone" in wanted:
        jobs.append(("tone", STANCE_ROOT / "tone", "seq_cls"))
    if "political" in wanted:
        jobs += [(d.name, d, "seq_cls") for d in sorted(POLITICAL_ROOT.glob("mdeberta-*")) if (d / "config.json").exists()]
    if "encoder" in wanted:
        encoder_dir = next((ROOT / c for c in MODEL_CANDIDATES if (ROOT / c).exists()), None)
        if encoder_dir is not None:
            jobs.append((encoder_dir.name, encoder_dir, "encoder"))

    for name, model_dir, kind in jobs:
        print(f"[{name}] exporting {model_dir.relative_to(ROOT)}")
        paths = onnx_backend.export(model_dir, kind, quantize=args.int8)
        for label, path in paths.items():
            print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.0f} MB, {label})")
        if kind == "seq_cls":
            _check_seq_cls(model_dir)
        else:
            _check_encoder(model_dir)
        _timing(model_dir, kind)
        # Drop cached sessions so GPU memory from this model does not stay allocated for the next one.
        onnx_backend._runtime.cache_clear()
        gc.collect()


if __name__ == "__main__":
    main()
