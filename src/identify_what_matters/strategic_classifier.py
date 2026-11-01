"""
Strategic Intelligence Neural Classifier (v2 Robust Multi-Task Architecture)
Wraps fine-tuned MultiTaskStrategicClassifier (mDeBERTa-v3) trained on A100 GPU VM to provide:
1. Threat / Friction Axis Classification (8 classes)
2. Criticality Level Prediction (Low, Medium, High, Critical)
3. Authenticity / Astroturf Detection (Organic, Astroturf, Bot Spam)
4. Trajectory Stage Classification (pre_breakout, breaking_out, peak_saturation, simmering_entropy, settling_quiet)
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model, ContextPooler

THREAT_AXES = [
    "institutional_integrity",
    "public_safety_crisis",
    "economic_distress",
    "youth_campus_agitation",
    "communal_regional_faultline",
    "infrastructure_governance_failure",
    "disinformation_rumor_panic",
    "general_civic_deliberation"
]
ID2AXIS = {i: a for i, a in enumerate(THREAT_AXES)}

CRITICALITY_LEVELS = ["low", "medium", "high", "critical"]
ID2CRIT = {i: c for i, c in enumerate(CRITICALITY_LEVELS)}

AUTHENTICITY_TYPES = ["organic", "astroturf", "bot_spam"]
ID2AUTH = {i: a for i, a in enumerate(AUTHENTICITY_TYPES)}

TRAJECTORY_STAGES = [
    "pre_breakout",
    "breaking_out",
    "peak_saturation",
    "simmering_entropy",
    "settling_quiet"
]
ID2STAGE = {i: s for i, s in enumerate(TRAJECTORY_STAGES)}

_MODEL_CACHE = {}

class MultiTaskStrategicClassifier(nn.Module):
    def __init__(self, model_dir: Path):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir)
        self.deberta = DebertaV2Model.from_pretrained(model_dir, config=self.config, torch_dtype=torch.float32)
        self.pooler = ContextPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        hidden_size = self.config.hidden_size
        self.axis_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(THREAT_AXES))
        )
        self.crit_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(CRITICALITY_LEVELS))
        )
        self.auth_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(AUTHENTICITY_TYPES))
        )
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(TRAJECTORY_STAGES))
        )

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(encoder_outputs[0])
        pooled = self.dropout(pooled)

        axis_logits = self.axis_classifier(pooled)
        crit_logits = self.crit_classifier(pooled)
        auth_logits = self.auth_classifier(pooled)
        stage_logits = self.stage_classifier(pooled)

        return axis_logits, crit_logits, auth_logits, stage_logits

def load_strategic_model(model_dir: Optional[Union[str, Path]] = None):
    global _MODEL_CACHE
    if model_dir is None:
        v4_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "strategic_models" / "strategic_mdeberta_v4"
        v3_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "strategic_models" / "strategic_mdeberta_v3"
        v2_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "strategic_models" / "strategic_mdeberta_v2"
        v1_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "strategic_models" / "strategic_mdeberta_v1"
        if v4_dir.exists() and (v4_dir / "strategic_heads.pt").exists():
            model_dir = v4_dir
        elif v3_dir.exists() and (v3_dir / "strategic_heads.pt").exists():
            model_dir = v3_dir
        elif v2_dir.exists() and (v2_dir / "strategic_heads.pt").exists():
            model_dir = v2_dir
        else:
            model_dir = v1_dir

    model_dir = Path(model_dir).resolve()
    cache_key = str(model_dir)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not model_dir.exists() or not (model_dir / "strategic_heads.pt").exists():
        return None, None, None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            free_mem = torch.cuda.mem_get_info()[0]
            if free_mem >= 2 * 1024 * 1024 * 1024:
                device = torch.device("cuda")
        except Exception:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = MultiTaskStrategicClassifier(model_dir)
        heads_state = torch.load(model_dir / "strategic_heads.pt", map_location="cpu", weights_only=False)
        
        # Check if full model state dict or separate heads dict
        if "deberta.embeddings.word_embeddings.weight" in heads_state:
            model.load_state_dict(heads_state)
        else:
            model.pooler.load_state_dict(heads_state["pooler"])
            model.axis_classifier.load_state_dict(heads_state["axis_classifier"])
            model.crit_classifier.load_state_dict(heads_state["crit_classifier"])
            model.auth_classifier.load_state_dict(heads_state["auth_classifier"])
            model.stage_classifier.load_state_dict(heads_state["stage_classifier"])

        try:
            model.to(device)
        except Exception:
            device = torch.device("cpu")
            model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = (model, tokenizer, device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Warning: Failed to load Strategic Neural Classifier from {model_dir}: {e}")
        return None, None, None

def classify_strategic(text: str, max_length: int = 128) -> Dict[str, Any]:
    """
    Classifies a post, comment, or headline across all 4 strategic dimensions:
    - threat_axis: Name and confidence distribution
    - criticality: Level (low/medium/high/critical) and risk score [0..1]
    - authenticity: organic vs astroturf vs bot_spam
    - trajectory_stage: Stage prediction
    """
    model, tokenizer, device = load_strategic_model()
    if model is None:
        return {
            "threat_axis": "general_civic_deliberation",
            "threat_axis_conf": 0.5,
            "criticality": "medium",
            "criticality_score": 0.5,
            "authenticity": "organic",
            "astroturf_prob": 0.1,
            "trajectory_stage": "simmering_entropy",
            "trajectory_stage_conf": 0.5
        }

    enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        a_l, c_l, au_l, s_l = model(input_ids, attention_mask)
        a_p = torch.softmax(a_l, dim=-1).squeeze(0).cpu().numpy()
        c_p = torch.softmax(c_l, dim=-1).squeeze(0).cpu().numpy()
        au_p = torch.softmax(au_l, dim=-1).squeeze(0).cpu().numpy()
        s_p = torch.softmax(s_l, dim=-1).squeeze(0).cpu().numpy()

    best_a_id = int(a_p.argmax())
    best_c_id = int(c_p.argmax())
    best_au_id = int(au_p.argmax())
    best_s_id = int(s_p.argmax())

    crit_weights = [0.1, 0.4, 0.75, 1.0]
    crit_score = float(sum(p * w for p, w in zip(c_p, crit_weights)))

    return {
        "threat_axis": ID2AXIS[best_a_id],
        "threat_axis_conf": float(a_p[best_a_id]),
        "criticality": ID2CRIT[best_c_id],
        "criticality_score": round(crit_score, 3),
        "authenticity": ID2AUTH[best_au_id],
        "astroturf_prob": round(float(au_p[1] + au_p[2]), 3),
        "trajectory_stage": ID2STAGE[best_s_id],
        "trajectory_stage_conf": float(s_p[best_s_id])
    }

def classify_strategic_batch(texts: List[str], max_length: int = 128, batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Batched inference for strategic neural classifier to maximize GPU/MPS throughput.
    """
    model, tokenizer, device = load_strategic_model()
    if model is None or not texts:
        return [classify_strategic(t, max_length) for t in texts]

    results = []
    crit_weights = [0.1, 0.4, 0.75, 1.0]

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            a_l, c_l, au_l, s_l = model(input_ids, attention_mask)
            a_p = torch.softmax(a_l, dim=-1).cpu().numpy()
            c_p = torch.softmax(c_l, dim=-1).cpu().numpy()
            au_p = torch.softmax(au_l, dim=-1).cpu().numpy()
            s_p = torch.softmax(s_l, dim=-1).cpu().numpy()

        for j in range(len(chunk)):
            best_a = int(a_p[j].argmax())
            best_c = int(c_p[j].argmax())
            best_au = int(au_p[j].argmax())
            best_s = int(s_p[j].argmax())
            crit_s = float(sum(p * w for p, w in zip(c_p[j], crit_weights)))

            results.append({
                "threat_axis": ID2AXIS[best_a],
                "threat_axis_conf": float(a_p[j][best_a]),
                "criticality": ID2CRIT[best_c],
                "criticality_score": round(crit_s, 3),
                "authenticity": ID2AUTH[best_au],
                "astroturf_prob": round(float(au_p[j][1] + au_p[j][2]), 3),
                "trajectory_stage": ID2STAGE[best_s],
                "trajectory_stage_conf": float(s_p[j][best_s])
            })

    return results
